"""The code for parsing FASTA files"""
import re
import logging
from textwrap import wrap
from collections import defaultdict

import numpy as np

from ..utils import tuplize
from ..proteins import Proteins

LOGGER = logging.getLogger(__name__)


def read_fasta(
    fasta_files,
    enzyme="[KR]",
    missed_cleavages=2,
    clip_nterm_methionine=False,
    min_length=6,
    max_length=50,
    semi=False,
    decoy_prefix="decoy_",
):
    """Parse a FASTA file, storing a mapping of peptides and proteins.

    Protein sequence information from the FASTA file is required to compute
    protein-level confidence estimates using the picked-protein approach.
    Decoys proteins must be included and must be of the have a description in
    format of `<prefix><protein ID>` for valid confidence estimates to be
    calculated.

    If you need to generate an appropriate FASTA file with decoy sequences for
    your database search, see :py:func:`mokapot.make_decoys()`.

    Importantly, the parameters below should match the conditions in which the
    PSMs were assigned as closely as possible. Enzyme specificity is provided
    using a regular expression. A table of common enzymes can be found here in
    the mokapot `cookbook
    <file:///Users/wfondrie/packages/mokapot/docs/build/html/cookbook.html#enzyme-regular-expressions>`_.

    Parameters
    ----------
    fasta_files : str or tuple of str
        The FASTA file(s) used for assigning the PSMs
    decoy_prefix : str, optional
        The prefix used to indicate a decoy protein in the description
        lines of the FASTA file.
    enzyme : str or compiled regex, optional
        A regular expression defining the enzyme specificity was used when
        assigning PSMs. The cleavage site is interpreted as the end of the
        match. The default is trypsin, without proline suppression: "[KR]".
    missed_cleavages : int, optional
        The allowed number of missed cleavages.
    clip_nterm_methionine : bool, optional
        Remove methionine residues that occur at the protein N-terminus.
    min_length : int, optional
        The minimum peptide length to consider.
    max_length : int, optional
        The maximum peptide length to consider.
    semi : bool, optional
        Was a semi-enzymatic digest used to assign PSMs? If :code:`True`, the
        protein database will likely contain many shared peptides and yield
        unhelpful protein-level confidence estimates.

    Returns
    -------
    Proteins object
        The parsed proteins as a :py:class:`~mokapot.proteins.Proteins`
        object.

    """
    if isinstance(enzyme, str):
        enzyme_regex = re.compile(enzyme)
    else:
        enzyme_regex = enzyme

    # Read in the fasta files
    LOGGER.info("Parsing FASTA files and digesting proteins...")
    fasta = _parse_fasta_files(fasta_files)
    # Build the initial mapping
    proteins = {}
    peptides = defaultdict(set)
    for entry in fasta:
        prot, seq = _parse_protein(entry)

        peps = digest(
            seq,
            enzyme_regex=enzyme_regex,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            semi=semi,
            clip_nterm_methionine=clip_nterm_methionine,
        )

        if peps:
            proteins[prot] = peps
            for pep in peps:
                peptides[pep].add(prot)

    total_prots = len(fasta)
    LOGGER.info("  - Parsed and digested %i proteins.", total_prots)
    LOGGER.info("  - %i had no peptides.", len(fasta) - len(proteins))
    LOGGER.info("  - Retained %i proteins.", len(proteins))
    del fasta

    # Sort proteins by number of peptides:
    proteins = {
        k: v for k, v in sorted(proteins.items(), key=lambda i: len(i[1]))
    }

    LOGGER.info("Matching target to decoy proteins...")
    # Build the decoy map:
    decoy_map = {}
    no_decoys = 0
    has_decoys = False
    has_targets = False
    for prot_name in proteins:
        if not prot_name.startswith(decoy_prefix):
            has_targets = True
            decoy = decoy_prefix + prot_name
            decoy_map[prot_name] = decoy
            if decoy in proteins.keys():
                has_decoys = True
            else:
                no_decoys += 1

    if not has_targets:
        raise ValueError("Only decoy proteins were found in the FASTA file.")

    if no_decoys and no_decoys < len(decoy_map):
        LOGGER.warning(
            "Found %i target proteins without matching decoys.", no_decoys
        )

    LOGGER.info("Building protein groups...")
    # Group Proteins
    num_before_group = len(proteins)
    proteins, peptides = _group_proteins(proteins, peptides)
    LOGGER.info(
        "\t- Aggregated %i proteins into %i protein groups.",
        num_before_group,
        len(proteins),
    )

    if not has_decoys:
        LOGGER.info("No decoy sequences were found in the FASTA file.")
        LOGGER.info(
            "  - Creating decoy protein groups that mirror the target "
            "proteins."
        )

    # unique peptides:
    LOGGER.info("Discarding shared peptides...")
    shared_peptides = {}
    unique_peptides = {}
    for pep, prots in peptides.items():
        if len(prots) == 1:
            unique_peptides[pep] = next(iter(prots))
        else:
            shared_peptides[pep] = "; ".join(prots)

    total_proteins = len(set(p for p in unique_peptides.values()))

    LOGGER.info(
        "  - Discarded %i peptides and %i proteins groups.",
        len(peptides) - len(unique_peptides),
        len(proteins) - total_proteins,
    )

    LOGGER.info(
        "  - Retained %i peptides from %i protein groups.",
        len(unique_peptides),
        total_proteins,
    )

    parsed = Proteins(
        decoy_prefix=decoy_prefix,
        peptide_map=unique_peptides,
        shared_peptides=shared_peptides,
        protein_map=decoy_map,
        has_decoys=has_decoys,
    )

    return parsed


def make_decoys(
    fasta,
    out_file,
    decoy_prefix="decoy_",
    enzyme="[KR]",
    reverse=False,
    concatenate=True,
):
    """
    Create a FASTA file with decoy sequences.

    Decoy sequences are generated by shuffling or reversing each
    enzymatic peptide in a sequence, preserving the first and
    last amino acids.

    Parameters
    ----------
    fasta : str or list of str
        One or more FASTA files containing target sequences.
    out_file : str
        The name of the output FASTA file.
    enzyme : str or compiled regex, optional
        A regular expression defining the enzyme specificity was used
        when assigning PSMs. The cleavage site is interpreted as the
        end of the match. The default is trypsin, without proline
        suppression: "[KR]".
    decoy_prefix : str, optional
        The prefix used to indicate a decoy protein.
    reverse : bool, optional
        Use reversed instead of shuffled sequences? Note that the
        difference here is arbitrary, because reversing can be thought
        of as a specific instance of shuffling.
    concatenate : bool, optional
        Concatenate decoy sequences to the provided target sequences?
        :code:`True` creates a FASTA file with target and decoy sequences;
        :code:`False` creates a FASTA file with only decoy sequences.

    Returns
    -------
    str
        The output FASTA file.
    """
    LOGGER.info("Parsing FASTA file(s)...")
    proteins = _parse_fasta_files(fasta)
    proteins = [_parse_protein(p) for p in proteins]

    rev_msg = {True: "Reversing", False: "Shuffling"}
    LOGGER.info("%s peptides in proteins...", rev_msg[reverse])
    decoys = _shuffle_proteins(proteins, decoy_prefix, enzyme, reverse)

    if concatenate:
        proteins += decoys
    else:
        proteins = decoys

    con_msg = {True: " target and", False: ""}
    LOGGER.info(
        "Writing%s decoy proteins to %s...", con_msg[concatenate], out_file
    )
    fasta = []
    for prot, seq in proteins:
        seq = "\n".join(wrap(seq))
        prot = ">" + prot
        fasta.append("\n".join([prot, seq]))

    fasta = "\n".join(fasta)

    with open(out_file, "w+") as out:
        out.write(fasta)

    return out_file


def digest(
    sequence,
    enzyme_regex="[KR]",
    missed_cleavages=0,
    clip_nterm_methionine=False,
    min_length=6,
    max_length=50,
    semi=False,
):
    """
    Digest a protein sequence into its constituent peptides.

    Parameters
    ----------
    sequence : str
        A protein sequence to digest.
    enzyme_regex : str or compiled regex, optional
        A regular expression defining the enzyme specificity. The end of the
        match should indicate the cleavage site.
    missed_cleavages : int, optional
        The maximum number of allowed missed cleavages.
    clip_nterm_methionine : bool, optional
        Remove methionine residues that occur at the protein N-terminus.
    min_length : int, optional
        The minimum peptide length.
    max_length : int, optional
        The maximum peptide length.
    semi : bool
        Allow semi-enzymatic cleavage.

    Returns
    -------
    peptides : set of str
        The peptides resulting from the digested sequence.
    """
    sites = _cleavage_sites(sequence, enzyme_regex)
    peptides = _cleave(
        sequence=sequence,
        sites=sites,
        missed_cleavages=missed_cleavages,
        min_length=min_length,
        max_length=max_length,
        semi=semi,
        clip_nterm_met=clip_nterm_methionine,
    )

    return peptides


# Private Functions -----------------------------------------------------------
def _parse_fasta_files(fasta_files):
    """Read a fasta file and divide into proteins

    Parameters
    ----------
    fasta_files : str or list of str
        One or more FASTA files.

    Returns
    -------
    proteins : list of str
        The raw protein headers and sequences.
    """
    fasta_files = tuplize(fasta_files)
    fasta = []
    for fasta_file in fasta_files:
        with open(fasta_file) as fa:
            fasta.append(fa.read())

    return "\n".join(fasta)[1:].split("\n>")


def _parse_protein(raw_protein):
    """Parse the raw string for a protein.

    Parameters
    ----------
    raw_protein : str
        The raw protein string.

    Returns
    -------
    header : str
        The protein name.
    sequence : str
        The protein sequence.
    """
    entry = raw_protein.splitlines()
    prot = entry[0].split(" ")[0]
    if len(entry) == 1:
        logging.warning("No sequence was detected for %s.", prot)
        return prot, ""

    seq = "".join(entry[1:])
    return prot, seq


def _shuffle_proteins(proteins, decoy_prefix, enzyme, reverse):
    """Shuffle protein sequences

    Parameters
    ----------
    proteins : list of list of str
        The proteins to shuffle.
    decoy_prefix : str
        The prefix indicating a decoy protein.
    enzyme : str or compiled regex
        The enzyme specificity to use.
    reverse : bool
        Reverse instead?

    Returns
    -------
    decoy_proteins : list of list of str
        The decoy proteins.
    """
    decoys = []
    perms = {}
    for prot, seq in proteins:
        decoy_prot = decoy_prefix + prot
        sites = _cleavage_sites(seq, enzyme)
        new_seq = list(seq)
        for start_idx, cleavage_site in enumerate(sites):
            end_idx = start_idx + 1
            if end_idx >= len(sites):
                continue

            # Keep the first and last AA the fixed:
            start = cleavage_site + 1
            end = sites[end_idx] - 1
            pep_len = end - start

            if pep_len <= 1:
                continue

            # Make permutations:
            if pep_len not in perms.keys():
                if reverse:
                    perms[pep_len] = np.flip(np.arange(pep_len))
                else:
                    base = np.arange(pep_len)
                    perm = base
                    tries = 0
                    while tries < 100 and np.array_equal(base, perm):
                        perm = np.random.permutation(base)
                        tries += 1

                    perms[pep_len] = perm

            new_seq[start:end] = [new_seq[i + start] for i in perms[pep_len]]

        decoys.append([decoy_prot, "".join(new_seq)])

    return decoys


def _cleavage_sites(sequence, enzyme_regex):
    """Find the cleavage sites in a sequence.

    Parameters
    ----------
    sequence : str
        A protein sequence to digest.
    enzyme_regex : str or compiled regex
        A regular expression defining the enzyme specificity.

    Returns
    -------
    sites : list of int
        The cleavage sites in the sequence.
    """
    if isinstance(enzyme_regex, str):
        enzyme_regex = re.compile(enzyme_regex)

    # Find the cleavage sites
    sites = (
        [0]
        + [m.end() for m in enzyme_regex.finditer(sequence)]
        + [len(sequence)]
    )
    return sites


def _cleave(
    sequence,
    sites,
    missed_cleavages,
    min_length,
    max_length,
    semi,
    clip_nterm_met,
):
    """Digest a protein sequence into its constituent peptides.

    Parameters
    ----------
    sequence : str
        A protein sequence to digest.
    sites : list of int
        The cleavage sites.
    missed_cleavages : int, optional
        The maximum number of allowed missed cleavages.
    min_length : int, optional
        The minimum peptide length.
    max_length : int, optional
        The maximum peptide length.
    semi : bool
        Allow semi-enzymatic cleavage.
    clip_nterm_met : bool
        Clip the N-Terminal methionine

    Returns
    -------
    peptides : set of str
        The peptides resulting from the digested sequence.
    """
    peptides = set()

    # Do the digest
    for start_idx, start_site in enumerate(sites):
        for diff_idx in range(1, missed_cleavages + 2):
            end_idx = start_idx + diff_idx
            if end_idx >= len(sites):
                continue

            end_site = sites[end_idx]
            peptide = sequence[start_site:end_site]
            if len(peptide) < min_length or len(peptide) > max_length:
                continue

            peptides.add(peptide)

            if clip_nterm_met and not start_idx and peptide.startswith("M"):
                if len(peptide[1:]) >= min_length:
                    peptides.add(peptide[1:])

            # Handle semi:
            if semi:
                for idx in range(1, len(peptide)):
                    sub_pep_len = len(peptide) - idx
                    if sub_pep_len < min_length:
                        break

                    if sub_pep_len > max_length:
                        continue

                    semi_pep = {peptide[idx:], peptide[:-idx]}
                    peptides = peptides.union(semi_pep)

    return peptides


def _group_proteins(proteins, peptides):
    """Group proteins when one's peptides are a subset of another's.

    WARNING: This function directly modifies `peptides` for the sake of
    memory.

    Parameters
    ----------
    proteins : dict[str, set of str]
        A map of proteins to their peptides
    peptides : dict[str, set of str]
        A map of peptides to their proteins

    Returns
    -------
    protein groups : dict[str, set of str]
        A map of protein groups to their peptides
    peptides : dict[str, set of str]
        A map of peptides to their protein groups.
    """
    grouped = {}
    for prot, peps in sorted(proteins.items(), key=lambda x: -len(x[1])):
        if not grouped:
            grouped[prot] = peps
            continue

        matches = set.intersection(*[peptides[p] for p in peps])
        matches = [m for m in matches if m in grouped.keys()]

        # If the entry is unique:
        if not matches:
            grouped[prot] = peps
            continue

        # Create new entries from subsets:
        for match in matches:
            new_prot = ", ".join([match, prot])
            # Update grouped proteins:
            grouped[new_prot] = grouped.pop(match)

            # Update peptides:
            for pep in grouped[new_prot]:
                peptides[pep].remove(match)
                if prot in peptides[pep]:
                    peptides[pep].remove(prot)

                peptides[pep].add(new_prot)

    return grouped, peptides
