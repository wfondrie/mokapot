"""
Handle proteins for the picked protein FDR.
"""
import re
import logging
from collections import defaultdict

from .utils import tuplize

LOGGER = logging.getLogger(__name__)


# Functions -------------------------------------------------------------------
def read_fasta(
    fasta_files,
    enzyme_regex="[KR]",
    missed_cleavages=0,
    min_length=6,
    max_length=50,
    semi=False,
    decoy_prefix="decoy_",
):
    """
    Parse a FASTA file into a dictionary.

    Parameters
    ----------
    fasta_files : str
        The FASTA file to parse.
    enzyme_regex : str or compiled regex, optional
        A regular expression defining the enzyme specificity.
    missed_cleavages : int, optional
        The maximum number of allowed missed cleavages.
    min_length : int, optional
        The minimum peptide length.
    max_length : int, optional
        The maximum peptide length.
    semi : bool
        Allow semi-enzymatic cleavage.
    decoy_prefix : str
        The prefix used to indicate decoy sequences.

    Returns
    -------
    unique_peptides : dict
        A dictionary matching unique peptides to proteins.
    decoy_map : dict
        A dictionary decoy proteins to their corresponding target proteins.
    """
    if isinstance(enzyme_regex, str):
        enzyme_regex = re.compile(enzyme_regex)

    fasta_files = tuplize(fasta_files)
    # Read in the fasta files
    LOGGER.info("Parsing FASTA files and digesting proteins...")
    fasta = []
    for fasta_file in fasta_files:
        with open(fasta_file) as fa:
            fasta.append(fa.read())

    fasta = "\n".join(fasta)[1:].split("\n>")

    # Build the initial mapping
    proteins = {}
    peptides = defaultdict(set)
    for entry in fasta:
        entry = entry.split("\n", 1)
        prot = entry[0].split(" ")[0]
        seq = entry[1].replace("\n", "")

        peps = digest(
            seq,
            enzyme_regex=enzyme_regex,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            semi=semi,
        )

        if peps:
            proteins[prot] = peps
            for pep in peps:
                peptides[pep].add(prot)

    total_prots = len(fasta)
    LOGGER.info("\t- Parsed and digested %i proteins.", total_prots)
    LOGGER.info("\t- %i had no peptides.", len(fasta) - len(proteins))
    LOGGER.info("\t- Retained %i proteins.", len(proteins))
    del fasta

    # Sort proteins by number of peptides:
    proteins = {
        k: v for k, v in sorted(proteins.items(), key=lambda i: len(i[1]))
    }

    LOGGER.info("Matching target to decoy proteins...")
    # Build the decoy map:
    decoy_map = {}
    no_decoys = 0
    for prot_name in proteins:
        if not prot_name.startswith(decoy_prefix):
            decoy = decoy_prefix + prot_name
            if decoy in proteins.keys():
                decoy_map[prot_name] = decoy
            else:
                no_decoys += 1

    if no_decoys:
        LOGGER.warning(
            "Found %i target proteins without matching decoys.", no_decoys
        )

    LOGGER.info("Building protein groups...")
    # Group Proteins
    num_before_group = len(proteins)
    proteins, peptides = _group_proteins(proteins, peptides)
    LOGGER.info(
        "\t -Aggregated %i proteins into %i protein groups.",
        num_before_group,
        len(proteins),
    )

    # unique peptides:
    LOGGER.info("Discarding shared peptides...")
    unique_peptides = {
        k: next(iter(v)) for k, v in peptides.items() if len(v) == 1
    }
    total_proteins = len(set(p for p in unique_peptides.values()))

    LOGGER.info(
        "\t- Discarded %i peptides and %i proteins groups.",
        len(peptides) - len(unique_peptides),
        len(proteins) - total_proteins,
    )
    LOGGER.info(
        "\t- Retained %i peptides from %i protein groups.",
        len(unique_peptides),
        total_proteins,
    )

    return unique_peptides, decoy_map


def digest(
    sequence,
    enzyme_regex="[KR]",
    missed_cleavages=0,
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
    if isinstance(enzyme_regex, str):
        enzyme_regex = re.compile(enzyme_regex)

    # Find the cleavage sites
    sites = (
        [0]
        + [m.end() for m in enzyme_regex.finditer(sequence)]
        + [len(sequence)]
    )

    peptides = _cleave(
        sequence=sequence,
        sites=sites,
        missed_cleavages=missed_cleavages,
        min_length=min_length,
        max_length=max_length,
        semi=semi,
    )

    return peptides


# Private Functions -----------------------------------------------------------
def _cleave(sequence, sites, missed_cleavages, min_length, max_length, semi):
    """
    Digest a protein sequence into its constituent peptides.

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
    """
    Group proteins when one's peptides are a subset of another's.

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
    for prot, peps in proteins.items():
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
                peptides[pep].add(new_prot)

    return grouped, peptides
