"""
Implementation of the picked-protein approach for protein-level
confidence estimates.
"""
import logging
import numpy as np
import pandas as pd

from .peptides import match_decoy
from . import utils

LOGGER = logging.getLogger(__name__)


def picked_protein(
    peptides, target_column, peptide_column, score_column, proteins
):
    """Perform the picked-protein approach

    Parameters
    ----------
    peptides : pandas.DataFrame
        A dataframe of the peptides.
    target_column : str
        The column in `peptides` indicating if the peptide is a target.
    peptide_column : str
        The column in `peptides` containing the peptide sequence.
    proteins : FastaProteins object
        A FastaProteins object.

    Returns
    -------
    pandas.DataFrame
        The aggregated proteins for confidence estimation.
    """
    keep = [target_column, peptide_column, score_column]

    # Trim the dataframe
    prots = peptides.loc[:, keep].rename(
        columns={peptide_column: "best peptide"}
    )

    # Strip modifications and flanking AA's from peptide sequences.
    prots["stripped sequence"] = (
        prots["best peptide"]
        .str.replace(r"[\[\(].*?[\]\)]", "")
        .str.replace(r"^.*?\.", "")
        .str.replace(r"\..*?$", "")
    )

    # Sometimes folks use lowercase letters for the termini or mods:
    if all(prots["stripped sequence"].str.islower()):
        seqs = prots["stripped sequence"].upper()
    else:
        seqs = prots["stripped sequence"].str.replace(r"[a-z]", "")

    prots["stripped sequence"] = seqs

    # There are two cases we need to deal with:
    # 1. The fasta contained both targets and decoys (ideal)
    # 2. The fasta contained only targets (less ideal)
    if proteins.has_decoys:
        prots["mokapot protein group"] = group_with_decoys(prots, proteins)

    else:
        LOGGER.info("Mapping decoy peptides to protein groups...")
        prots["mokapot protein group"] = group_without_decoys(
            prots, target_column, proteins
        )

    # Verify that unmatched peptides are shared:
    unmatched = pd.isna(prots["mokapot protein group"])
    if not proteins.has_decoys:
        unmatched[~prots[target_column]] = False

    unmatched_prots = prots.loc[unmatched, :]
    shared = unmatched_prots["stripped sequence"].isin(
        proteins.shared_peptides
    )

    shared_unmatched = (~shared).sum()
    num_shared = len(shared) - shared_unmatched
    LOGGER.debug(
        "%i out of %i peptides were discarded as shared peptides.",
        num_shared,
        len(prots),
    )

    if shared_unmatched:
        LOGGER.debug("%s", unmatched_prots.loc[~shared, "stripped sequence"])
        if shared_unmatched / len(prots) > 0.10:
            raise ValueError(
                "Fewer than 90% of all peptides could be matched to proteins. "
                "Verify that your digest settings are correct."
            )

        LOGGER.warning(
            "%i out of %i peptides could not be mapped."
            "Check your digest settings.",
            shared_unmatched,
            len(prots),
        )

    # Verify that reasonable number of decoys were matched.
    if proteins.has_decoys:
        num_unmatched_decoys = unmatched_prots[target_column][~shared].sum()
        total_decoys = (~prots[target_column]).sum()
        if num_unmatched_decoys / total_decoys > 0.05:
            raise ValueError(
                "Fewer than 5% of decoy peptides could be mapped to proteins."
                " Was the correct FASTA file and digest settings used?"
            )

    prots = prots.loc[~unmatched, :]
    prots["decoy"] = (
        prots["mokapot protein group"]
        .str.split(",", expand=True)[0]
        .map(lambda x: proteins.protein_map.get(x, x))
    )

    prot_idx = utils.groupby_max(prots, ["decoy"], score_column)
    final_cols = [
        "mokapot protein group",
        "best peptide",
        "stripped sequence",
        score_column,
        target_column,
    ]
    return prots.loc[prot_idx, final_cols]


def group_with_decoys(peptides, proteins):
    """Retrieve the protein group in the case where the FASTA has decoys.

    Parameters
    ----------
    peptides : pandas.DataFrame
        The peptide dataframe.
    proteins : FastaProteins object

    Returns
    -------
    pandas.Series
        The protein group for each peptide.
    """
    return peptides["stripped sequence"].map(proteins.peptide_map.get)


def group_without_decoys(peptides, target_column, proteins):
    """Retrieve the protein group with a target-only FASTA.

    Build a dictionary mapping the decoy peptides to a plausible unique
    target peptide. Then proceed to map as with the targets.

    Parameters
    ----------
    peptides : pandas.DataFrame
        The peptide dataframe.
    target_column : str
        The column indicating if the peptide is a target.
    proteins : a FastaProteins object

    Returns
    -------
    pandas.Series
        The protein group for each peptide.
    """
    decoys = pd.Series(
        peptides.loc[~peptides[target_column], "stripped sequence"].unique()
    )

    # decoys is now a dict mapping decoy peptides to target peptides
    decoys = match_decoy(decoys, pd.Series(proteins.peptide_map.keys()))

    # Map decoys to target protein group:
    decoy_map = {}
    for decoy_peptide, target_peptide in decoys.items():
        protein_group = proteins.peptide_map[target_peptide].split(", ")
        protein_group = [proteins.decoy_prefix + p for p in protein_group]
        decoy_map[decoy_peptide] = ", ".join(protein_group)

    # First lookup targets:
    prots = peptides["stripped sequence"].map(proteins.peptide_map.get)
    prots[prots.isna()] = peptides[prots.isna()]["stripped sequence"].map(
        decoy_map.get
    )
    return prots
