"""
Implementation of the picked-protein approach for protein-level
confidence estimates.
"""

import logging

import numpy as np
import pandas as pd

from mokapot import utils
from mokapot.column_defs import STANDARD_COLUMN_NAME_MAP
from mokapot.peptides import match_decoy
from mokapot.proteins import Proteins

LOGGER = logging.getLogger(__name__)

PROT_GROUP_NAME = STANDARD_COLUMN_NAME_MAP["protein_group"]
STRIP_SEQUENCE_NAME = STANDARD_COLUMN_NAME_MAP["stripped_sequence"]


def picked_protein(
    peptides: pd.DataFrame,
    target_column: str,
    peptide_column: str,
    score_column: str,
    proteins: Proteins,
    rng: int | np.random.Generator,
) -> pd.DataFrame:
    """Perform the picked-protein approach

    Parameters
    ----------
    peptides : pandas.DataFrame
        A dataframe of the peptides.
    target_column : str
        The column in `peptides` indicating if the peptide is a target.
    peptide_column : str
        The column in `peptides` containing the peptide sequence.
    score_column : str
        The column in `peptides` containing the score.
    proteins : Proteins object
        A Proteins object.
    rng : int or numpy.random.Generator
        The random number generator.

    Returns
    -------
    pandas.DataFrame
        The aggregated proteins for confidence estimation.
    """
    keep = [target_column, peptide_column, score_column]

    # Trim the dataframe
    try:
        prots = peptides.loc[:, keep].rename(
            columns={peptide_column: "best_peptide"}
        )
    except KeyError as e:
        not_found_cols = [x for x in keep if x not in peptides.columns]
        raise ValueError(
            f"Columns not found in dataframe: {not_found_cols}"
            f" Found columns: {peptides.columns}"
        ) from e

    prots[STRIP_SEQUENCE_NAME] = strip_peptides(prots["best_peptide"])

    # There are two cases we need to deal with:
    # 1. The fasta contained both targets and decoys (ideal)
    # 2. The fasta contained only targets (less ideal)
    if proteins.has_decoys:
        prots[PROT_GROUP_NAME] = group_with_decoys(prots, proteins)

    else:
        LOGGER.info("Mapping decoy peptides to protein groups...")
        prots[PROT_GROUP_NAME] = group_without_decoys(
            prots, target_column, proteins
        )

    # Verify that unmatched peptides are shared:
    unmatched = pd.isna(prots[PROT_GROUP_NAME])
    if not proteins.has_decoys:
        unmatched[~prots[target_column]] = False

    unmatched_prots = prots.loc[unmatched, :]
    shared = unmatched_prots[STRIP_SEQUENCE_NAME].isin(
        proteins.shared_peptides.keys()
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
        LOGGER.warning(
            "%i out of %i peptides could not be mapped. "
            "Please check your digest settings.",
            shared_unmatched,
            len(prots),
        )

        if shared_unmatched / len(prots) > 0.10:
            raise ValueError(
                "Fewer than 90% of all peptides could be matched to proteins. "
                "Please verify that your digest settings are correct."
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
        prots[PROT_GROUP_NAME]
        .str.split(",", expand=True)[0]
        .map(lambda x: proteins.protein_map.get(x, x))
    )

    prot_idx = utils.groupby_max(prots, ["decoy"], score_column, rng)
    final_cols = [
        PROT_GROUP_NAME,
        "decoy",
        "best_peptide",
        STRIP_SEQUENCE_NAME,
        score_column,
        target_column,
    ]
    return prots.loc[prot_idx, final_cols]


def strip_peptides(sequences: pd.Series) -> pd.Series:
    """Strip modifications and flanking AA's from peptide sequences.

    Parameters
    ----------
    sequences : pandas.Series
        The peptide sequences.

    Returns
    -------
    pandas.Series
        The stripped peptide sequences.

    Example
    -------
    >>> pep = pd.Series(["A.LES[+79.]LIEK.A"])
    >>> srip_peptides(pep)
    0    LESLIEK
    dtype: object
    """
    # Strip modifications and flanking AA's from peptide sequences.
    sequences = (
        sequences.str.replace(r"[\[\(].*?[\]\)]", "", regex=True)
        .str.replace(r"^.*?\.", "", regex=True)
        .str.replace(r"\..*?$", "", regex=True)
        .str.strip("-")
    )

    # Sometimes folks use lowercase letters for the termini or mods:
    if all(sequences.str.islower()):
        sequences = sequences.str.upper()
    else:
        sequences = sequences.str.replace(r"[a-z]", "", regex=True)

    return sequences


def group_with_decoys(peptides: pd.DataFrame, proteins: Proteins):
    """Retrieve the protein group in the case where the FASTA has decoys.

    Parameters
    ----------
    peptides : pandas.DataFrame
        The peptide dataframe.
    proteins : a Proteins object

    Returns
    -------
    pandas.Series
        The protein group for each peptide.
    """
    return peptides[STRIP_SEQUENCE_NAME].map(proteins.peptide_map.get)


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
    proteins : a Proteins object

    Returns
    -------
    pandas.Series
        The protein group for each peptide.
    """
    decoys = pd.Series(
        peptides.loc[~peptides[target_column], STRIP_SEQUENCE_NAME].unique()
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
    prots = peptides[STRIP_SEQUENCE_NAME].map(proteins.peptide_map.get)
    prots[prots.isna()] = peptides[prots.isna()][STRIP_SEQUENCE_NAME].map(
        decoy_map.get
    )
    return prots
