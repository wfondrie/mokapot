"""
Implementation of the picked-protein approach for protein-level
confidence estimates.
"""
import logging
from collections import defaultdict

import numpy as np
import polars as pl

from .proteins import Proteins
from .schema import PsmSchema

LOGGER = logging.getLogger(__name__)


def picked_protein(
    data: pl.LazyFrame,
    schema: PsmSchema,
    proteins: Proteins,
    rng: np.random.Generator,
) -> pl.LazyFrame:
    """Perform the picked-protein approach.

    Parameters
    ----------
    data : polars.LazyFrame
        A collection of examples, where the rows are an example and the columns
        are features or metadata describing them. We expect this dataframe to
        be sorted already.
    schema : mokapot.PsmSchema
        The meaning of the columns in the data.
    proteins : mokapot.Proteins
        The proteins to use for protein-level confidence estimation. This
        may be created with :py:func:`mokapot.read_fasta()`.
    rng : np.random.Generator
        A seed or generator used for cross-validation split creation and to
        break ties, or :code:`None` to use the default random number
        generator state.

    Returns
    -------
    polars.LazyFrame
        The aggregated proteins for confidence estimation.
    """
    data = data.select([schema.target, schema.peptide]).with_columns(
        pl.col(schema.peptide)
        .str.replace(r"[\[\(].*?[\]\)]", "")
        .str.replace(r"^.*?\.", "")
        .str.replace(r"\..*?$", "")
        .str.replace(r"[a-z]", "")
        .alias("stripped sequence")
    )

    # There are two cases we need to deal with:
    # 1. The fasta contained both targets and decoys (ideal)
    # 2. The fasta contained only targets (less ideal)
    if proteins.has_decoys:
        data = data.with_columns(
            pl.col("stripped sequence")
            .map_dict(proteins.peptide_map)
            .alias("mokapot protein group")
        )
    else:
        LOGGER.info("Mapping decoy peptides to protein groups...")
        data = data.with_columns(
            pl.when(pl.col(schema.target))
            .then(pl.col("stripped sequence").map_dict(proteins.peptde_map))
            .otherwise(
                pl.col("stripped sequence").map_dict(
                    group_without_decoys(
                        data=data, schema=schema, proteins=proteins, rng=rng
                    )
                )
            )
            .alias("mokapo protein group")
        )

    # Verify that unmatched peptides are shared:
    unmatched = (
        data.filter(pl.col("mokapot protein group").is_null())
        .select(
            [
                pl.col(schema.target),
                pl.col("stripped sequence"),
            ]
        )
        .with_columns(
            pl.col("stripped sequence")
            .is_in(proteins.shared_peptides.keys())
            .alias("is_shared")
        )
    )

    # Counts of everything:
    n_total = data.select(pl.count()).collect(streaming=True).item()
    n_unmatched = unmatched.select(pl.count()).collect(streaming=True).item()
    n_targets = (
        data.select(pl.col(schema.target).sum()).collect(streaming=True).item()
    )

    n_shared = (
        unmatched.select(pl.col("is_shared") & pl.col(schema.target))
        .sum()
        .collect(streaming=True)
        .item()
    )

    LOGGER.debug(
        "%i out of %i target peptides were discarded as shared peptides.",
        n_shared,
        n_targets,
    )

    if n_unmatched - n_shared:
        LOGGER.warning(
            "%i out of %i peptides could not be mapped. "
            "Please check your digest settings.",
            n_unmatched - n_shared,
            n_total,
        )

        if (n_unmatched - n_shared) / n_total > 0.10:
            raise ValueError(
                "Fewer than 90% of all peptides could be matched to proteins. "
                "Please verify that your digest settings are correct."
            )

    data = data.filter(
        ~pl.col("mokapot protein gruop").is_null()
    ).with_columns(
        pl.col("mokapot protein group")
        .str.split(",")
        .list.first()
        .map_dict(proteins.protein_map, default=pl.first())
    )

    return data


def group_without_decoys(
    data: pl.LazyFrame,
    schema: PsmSchema,
    proteins: Proteins,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Retrieve the protein group with a target-only FASTA.

    Build a dictionary mapping the decoy peptides to a plausible unique
    target peptide. Then proceed to map as with the targets.

    Parameters
    ----------
    data : polars.LazyFrame
        A collection of examples, where the rows are an example and the columns
        are features or metadata describing them. We expect this dataframe to
        be sorted already.
    schema : mokapot.PsmSchema
        The meaning of the columns in the data.
    proteins : mokapot.Proteins
        The proteins to use for protein-level confidence estimation. This
        may be created with :py:func:`mokapot.read_fasta()`.
    rng : np.random.Generator
        A seed or generator used for cross-validation split creation and to
        break ties, or :code:`None` to use the default random number
        generator state.

    Returns
    -------
    dict str, str
        The protein group for each decoy peptide.
    """
    decoys = (
        data.filter(~pl.col(schema.target))
        .select("stripped sequence")
        .unique()
        .collect(streaming=True)
        .to_series()
    )

    # decoy is now a dict mapping decoy peptides to target peptides
    decoy_dict = match_decoys(
        decoys=decoys,
        targets=pl.Series("targets", proteins.peptide_map.keys()),
        rng=rng,
    )

    # Map decoys to target protein group:
    decoy_map = {}
    for decoy_peptide, target_peptide in decoy_dict.items():
        protein_group = proteins.peptide_map[target_peptide].split(", ")
        protein_group = [proteins.decoy_prefix + p for p in protein_group]
        decoy_map[decoy_peptide] = ", ".join(protein_group)

    return decoy_map


def match_decoys(
    decoys: pl.Series,
    targets: pl.Series,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Find a corresponding target for each decoy.

    Matches a decoy to a unique random target peptide that
    has the same amino acid composition, including modifications.
    If none can be found, an :code:`nan` is returned for that
    peptide.

    Parameters
    ----------
    decoys : polars.Series
        A collection of stripped decoy peptide sequences
    targets : polars.Series
        A collection of stripped target peptide sequences
    rng : np.random.Generator
        The random number generator to use.

    Returns
    -------
    dict of str, str
        The corresponding target peptide for each
        decoy peptide.
    """
    targets = targets.shuffle(rng.integers(1, 9999))
    decoys = decoys.shuffle(rng.integers(1, 9999))

    # Get the compositions of each:
    targets = residue_sort(targets)
    decoys = residue_sort(decoys)

    # Build a map of composition to lists of peptides:
    targ_map = defaultdict(list)
    for peptide, comp in targets.iter_rows():
        targ_map[comp].append(peptide)

    # Find the first target peptide that matches the decoy composition
    decoy_map = {}
    for peptide, comp in decoys.iter_rows():
        try:
            decoy_map[peptide] = targ_map[comp].pop()
        except IndexError:
            continue

    return decoy_map


def residue_sort(peptides: pl.Series) -> pl.DataFrame:
    """Sort peptide sequences by amino acid.

    This function also considers potential modifications

    Parameters
    ----------
    peptides : polars.Series
        A collection of peptides

    Returns
    -------
    polars.DataFrame
        A lexographically sorted sequence that respects
        modifications.
    """
    peptides.name = "peptide"
    df = (
        peptides.to_frame()
        .lazy()
        .groupby("peptide")
        .agg(pl.col("targets").str.explode().sort().alias("sorted"))
        .with_columns(pl.col("sorted").list.join(""))
        .collect()
    )
    return df
