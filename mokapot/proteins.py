"""Handle proteins for the picked protein FDR."""
import logging
from collections import defaultdict

import numpy as np
import polars as pl

from . import utils
from .mixins import RngMixin
from .schema import PsmSchema

LOGGER = logging.getLogger(__name__)


class Proteins(RngMixin):
    """Map peptides to protein groups.

    This class stores the mapping of peptides to proteins and the mapping of
    target proteins to their corresponding decoy proteins. It is required for
    protein-level confidence estimation using the picked-protein approach.

    We recommend creating Proteins objects using the
    :py:func:`mokapot.read_fasta()` function.

    Parameters
    ----------
    peptides : dict
        A mapping of peptides to the proteins that may have
        generated them.
    rng : int or np.random.Generator, optional
        The random number generator used to create decoy protein groups.

    Attributes
    ----------
    peptides : polars.LazyFrame
        A mapping of peptides to the proteins that may have
        generated them.
    rng : np.random.Generator
        The random number generator used to create decoy protein groups.
    """

    def __init__(
        self,
        peptides: dict[str, list[str]],
        rng: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize the Proteins object."""
        self.proteins = "mokapot protein group"
        self.protein_counts = "# mokapot protein groups"
        self.seqs = "stripped sequence"
        self.target_seqs = "target_seq"

        self.data = pl.DataFrame(
            (
                (pep, ";".join(prot), len(prot))
                for pep, prot in peptides.items()
            ),
            schema={
                self.target_seqs: pl.Utf8,
                self.proteins: pl.Categorical,
                self.protein_counts: pl.UInt16,
            },
        ).lazy()

        self.rng = rng

        # Unique peptides:
        LOGGER.info("Ignoring shared peptides...")
        counts = self.data.select(
            [
                pl.count().alias("total"),
                (pl.col(self.protein_counts) > 1).sum().alias("is_shared"),
                pl.col(self.proteins)
                .filter(pl.col(self.protein_counts) == 1)
                .unique()
                .count()
                .alias("proteins"),
            ]
        ).collect(streaming=True)

        LOGGER.info(
            "  - Ignored %i shared peptides for protein inference.",
            counts["is_shared"].item(),
        )

        LOGGER.info(
            "  - Retained %i peptides from %i protein groups.",
            counts["total"].item() - counts["is_shared"].item(),
            counts["proteins"].item(),
        )

    def pick(
        self,
        data: pl.LazyFrame,
        schema: PsmSchema,
    ) -> pl.LazyFrame:
        """Perform the picked-protein approach.

        Parameters
        ----------
        data : polars.LazyFrame
            A collection of examples, where the rows are an example and the
            columns are features or metadata describing them. We expect this
            dataframe to be sorted already.
        schema : mokapot.PsmSchema
            The meaning of the columns in the data.

        Returns
        -------
        polars.LazyFrame
            The aggregated proteins for confidence estimation.
        """
        LOGGER.info("Mapping decoy peptides to protein groups...")

        data = (
            utils.make_lazy(data)
            .select([schema.target, schema.peptide])
            .with_columns(
                pl.col(schema.peptide)
                .str.replace(r"[\[\(].*?[\]\)]", "")
                .str.replace(r"^.*?\.", "")
                .str.replace(r"\..*?$", "")
                .str.replace(r"[a-z]", "")
                .cast(pl.Categorical)
                .alias(self.seqs)
            )
        )

        # Add protein groups:
        with pl.StringCache():
            data = data.join(
                self._group(data=data, schema=schema),
                on=[self.seqs, schema.target],
                how="left",
            )

            # Verify that unmatched peptides are shared.
            n_total, n_missing = (
                data.select(
                    [
                        pl.count().alias("total"),
                        pl.col(self.proteins).is_null().sum().alias("missing"),
                    ]
                )
                .collect(streaming=True)
                .rows()[0]
            )

        if n_missing:
            LOGGER.warning(
                "%i out of %i peptides could not be mapped. "
                "Please check your digest settings.",
                n_missing,
                n_total,
            )

            if n_missing / n_total > 0.10:
                raise ValueError(
                    "Fewer than 90% of all peptides could be matched to "
                    "proteins. Please verify that your digest settings are "
                    "correct."
                )

        return data

    def _group(
        self,
        data: pl.LazyFrame,
        schema: PsmSchema,
    ) -> dict[str, str]:
        """Retrieve the protein groups.

        Build a mapping of the decoy peptides to a plausible unique
        target peptide. Then proceed to map to proteins as with the targets.

        Parameters
        ----------
        data : polars.LazyFrame
            A collection of examples, where the rows are an example and the
            columns are features or metadata describing them. We expect this
            dataframe to be sorted already.
        schema : mokapot.PsmSchema
            The meaning of the columns in the data.
        rng : np.random.Generator
            A seed or generator used for cross-validation split creation and to
            break ties, or :code:`None` to use the default random number
            generator state.

        Returns
        -------
        polars.LazyFrame
            The protein group for each peptide.
        """
        decoys = (
            data.filter(~pl.col(schema.target))
            .select(self.seqs)
            .unique()
            .cast(pl.Utf8)
            .collect(streaming=True)
            .to_series()
        )

        # All of the theoretical targets:
        targets = (
            self.data.select(self.target_seqs)
            .collect(streaming=True)
            .to_series()
        )

        group_df = (
            match_decoys(
                decoys=decoys,
                targets=targets,
                rng=self.rng,
            )
            .lazy()
            .join(self.data, how="left", on=self.target_seqs)
            .drop(self.target_seqs)
            .with_columns(pl.lit(False).alias(schema.target))
            .join(
                self.data.rename({self.target_seqs: self.seqs}).with_columns(
                    pl.lit(True).alias(schema.target)
                ),
                how="outer",
                on=[
                    self.seqs,
                    schema.target,
                    self.proteins,
                    self.protein_counts,
                ],
            )
            .with_columns(pl.col(self.seqs).cast(pl.Categorical))
        )
        return group_df


def match_decoys(
    decoys: pl.Series,
    targets: pl.Series,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Find a corresponding target for each decoy.

    Matches a decoy to a unique random target peptide that
    has the same amino acid composition.

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
    polars.DataFrame
        The corresponding target peptide for each
        decoy peptide.
    """
    names = [decoys.name, targets.name]
    targets = residue_sort(targets.shuffle(rng.integers(1, 9999)))
    decoys = residue_sort(decoys.shuffle(rng.integers(1, 9999)))

    # Check for overlaps with targets and decoys:
    overlap = (
        targets.join(decoys, on="peptide", how="inner")
        .select("peptide")
        .rename({"peptide": names[0]})
    )

    if n_overlap := overlap.select(pl.count()).collect().item():
        LOGGER.warning(
            "%i decoy peptides are identical to target sequences.",
            n_overlap,
        )

    # Build a map of composition to lists of peptides:
    targ_map = defaultdict(list)
    for peptide, comp in targets.collect(streaming=True).iter_rows():
        targ_map[comp].append(peptide)

    # Find the first target peptide that matches the decoy composition
    decoy_df = []
    for peptide, comp in decoys.collect(streaming=True).iter_rows():
        try:
            decoy_df.append((peptide, targ_map[comp].pop()))
        except IndexError:
            continue

    return pl.DataFrame(decoy_df, schema=names)


def residue_sort(peptides: pl.Series) -> pl.DataFrame:
    """Sort peptide sequences by amino acid.

    This function also considers potential modifications

    Parameters
    ----------
    peptides : polars.Series
        A collection of peptides

    Returns
    -------
    polars.LazyFrame
        A lexographically sorted sequence that respects
        modifications.
    """
    peptides = peptides.rename("peptide")
    return (
        peptides.to_frame()
        .lazy()
        .group_by("peptide")
        .agg(
            pl.col("peptide")
            .cast(pl.Utf8)
            .str.explode()
            .sort()
            .alias("sorted")
        )
        .with_columns(pl.col("sorted").list.join("").cast(pl.Categorical))
    )
