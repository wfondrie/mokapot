"""Handle proteins for the picked protein FDR."""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass

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
    peptides : dict
        A mapping of peptides to the proteins that may have
        generated them.
    rng : np.random.Generator
        The random number generator used to create decoy protein groups.
    """

    def __init__(
        self,
        peptides: dict[Peptide, list[str]],
        rng: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize the Proteins object."""
        self.proteins = "mokapot protein group"
        self.protein_counts = "# mokapot protein groups"
        self.seqs = "stripped sequence"

        LOGGER.info("Ignoring shared peptides...")
        self.peptides = {k: (";".join(v), len(v)) for k, v in peptides.items()}
        self.rng = rng

        n_shared = sum(v[1] > 1 for v in self.peptides.values())
        LOGGER.info(
            "  - Ignored %i shared peptides for protein inference.",
            n_shared,
        )

        LOGGER.info(
            "  - Retained %i peptides from %i protein groups.",
            len(self.peptides) - n_shared,
            len({v[0] for v in self.peptides.values() if v[1] == 1}),
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
                self._group(data, schema),
                on=[self.seqs, schema.target],
                how="left",
            )

            # Verify that unmatched peptides are shared.
            missing = (
                data.filter(
                    pl.col(schema.target) & pl.col(self.proteins).is_null()
                )
                .select(self.seqs)
                .unique()
                .collect(streaming=True)
                .to_series()
                .to_list()
            )

            n_total = (
                data.select(self.seqs)
                .unique()
                .select(pl.count())
                .collect(streaming=True)
                .item()
            )

        if len(missing):
            LOGGER.warning(
                "%i out of %i peptides could not be mapped. "
                "Please check your digest settings.",
                len(missing),
                n_total,
            )

            if len(missing) / n_total > 0.10:
                raise ValueError(
                    "Fewer than 90% of all peptides could be matched to "
                    "proteins. Please verify that your digest settings are "
                    "correct."
                )

        return data

    def _group(self, seq_df: pl.DataFrame, schema: PsmSchema) -> pl.LazyFrame:
        """Get the protein groups for peptide sequences.

        Parameters
        ----------
        seq_df : polars.DataFrame
            The DataFrame of peptide sequences.
        schema : PsmSchema
            The column meanings.

        Returns
        -------
        pl.LazyFrame
            A mapping of peptides to protein groups.
        """
        seq_df = (
            seq_df.select(self.seqs, schema.target)
            .unique()
            .collect(streaming=True)
            .group_by(schema.target)
        )

        groups = []
        for label, seqs in seq_df:
            seqs = seqs.select(self.seqs).to_series()
            if label:
                df = [(s, *self.peptides[s]) for s in seqs]
                df = pl.DataFrame(
                    df,
                    schema=[seqs.name, self.proteins, self.protein_counts],
                )
            else:
                df = self._match_decoys(seqs)

            groups.append(
                df.lazy().with_columns(
                    [
                        pl.lit(label).alias(schema.target),
                        pl.col(seqs.name).cast(pl.Categorical),
                        pl.col(self.proteins).cast(pl.Categorical),
                    ]
                )
            )

        return pl.concat(groups, how="vertical")

    def _match_decoys(
        self,
        decoys: pl.LazyFrame,
    ) -> pl.DataFrame:
        """Find a corresponding target for each decoy.

        Matches a decoy to a unique random target peptide that
        has the same amino acid composition.

        Parameters
        ----------
        decoys : polars.Series
            A collection of stripped decoy peptide sequences
        rng : np.random.Generator
            The random number generator to use.

        Returns
        -------
        polars.DataFrame
            The corresponding target peptide for each
            decoy peptide.
        """
        prev_rng_state = random.getstate()
        random.seed(int(self.rng.integers(1, 9999)))
        names = [decoys.name, self.proteins, self.protein_counts]
        decoys = decoys.shuffle(self.rng.integers(1, 9999))

        target_comps = defaultdict(list)
        for target, prot in self.peptides.items():
            target_comps["".join(sorted(str(target)))].append(prot)

        # Shuffle them:
        for val in target_comps.values():
            random.shuffle(val)

        # Find the first target that matches a decoy composition:
        decoy_df = []
        for decoy in decoys:
            try:
                decoy_df.append(
                    (
                        decoy,
                        *target_comps["".join(sorted(str(decoy)))].pop(),
                    )
                )
            except IndexError:
                continue

        random.setstate(prev_rng_state)
        return pl.DataFrame(decoy_df, schema=names)


@dataclass
class Peptide:
    """A peptide sequence.

    This class stores the peptide as a substring of the protein,
    in an effort to reduce memory pressure.

    Parameters
    ----------
    protein : str
        The sequence of the protein.
    substr : clice
        The subsequence defining the peptide.
    """

    protein: str
    substr: slice

    def __str__(self) -> str:
        """The peptide sequence."""
        return self.protein[self.substr]

    def __hash__(self) -> str:
        """Hash the peptide sequence."""
        return hash(self.protein[self.substr])

    def __len__(self) -> int:
        """Get the sequence length."""
        return self.substr.stop - self.substr.start
