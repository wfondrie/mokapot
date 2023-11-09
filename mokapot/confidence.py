"""Confidence estimation in mokapot.

One of the primary purposes of mokapot is to assign confidence estimates to
PSMs. This task is accomplished by ranking PSMs according to a score and using
an appropriate confidence estimation procedure for the type of data. Mokapot
can provide confidence estimates based any score, regardless of whether it was
the result of a learned :py:func:`~mokapot.model.Model` object or provided
independently.

The following classes store the confidence estimates for a dataset based on the
provided score. They provide utilities to access, save, and plot these
estimates for the various relevant levels (i.e. PSMs, peptides, and proteins).
The :py:class:`~mokapot.confidence.PsmConfidence` class is appropriate for most
data-dependent acquisition proteomics datasets.

We recommend using the :py:func:`~mokapot.brew()` function or the
:py:meth:`~mokapot.PsmDataset.assign_confidence()` method to obtain these
confidence estimates, rather than initializing the classes below directly.
"""
from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import polars as pl
from triqler import qvality

from . import qvalues, utils, writers
from .base import BaseData
from .proteins import Proteins
from .schema import PsmSchema

LOGGER = logging.getLogger(__name__)


class Confidence(BaseData):
    """Estimate and store the statistical confidence for a collection of PSMs.

    Parameters
    ----------
    data : polars.DataFrame, polars.LazyFrame, or pandas.DataFrame
        A collection of examples, where the rows are an example and the columns
        are features or metadata describing them.
    schema : mokapot.PsmSchema
        The meaning of the columns in the data.
    scores : np.ndarray
        A vector containing the score of each PSM.
    desc : bool
        Are higher scores better?
    proteins : mokapot.Proteins
        The proteins to use for protein-level confidence estimation. This
        may be created with :py:func:`mokapot.read_fasta()`.
    eval_fdr : float
        The false discovery rate threshold for choosing the best feature and
        creating positive labels during the trainging procedure.
    rng : int or np.random.Generator
        A seed or generator used for cross-validation split creation and to
        break ties, or :code:`None` to use the default random number
        generator state.
    unit : str
        The unit to use in logging messages.
    levels : list of TdcLevel
        The levels on which to compute confidence estimates.

    Attributes
    ----------
    data : polars.LazyFrame
    columns : list of str
    targets : numpy.ndarray
    proteins : mokapot.Proteins
    rng : numpy.random.Generator
    results : ConfidenceEstimates
    decoy_results : ConfidenceEstimates

    :meta private:
    """

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame | dict,
        schema: PsmSchema,
        scores: np.ndarray | pl.Series,
        desc: bool,
        proteins: Proteins | None,
        eval_fdr: float,
        rng: float | None,
        unit: str,
        levels: list[TdcLevel],
    ) -> None:
        """Initialize a Confidence object."""
        super().__init__(
            data=data,
            schema=schema,
            proteins=proteins,
            eval_fdr=eval_fdr,
            unit=unit,
            rng=rng,
        )

        keep_cols = [
            c
            for c in self.data.columns
            if c not in schema.features or c in schema.metadata
        ]

        # Create the scores Series.
        try:
            self.schema.score = scores.name
        except AttributeError:
            self.schema.score = "provided score"

        self.schema.desc = bool(desc)
        self._levels = levels
        self._data = self.data.select(keep_cols).with_columns(
            pl.Series(scores).alias(self.schema.score)
        )

        # Add proteins if necessary
        if self.proteins is not None:
            protein_groups = self.proteins.pick(self.data, self.schema)
            self._data = self.data.join(
                protein_groups,
                on=[self.schema.target, self.schema.peptide],
                how="left",
            )

            self._levels.append(
                TdcLevel(
                    name="proteins",
                    columns="mokapot protein group",
                    unit="protein groups",
                    schema=self.schema,
                    rng=self.rng,
                )
            )

        # Transform the labels into the correct format,
        # replacing the original coluimn in the dataframe.
        # Also sort for TDC.
        # The '__rand__' column is used to ensure ties are broken
        # randomly.
        with pl.StringCache():
            n_rows = (
                self.data.select(pl.count()).collect(streaming=True).item()
            )

            self._data = (
                self.data.with_columns(
                    pl.lit(self.targets).alias(schema.target),
                    pl.lit(self.rng.random(n_rows, dtype=np.float32)).alias(
                        "__rand__"
                    ),
                )
                .sort(by=[schema.score, "__rand__"], descending=schema.desc)
                .drop("__rand__")
            )

            # These attribute holds the results as DataFrames:
            self._results, self._decoy_results = self._assign_confidence()

    def __repr__(self) -> str:
        """The string representation for these objects."""
        return str(self.results)

    @property
    def results(self) -> ConfidenceEstimates:
        """The confidence estimates for target examples."""
        return self._results

    @property
    def decoy_results(self) -> ConfidenceEstimates:
        """The confidence estimates for decoy examples."""
        return self._decoy_results

    @property
    def desc(self) -> bool:
        """Are higher scores better?."""
        return self.schema.desc

    @property
    def levels(self) -> list[str]:
        """The available levels for confidence estimates."""
        return [x.name for x in self._levels]

    def to_parquet(
        self,
        *,
        dest_dir: PathLike | None = None,
        stem: str | None = None,
        decoys: bool = False,
        ext: str = "parquet",
        **kwargs: dict,
    ) -> list[Path, ...]:
        """Save confidence estimates to Apache Parquet files.

        Write the confidence estimates for each of the available levels
        (i.e. PSMs, peptides, proteins) to a Parquet file. Apache Parquet
        is a popular and effecient columnar data format and core part of
        modern data infrastructure.

        If deciding between Parquet or a text format, we recommend Parquet.

        Parameters
        ----------
        dest_dir : PathLike or None, optional
            The directory in which to save the files. :code:`None` will use the
            current working directory.
        stem : str or None, optional
            An optional prefix for the confidence estimate files. The suffix
            will always be "mokapot.{level}.{ext}" where "{level}" indicates
            the level at which confidence estimation was performed (i.e. PSMs,
            peptides, proteins).
        decoys : bool, optional
            Save decoys confidence estimates as well?
        ext : str, optional
            The extention to use when saving the files.
        **kwargs : dict
            Keyword arguments passed to
            :py:method:`polars.LazyFrame.sink_parquet()`.

        Returns
        -------
        list of Path
            The paths to the saved files.

        """
        return writers.to_parquet(
            conf=self,
            dest_dir=dest_dir,
            stem=stem,
            decoys=decoys,
            ext=ext,
            **kwargs,
        )

    def to_txt(
        self,
        *,
        dest_dir: PathLike | None = None,
        stem: str | None = None,
        separator: str = "\t",
        decoys: bool = False,
        ext: str = "txt",
    ) -> list[Path, ...]:
        """Save confidence estimates to delimited text files.

        Write the confidence estimates for each of the available levels
        (i.e. PSMs, peptides, proteins) to separate flat text files using the
        specified delimiter.

        Parameters
        ----------
        dest_dir : PathLike or None, optional
            The directory in which to save the files. :code:`None` will use the
            current working directory.
        stem : str or None, optional
            An optional prefix for the confidence estimate files. The suffix
            will always be "mokapot.{level}.{ext}" where "{level}" indicates
            the level at which confidence estimation was performed (i.e. PSMs,
            peptides, proteins).
        separator : str, optional
            The delimiter to use.
        decoys : bool, optional
            Save decoys confidence estimates as well?
        ext : str, optional
            The extention to use when saving the files.

        Returns
        -------
        list of Path
            The paths to the saved files.

        """
        return writers.to_txt(
            conf=self,
            dest_dir=dest_dir,
            stem=stem,
            separator=separator,
            decoys=decoys,
            ext=ext,
        )

    def to_csv(
        self,
        *,
        dest_dir: PathLike | None = None,
        stem: str | None = None,
        decoys: bool = False,
        ext: str = "csv",
        **kwargs: dict,
    ) -> list[Path, ...]:
        """Save confidence estimates to comma-separated value files.

        Write the confidence estimates for each of the available levels
        (i.e. PSMs, peptides, proteins) to CSV files using the
        specified delimiter.

        Parameters
        ----------
        conf : mokapot.Confidence
            The mokapot confidence estimates.
        dest_dir : PathLike or None, optional
            The directory in which to save the files. :code:`None` will use the
            current working directory.
        stem : str or None, optional
            An optional prefix for the confidence estimate files. The suffix
            will always be "mokapot.{level}.csv" where "{level}" indicates the
            level at which confidence estimation was performed (i.e. PSMs,
            peptides, proteins).
        decoys : bool, optional
            Save decoys confidence estimates as well?
        ext : str, optional
            The extention to use when saving the files.
        **kwargs : dict
            Keyword arguments passed to
            :py:method:`polars.LazyFrame.sink_csv()`.

        Returns
        -------
        list of Path
            The paths to the saved files.

        """
        return writers.to_csv(
            conf=self,
            dest_dir=dest_dir,
            stem=stem,
            decoys=decoys,
            ext=ext,
            **kwargs,
        )

    def _assign_confidence(self) -> None:
        """Assign confidence estimates for the various levels."""
        data = self._data
        targets = {}
        decoys = {}
        tgt_expr = pl.col(self.schema.target)  # Bool, True if target
        filter_expr = pl.col("mokapot q-value") <= self.eval_fdr
        for level in self._levels:
            # This is kind of hacky, but I haven't figured out a better way to
            # do this yet.
            if level.name == "proteins":
                data = data.filter(
                    pl.col("# mokapot protein groups") == 1
                ).drop("# mokapot protein groups")

            data = level.assign_confidence(data)
            targets[level] = data.filter(tgt_expr).drop([self.schema.target])
            decoys[level] = data.filter(~tgt_expr).drop([self.schema.target])

            n_accepted = (
                data.filter(tgt_expr & filter_expr)
                .select(pl.count())
                .collect(streaming=True)
                .item()
            )

            LOGGER.info(
                "  - Found %i %s with q<=%g",
                n_accepted,
                level.unit,
                self.eval_fdr,
            )
            data = data.clone().drop(["mokapot q-value", "mokapot PEP"])

        return (
            ConfidenceEstimates(self.eval_fdr, targets),
            ConfidenceEstimates(self.eval_fdr, decoys),
        )


class PsmConfidence(Confidence):
    """Assign confidence estimates to a set of PSMs.

    Estimate q-values and posterior error probabilities (PEPs) for PSMs,
    peptides, and optionally proteins when ranked by the provided scores.

    Parameters
    ----------
    data : polars.DataFrame, polars.LazyFrame, or pandas.DataFrame
        A collection of examples, where the rows are an example and the columns
        are features or metadata describing them.
    schema : mokapot.PsmSchema
        The meaning of the columns in the data.
    scores : np.ndarray
        A vector containing the score of each PSM.
    desc : bool
        Are higher scores better?
    proteins : mokapot.Proteins
        The proteins to use for protein-level confidence estimation. This
        may be created with :py:func:`mokapot.read_fasta()`.
    eval_fdr : float
        The false discovery rate at which to summarize results.
    rng : int or np.random.Generator
        A seed or generator used for cross-validation split creation and to
        break ties, or :code:`None` to use the default random number
        generator state.

    Attributes
    ----------
    data : polars.LazyFrame
    columns : list of str
    targets : numpy.ndarray
    proteins : mokapot.Proteins
    rng : numpy.random.Generator
    results : ConfidenceEstimates
    decoy_results : ConfidenceEstimates
    """

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame | dict,
        schema: PsmSchema,
        scores: np.ndarray | pl.Series,
        desc: bool = True,
        proteins: Proteins | None = None,
        eval_fdr: float = 0.01,
        rng: float | None = None,
    ) -> None:
        """Initialize a a LinearPsmConfidence object."""
        LOGGER.info("Performing target-decoy competition...")
        super().__init__(
            data=data,
            schema=schema,
            scores=scores,
            desc=desc,
            proteins=proteins,
            eval_fdr=eval_fdr,
            rng=rng,
            unit="PSMs",
            levels=[
                TdcLevel(
                    name="psms",
                    columns=schema.spectrum,
                    unit="PSMs",
                    schema=schema,
                    rng=rng,
                ),
                TdcLevel(
                    name="peptides",
                    columns=schema.peptide,
                    unit="peptides",
                    schema=schema,
                    rng=rng,
                ),
            ],
        )

    def to_flashlfq(self, out_file: str = "mokapot.flashlfq.txt") -> str:
        """Save confidenct peptides for quantification with FlashLFQ.

        `FlashLFQ <https://github.com/smith-chem-wisc/FlashLFQ>`_ is an
        open-source tool for label-free quantification. For mokapot to save
        results in a compatible format, a few extra columns are required to
        be present, which specify the MS data file name, the theoretical
        peptide monoisotopic mass, the retention time, and the charge for each
        PSM. If these are not present, saving to the FlashLFQ format is
        disabled.

        Note that protein grouping in the FlashLFQ results will be more
        accurate if proteins were added for analysis with mokapot.

        Parameters
        ----------
        out_file : str, optional
            The output file to write.

        Returns
        -------
        str
            The path to the saved file.

        """
        return writers.to_flashlfq(self, out_file)


@dataclass
class TdcLevel:
    """A level for confidence estimation.

    Parameters
    ----------
    name : str
        The name to use in reference attributes and such.
    columns : str or list of str
        The columns to perform competition on.
    unit : str
        The unit to use in logging messages.
    schema : PsmSchema
        The column schema to reference.
    rng : numpy.random.Generator
        The random number generator
    """

    name: str
    columns: str | list[str]
    unit: str
    schema: PsmSchema
    rng: np.random.Generator

    def __post_init__(self) -> None:
        """Make sure columns are a list."""
        self.columns = utils.listify(self.columns)

    def __hash__(self) -> int:
        """Make TdcLevels hashable."""
        return hash(self.name)

    def assign_confidence(  # noqa: C901
        self,
        data: pl.LazyFrame,
    ) -> pl.DataFrame:
        """Compute confidence estimates at the specified level.

        Parameters
        ----------
        data : polars.LazyFrame
            The data to compute estimates from. We expect
            the data to be sorted!
        desc : bool
            Are higher score better?

        Returns
        -------
        polars.LazyFrame
            The data with q-values.

        :meta private:
        """
        # Define some shorhand:
        score = self.schema.score
        target = self.schema.target
        is_proteins = "mokapot protein group" in self.columns

        keep = [*self.columns, target, score]
        if self.schema.group is not None:
            keep += self.schema.group

        tdc_df = (
            data.group_by(self.columns)
            .first()
            .sort(by=self.schema.score, descending=self.schema.desc)
            .select(keep)
            .collect(streaming=True)
        )

        if is_proteins:
            # Due to string caching headaches...
            tdc_df = tdc_df.with_columns(
                pl.col("mokapot protein group").cast(pl.Utf8)
            )

        # Compute q-values
        conf_out = []
        if self.schema.group is not None:
            group_iter = tdc_df.group_by(self.schema.group)
        else:
            group_iter = [(None, tdc_df)]

        for grp_name, grp_df in group_iter:
            if grp_name is not None:
                LOGGER.info("=== Group %s ===", grp_name)

            # Collect the data needed to compute q-values and PEPs:
            score = self.schema.score
            target = self.schema.target

            # Warn if no decoys are available.
            if grp_df[target].all():
                LOGGER.warning(
                    "No decoy %s remain for confidence estimation. "
                    "Confidence estimates may be unreliable.",
                    self.unit,
                )

            LOGGER.info("Assiging q-values to %s...", self.unit)
            qvals = qvalues.tdc(
                scores=grp_df[score],
                target=grp_df[target],
            )

            # Calculate PEPs
            LOGGER.info("Assiging PEPs to %s...", self.unit)
            try:
                # Handle the random state:
                prev_state = np.random.get_state()
                np.random.seed(self.rng.integers(1, 9999))

                # We need to capture stdout here:
                msg = io.StringIO()
                with contextlib.redirect_stdout(msg):
                    _, peps = qvality.getQvaluesFromScores(
                        grp_df.filter(pl.col(target))[score],
                        grp_df.filter(~pl.col(target))[score],
                        includeDecoys=True,
                    )

                if msg.getvalue().startswith("Warning: IRLS did not converge"):
                    LOGGER.warning("PEP calculation did not converge.")

                # Restore the random state:
                np.random.set_state(prev_state)
            except SystemExit as msg:
                if "no decoy hits available for PEP calculation" in str(msg):
                    peps = [0] * len(grp_df)
                else:
                    raise

            # Add the columns
            conf_out.append(
                grp_df.with_columns(
                    [
                        pl.Series(qvals).alias("mokapot q-value"),
                        pl.Series(peps).alias("mokapot PEP"),
                    ]
                ).lazy()
            )

            # Cast back to categorical:
            if is_proteins:
                conf_out[-1] = conf_out[-1].with_columns(
                    pl.col("mokapot protein group").cast(pl.Categorical)
                )

        return data.join(
            pl.concat(conf_out, how="vertical"),
            on=[*self.columns, target, score],
            how="inner",
        )


class ConfidenceEstimates:
    """Store the confidence estimate result tables.

    Parameters
    ----------
    eval_fdr: float
        The FDR threshold to use for printing.
    levels: dict of TdcLevel, pl.LazyFrame
        The level and result table for each level.

    Attributes
    ----------
    eval_fdr : float
       The FDR threshold to use for printing.
    """

    def __init__(
        self,
        eval_fdr: float,
        levels: dict[TdcLevel, pl.LazyFrame],
    ) -> None:
        """Initialize the object."""
        self.eval_fdr = eval_fdr
        for level, table in levels.items():
            setattr(self, level.name, table)

        self._levels = tuple(levels.keys())

    def __getitem__(self, level: str) -> pl.LazyFrame:
        """Get the confidence estimates."""
        if level in self._levels:
            return getattr(self, level)

        raise KeyError(f"{level} is not a valid level.")

    def __iter__(self) -> tuple[TdcLevel, pl.LazyFrame]:
        """Iterate over the result tables."""
        for level in self._levels:
            yield (level, getattr(self, level.name))

    def __repr__(self) -> str:
        """How to print the class."""
        lines = ["Confidence estimates for:"]
        with pl.StringCache():
            for level, table in self:
                num = (
                    table.select(
                        (pl.col("mokapot q-value") <= self.eval_fdr).sum()
                    )
                    .collect(streaming=True)
                    .item()
                )

                summary = f"({num} {level.unit} at q<={self.eval_fdr})"
                lines.append(f"  - {level.name} {summary}")

        return "\n".join(lines)
