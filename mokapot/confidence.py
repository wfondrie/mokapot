"""One of the primary purposes of mokapot is to assign confidence estimates to
PSMs. This task is accomplished by ranking PSMs according to a score and using
an appropriate confidence estimation procedure for the type of data. Mokapot
can provide confidence estimates based any score, regardless of whether it was
the result of a learned :py:func:`~mokapot.model.Model` instance or provided
independently.

The following classes store the confidence estimates for a dataset based on the
provided score. They provide utilities to access, save, and plot these
estimates for the various relevant levels (i.e. PSMs, peptides, and proteins).
The :py:class:`~mokapot.confidence.PsmConfidence` class is appropriate for most
data-dependent acquisition proteomics datasets. In contrast, the
:py:class:`~mokapot.confidence.PeptideConfidence` class uses peptide-level
competition to assign confidence to peptides, which is particularly relevant
for data-independent acquisition proteomics datasets.

We recommend using the :py:func:`~mokapot.brew()` function or the
:py:meth:`~mokapot.PsmDataset.assign_confidence()` method to obtain these
confidence estimates, rather than initializing the classes below directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from triqler import qvality

from . import qvalues, utils
from .base import BaseData
from .picked_protein import picked_protein
from .proteins import Proteins
from .schema import PsmSchema
from .writers import to_flashlfq, to_txt

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
            pl.lit(scores).alias(self.schema.score)
        )

        # Add proteins if necessary
        if self.proteins is not None:
            protein_groups = picked_protein(
                data=self.data,
                schema=self.schema,
                proteins=self.proteins,
                rng=self.rng,
            )
            self._data = self.data.with_columns(
                pl.lit(protein_groups).alias("mokapot protein group")
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
        n_rows = self.data.select(pl.count()).collect(streaming=True).item()
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
        self.confidence_estimates = {}
        self.decoy_confidence_estimates = {}
        self.num_accepted = {}
        self._assign_confidence()

    @property
    def desc(self) -> bool:
        """Are higher scores better?"""
        return self.schema.desc

    @property
    def levels(self) -> list[str]:
        """
        The available levels for confidence estimates.
        """
        return [x.name for x in self._levels]

    def __getattr__(self, attr):
        """Add confidence estimates as attributes."""
        if "confidence_estimates" not in vars(self):
            raise AttributeError

        try:
            return self.confidence_estimates[attr].collect(streaming=True)
        except KeyError as exc:
            raise AttributeError from exc

    def to_txt(self, dest_dir=None, file_root=None, sep="\t", decoys=False):
        """Save confidence estimates to delimited text files.

        Parameters
        ----------
        dest_dir : str or None, optional
            The directory in which to save the files. `None` will use the
            current working directory.
        file_root : str or None, optional
            An optional prefix for the confidence estimate files. The suffix
            will always be "mokapot.{level}.txt", where "{level}" indicates the
            level at which confidence estimation was performed (i.e. PSMs,
            peptides, proteins).
        sep : str, optional
            The delimiter to use.
        decoys : bool, optional
            Save decoys confidence estimates as well?

        Returns
        -------
        list of str
            The paths to the saved files.

        """
        return to_txt(
            self,
            dest_dir=dest_dir,
            file_root=file_root,
            sep=sep,
            decoys=decoys,
        )

    def _assign_confidence(self) -> None:
        """Assign confidence estimates for the various levels."""
        data = self._data
        filter_expr = pl.col("mokapot q-value") <= self.eval_fdr
        for level in self._levels:
            data = level.assign_confidence(data)
            self.confidence_estimates[level.name] = data.filter(
                pl.col(self.schema.target)
            ).drop([self.schema.target])

            self.decoy_confidence_estimates[level.name] = data.filter(
                ~pl.col(self.schema.target)
            ).drop([self.schema.target])

            n_accepted = (
                data.filter(pl.col(self.schema.target) & filter_expr)
                .select(pl.count())
                .collect(streaming=True)
                .item()
            )

            self.num_accepted[level.name] = n_accepted
            LOGGER.info(
                "  - Found %i %s with q<=%g",
                n_accepted,
                level.unit,
                self.eval_fdr,
            )
            data = data.clone().drop(["mokapot q-value", "mokapot PEP"])


class PsmConfidence(Confidence):
    """Assign confidence estimates to a set of PSMs

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
        The false discovery rate threshold for choosing the best feature and
        creating positive labels during the trainging procedure.
    rng : int or np.random.Generator
        A seed or generator used for cross-validation split creation and to
        break ties, or :code:`None` to use the default random number
        generator state.

    Attributes
    ----------
    levels : list of str
    psms : pandas.DataFrame
        Confidence estimates for PSMs in the dataset.
    peptides : pandas.DataFrame
        Confidence estimates for peptides in the dataset.
    proteins : pandas.DataFrame or None
        Confidence estimates for proteins in the dataset.
    confidence_estimates : Dict[str, pandas.DataFrame]
        A dictionary of confidence estimates at each level.
    decoy_confidence_estimates : Dict[str, pandas.DataFrame]
        A dictionary of confidence estimates for the decoys at each level.
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
        """Initialize a a LinearPsmConfidence object"""
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

    def __repr__(self):
        """How to print the class"""
        base = (
            "A mokapot.confidence.LinearConfidence object:\n"
            f"\t- PSMs at q<={self.eval_fdr:g}: {self.num_accepted['psms']}\n"
            f"\t- Peptides at q<={self.eval_fdr:g}: "
            f"{self.num_accepted['peptides']}\n"
        )

        if self.proteins is not None:
            base += (
                f"\t- Protein groups at q<={self.eval_fdr:g}: "
                f"{self.num_accepted['proteins']}\n"
            )

        return base

    def to_flashlfq(self, out_file="mokapot.flashlfq.txt"):
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
        return to_flashlfq(self, out_file)


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

    def __post_init__(self):
        """Make sure columns are a list."""
        self.columns = utils.listify(self.columns)

    def assign_confidence(
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
        keep = [*self.columns, target, score]
        if self.schema.group is not None:
            keep += self.schema.group

        tdc_df = (
            data.clone()
            .groupby(self.columns)
            .first()
            .sort(by=self.schema.score, descending=self.schema.desc)
            .select(keep)
            .collect(streaming=True)
        )

        # Compute q-values
        conf_out = []
        if self.schema.group is not None:
            group_iter = tdc_df.groupby(self.schema.group)
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
                _, peps = qvality.getQvaluesFromScores(
                    grp_df.filter(pl.col(target))[score],
                    grp_df.filter(~pl.col(target))[score],
                    includeDecoys=True,
                )
            except SystemExit as msg:
                if "no decoy hits available for PEP calculation" in str(msg):
                    peps = 0
                else:
                    raise

            # Add the columns
            conf_out.append(
                grp_df.with_columns(
                    [
                        pl.lit(qvals).alias("mokapot q-value"),
                        pl.lit(peps).alias("mokapot PEP"),
                    ]
                )
            )

        data = data.join(
            pl.concat(conf_out, how="vertical").lazy(),
            on=[*self.columns, target, score],
            how="left",
        )

        return data
