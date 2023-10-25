"""Store and use peptide detections.

The :py:class:`PsmDataset` class is used to define a collection
peptide-spectrum matches. The :py:class:`PsmDataset` class is suitable for
most types of data-dependent acquisition proteomics experiments.
"""
from __future__ import annotations

import copy
import logging
import math
from functools import partial

import numpy as np
import polars as pl

from . import qvalues
from .base import BaseData
from .confidence import Confidence, PsmConfidence
from .proteins import Proteins
from .schema import PsmSchema

LOGGER = logging.getLogger(__name__)


class _BaseDataset(BaseData):
    """
    Store a dataset.

    Parameters
    ----------
    data : polars.DataFrame, polars.LazyFrame, or pandas.DataFrame
        A collection of PSMs, where the rows are PSMs and the columns are
        features or metadata describing them.
    schema : mokapot.PsmSchema
        The meaning of the columns in the data.
    proteins : mokapot.Proteins, optional
        The proteins to use for protein-level confidence estimation. This
        may be created with :py:func:`mokapot.read_fasta()`.
    eval_fdr : float, optional
        The false discovery rate threshold for choosing the best feature and
        creating positive labels during the trainging procedure.
    rng : int or np.random.Generator, optional
        A seed or generator used for cross-validation split creation and to
        break ties, or :code:`None` to use the default random number
        generator state.
    unit : str
        The unit to use in logging messages.
    straification : list[str]
        Columns in the data to use for grouping into cross-validation folds.
        examples in the same group are always assigned to the same
        cross-validation fold.
    subset: int or None
        The maximum number of examples to use for training.

    Attributes
    ----------
    data : polars.LazyFrame
    columns : list of str
    targets : numpy.ndarray
    features : numpy.ndarray
    proteins : mokapot.Proteins
    best_feature : tuple of (str, bool)
    rng : numpy.random.Generator
    subset : int or None
        The maximum number of examples to use for training.

    :meta private:
    """

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame | dict,
        schema: PsmSchema,
        proteins: Proteins | None,
        eval_fdr: float,
        rng: int | np.random.Generator | None,
        unit: str,
        stratification: list[str],
        subset: int | None,
    ) -> None:
        """Initialize an object."""
        super().__init__(
            data=data,
            schema=schema,
            proteins=proteins,
            eval_fdr=eval_fdr,
            rng=rng,
            unit=unit,
        )

        self.subset = subset
        self._stratification = stratification

        # Added later:
        self._folds = None

        # Provide some logging about the dataset:
        LOGGER.info("Using %i features:", len(self.schema.features))
        for i, feat in enumerate(self.schema.features):
            LOGGER.info("  (%i)\t%s", i + 1, feat)

        LOGGER.info("Found %i %s.", len(self), unit)

        targets = self.targets
        num_targets = targets.sum()
        num_decoys = (~targets).sum()
        LOGGER.info(
            "  - %i target %s and %i decoy %s detected.",
            num_targets,
            self._unit,
            num_decoys,
            self._unit,
        )

        # Validate the target column.
        if not num_targets:
            raise ValueError(f"No target {self._unit} were detected.")
        if not num_decoys:
            raise ValueError(f"No decoy {self._unit} were detected.")
        if not len(self):
            raise ValueError(f"No {self._unit} were detected.")

    @property
    def features(self) -> np.ndarray:
        """The feature matrix."""
        return (
            self.data.select(self.schema.features)
            .collect(streaming=True)
            .to_numpy()
        )

    @property
    def best_feature(self) -> tuple(str, bool):
        """The best feature for separating target and decoy examples."""
        if self.schema.score is None:
            self._find_best_feature()

        return self.schema.score, self.schema.desc

    def update_labels(
        self,
        scores: np.ndarray,
        desc: bool = True,
    ) -> np.ndarray:
        """Return the label for each example, given the scores.

        Used during model training to define positive examples, which are
        traditionally the target examples that fall within a specified FDR
        threshold.

        Parameters
        ----------
        scores : numpy.ndarray
            The score used to rank the examples.
        desc : bool, default
            Are higher scores better?

        Returns
        -------
        np.ndarray
            The label of each example, where 1 indicates a positive example, -1
            indicates a negative example, and 0 removes the example from
            training.

        """
        if len(scores.shape) > 1 and sum(np.array(scores.shape) > 0) == 1:
            scores = scores.flatten()

        qvals = qvalues.tdc(scores, target=self.targets, desc=desc)
        unlabeled = np.logical_and(qvals > self.eval_fdr, self.targets)
        new_labels = np.ones(len(qvals), dtype=np.byte)
        new_labels[~self.targets] = -1
        new_labels[unlabeled] = 0
        return new_labels

    def _find_best_feature(self) -> None:
        """Get the best feature.

        Find the best feature to separate targets from decoys at the
        specified false-discovery rate threshold.

        Parameters
        ----------
        eval_fdr : float
            The false-discovery rate threshold used to define the
            best feature.
        """
        best_positives = 0
        for desc in (True, False):
            update = partial(self.update_labels, desc=desc)
            labs = self.data.select(
                pl.col(*self.schema.features).map_batches(update).explode()
            ).collect()

            num_passing = labs.select((pl.all() == 1).sum())
            best_feat = (
                num_passing.lazy()
                .melt()
                .sort("value")
                .tail(1)
                .collect()[0, "variable"]
            )

            num_passing = num_passing[0, best_feat]

            if num_passing > best_positives:
                best_positives = num_passing
                self.schema.score = best_feat
                self.schema.desc = desc

        if self.schema.score is None:
            raise RuntimeError("No PSMs found below the 'eval_fdr'.")

    def calibrate_scores(
        self,
        scores: np.ndarray,
        desc: bool = True,
    ) -> np.ndarray:
        """
        Calibrate scores as described in Granholm et al. [1]_.

        .. [1] Granholm V, Noble WS, Käll L. A cross-validation scheme
           for machine learning algorithms in shotgun proteomics. BMC
           Bioinformatics. 2012;13 Suppl 16(Suppl 16):S3.
           doi:10.1186/1471-2105-13-S16-S3

        Parameters
        ----------
        scores : numpy.ndarray
            The scores for each PSM.
        eval_fdr: float, optional
            The FDR threshold to use for calibration
        desc: bool, optional
            Are higher scores better?

        Returns
        -------
        numpy.ndarray
            An array of calibrated scores.
        """
        labels = self.update_labels(scores, desc)
        pos = labels == 1
        if not pos.sum():
            raise RuntimeError(
                "No target PSMs were below the 'eval_fdr' threshold."
            )

        target_score = np.min(scores[pos])
        decoy_score = np.median(scores[labels == -1])

        return (scores - target_score) / (target_score - decoy_score)

    def _create_folds(self, folds: int) -> None:
        """Create the data splits for cross-validation.

        Each fold contains examples that are stratified by specific columns.
        For example, PSMs are stratefied by spectra. After folds are created
        they can be accessed with the :code:`fold()` method.

        Parameters
        ----------
        folds: int
            The number of splits to generate.
        """
        groups = (
            self.data.select(self._stratification)
            .unique()
            .collect(streaming=True)
            .sample(
                fraction=1,
                shuffle=True,
                seed=self.rng.integers(1, 10000),
            )
        )

        # The numbers in each fold:
        nums = [math.ceil(len(groups) / folds)] * (folds - 1)
        nums.append(len(groups) - sum(nums))

        # The offsets for each:
        stops = list(np.array(nums).cumsum())

        self._folds = [
            groups.lazy().head(i).tail(n).collect()
            for i, n in zip(stops, nums)
        ]

    def fold(self, index: int, train: bool, folds: int = 3) -> PsmDataset:
        """Load a cross-validation fold.

        Parameters
        ----------
        index : int
            The cross-validation fold to load. Must be a value in [1, `folds`]
        train : bool
            Return the training set for the fold? Otherwise, the test set is
            returned.
        folds : int, optional
            The number of cross-validation folds to create.
        """
        if self._folds is None or len(self._folds) != folds:
            self._create_folds(folds)

        fold_data = self.data.join(
            self._folds[index].lazy(),
            on=self._stratification,
            how="anti" if train else "inner",
        )

        if train and self.subset is not None:
            fold_data = fold_data.head(self.subset)

        new_dset = copy.copy(self)
        new_dset._data = fold_data
        new_dset._best_feature = None
        new_dset._folds = None
        new_dset._len = None
        return new_dset

    def assign_confidence(
        self,
        scores: str | pl.Series | np.ndarray | None = None,
        desc: bool = True,
    ) -> Confidence:
        """Assign confidence estimates.

        If :code:`scores` is not provided, the best feature
        or the specifided score in the data will be used.

        Parameters
        ----------
        scores : str, polars.Series, or numpy.ndarray, optionl
            The scores by which to rank examples.
        desc : bool
            Are higher scores better?

        Returns
        -------
        Confidence
            The confidence estimates at various levels.
        """
        if scores is None:
            feat, desc = self.best_feature
            scores = self.data.select(feat).collect(streaming=True).to_series()

        return PsmConfidence(
            data=self.data,
            schema=self.schema,
            scores=scores,
            desc=desc,
            proteins=self.proteins,
            eval_fdr=self.eval_fdr,
            rng=self.rng,
        )


class PsmDataset(_BaseDataset):
    """Store and analyze a collection of PSMs.

    Store a collection of PSMs from data-dependent acquisition proteomics
    experiments and and pepare them for mokapot analysis.

    Parameters
    ----------
    data : polars.DataFrame, polars.LazyFrame, or pandas.DataFrame
        A collection of PSMs, where the rows are PSMs and the columns are
        features or metadata describing them.
    schema : mokapot.PsmSchema
        The meaning of the columns in the data.
    proteins : mokapot.Proteins, optional
        The proteins to use for protein-level confidence estimation. This
        may be created with :py:func:`mokapot.read_fasta()`.
    eval_fdr : float, optional
        The false discovery rate threshold for choosing the best feature and
        creating positive labels during the trainging procedure.
    subset: int, optional
        The maximum number of examples to use for training.
    rng : int or np.random.Generator, optional
        A seed or generator used for cross-validation split creation and to
        break ties, or :code:`None` to use the default random number
        generator state.

    Attributes
    ----------
    data : polars.LazyFrame
    schema : PsmSchema
        The meaning of the columns in the data.
    columns : list of str
    targets : numpy.ndarray
    features : numpy.ndarray
    proteins : mokapot.Proteins
    best_feature : mokapot.dataset.BestFeature
    subset : int or None
        The maximum number of PSMs to use for training.
    rng : numpy.random.Generator
    """

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        schema: PsmSchema,
        proteins: Proteins | None = None,
        eval_fdr: float = 0.01,
        subset: int | None = None,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize a PsmDataset object."""
        super().__init__(
            data=data,
            schema=schema,
            proteins=proteins,
            eval_fdr=eval_fdr,
            rng=rng,
            unit="PSMs",
            stratification=schema.spectrum,
            subset=subset,
        )

    def __repr__(self) -> str:
        """How to print the class."""
        targets = self.targets
        n_spectra = (
            self.data.select(self.schema.spectrum)
            .unique()
            .select(pl.count())
            .collect(streaming=True)
            .item()
        )

        n_peptides = (
            self.data.select(self.schema.peptide)
            .unique()
            .select(pl.count())
            .collect(streaming=True)
            .item()
        )

        features = "\n      ".join(self.schema.features)

        return (
            f"A mokapot.PsmDataset with {len(self)} PSMs:\n"
            "  - Protein confidence estimates enabled: "
            f"{self.proteins is not None}\n"
            f"  - Target PSMs: {targets.sum()}\n"
            f"  - Decoy PSMs: {(~targets).sum()}\n"
            f"  - Unique spectra: {n_spectra}\n"
            f"  - Unique peptides: {n_peptides}\n"
            f"  - Features:\n      {features}"
        )
