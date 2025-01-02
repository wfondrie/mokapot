from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Hashable

import numpy as np
import pandas as pd
from typeguard import typechecked

import mokapot.utils as utils
from mokapot.qvalues import tdc

from ..column_defs import ColumnGroups
from ..tabular_data import TabularDataReader


@dataclass
class BestFeatureProperties:
    name: str | Hashable
    positives: int
    fdr: float
    descending: bool


@dataclass
class LabeledBestFeature:
    feature: BestFeatureProperties
    new_labels: np.ndarray


class PsmDataset(ABC):
    """Store a collection of PSMs and their features.

    Note: Currently, the derived classes LinearPsmDataset and OnDiskPsmDataset
    don't have anything in common, so maybe this class can be removed in the
    future.
    """

    def __init__(
        self,
        rng,
    ):
        """Initialize a PsmDataset"""
        self._proteins = None
        self.rng = rng

    @property
    def rng(self):
        """The random number generator for model training."""
        return self._rng

    @rng.setter
    def rng(self, rng):
        """Set the random number generator"""
        self._rng = np.random.default_rng(rng)

    def get_column_names(self) -> tuple[str, ...]:
        """Return all columns available in the dataset."""
        return self.column_groups.columns

    @property
    def columns(self) -> list[str]:
        """Return all columns available in the dataset."""
        return list(self.column_groups.columns)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the feature columns.

        These are the columns that can be used to train models
        or used directly as confidence estimates. Classically
        this will be columns like search engine scores, one hot
        encoded charges, etc.
        """
        return self.column_groups.feature_columns

    @property
    def metadata_columns(self) -> tuple[str, ...]:
        """Returns all columns that are not features in a dataset."""
        # TODO: Move this to the column groups
        feats = self.feature_columns
        return tuple(c for c in self.get_column_names() if c not in feats)

    @property
    def confidence_level_columns(self) -> tuple[str, ...]:
        """Return the column names that can be used as levels.

        The levels are the multiple levels at which the data can be
        aggregated. For example, peptides, modified peptides, precursors,
        and peptide groups are levels.
        """
        use = [self.peptide_column] + list(self.extra_confidence_level_columns)
        return utils.tuplize(use)

    @property
    @abstractmethod
    def column_groups(self) -> ColumnGroups:
        raise NotImplementedError

    @property
    def extra_confidence_level_columns(self) -> tuple[str, ...]:
        return self.column_groups.extra_confidence_level_columns

    @property
    @abstractmethod
    def spectra_dataframe(self) -> pd.DataFrame:
        """Return the spectra dataframe.

        The spectra dataframe is meant to contain all information on
        the spectra but not the scores.
        """
        raise NotImplementedError

    @property
    def spectrum_columns(self) -> tuple[str, ...]:
        """Return the spectrum columns.

        The spectrum columns are the columns that uniquely identify a
        mass spectrum.

        Within mokapot these are the elements that will compete with each
        other in the confidence estimation process.
        """
        return self.column_groups.spectrum_columns

    @property
    def target_column(self) -> str:
        return self.column_groups.target_column

    @property
    @abstractmethod
    def target_values(self) -> np.ndarray[bool]:
        raise NotImplementedError

    @property
    def peptide_column(self) -> str:
        return self.column_groups.peptide_column

    @property
    @abstractmethod
    def reader(self) -> TabularDataReader:
        raise NotImplementedError

    @abstractmethod
    def get_default_extension(self) -> str:
        """Return the default extension as output.

        Returns the default extension used as an output
        for this type of reader.
        """
        # TODO: Move this to the writers.
        raise NotImplementedError

    @abstractmethod
    def _split(self, folds, rng):
        """
        Get the indices for random, even splits of the dataset.

        Each tuple of integers contains the indices for a random subset of
        PSMs. PSMs are grouped by spectrum, such that all PSMs from the same
        spectrum only appear in one split. The typical use for this method
        is to split the PSMs into cross-validation folds.

        Parameters
        ----------
        folds: int
            The number of splits to generate.

        Returns
        -------
        A tuple of tuples of ints
            Each of the returned tuples contains the indices  of PSMs in a
            split.
        """

        raise NotImplementedError

    @abstractmethod
    def read_data(
        self,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def read_data_chunked(
        self,
        *,
        chunk_size: int,
        columns: list[str] | None = None,
    ) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        raise NotImplementedError

    @abstractmethod
    def find_best_feature(self, eval_fdr: float) -> LabeledBestFeature:
        raise NotImplementedError

    @property
    @abstractmethod
    def scores(self) -> np.ndarray | None:
        # q: should i rename this to mokapot_scores?
        raise NotImplementedError

    @scores.setter
    @abstractmethod
    def scores(self, scores: np.ndarray | None):
        raise NotImplementedError


@typechecked
def update_labels(
    scores: np.ndarray[float] | pd.Series,
    targets: np.ndarray[bool] | pd.Series,
    eval_fdr: float = 0.01,
    desc: bool = True,
) -> np.ndarray[bool]:
    """Return the label for each PSM, given it's score.

    This method is used during model training to define positive examples,
    which are traditionally the target PSMs that fall within a specified
    FDR threshold.

    Parameters
    ----------
    scores : numpy.ndarray
        The score used to rank the PSMs.
    eval_fdr : float
        The false discovery rate threshold to use.
    desc : bool
        Are higher scores better?

    Returns
    -------
    np.ndarray
        The label of each PSM, where 1 indicates a positive example, -1
        indicates a negative example, and 0 removes the PSM from training.
        Typically, 0 is reserved for targets, below a specified FDR
        threshold.
    """
    if isinstance(scores, pd.Series):
        scores = scores.values.astype(float)
    if isinstance(targets, pd.Series):
        targets = targets.values.astype(bool)

    qvals = tdc(scores, target=targets, desc=desc)
    unlabeled = np.logical_and(qvals > eval_fdr, targets)
    new_labels = np.ones(len(qvals))
    new_labels[~targets] = -1
    new_labels[unlabeled] = 0
    return new_labels


def calibrate_scores(scores, targets, eval_fdr, desc=True):
    """
    Calibrate scores as described in Granholm et al. [1]_

    .. [1] Granholm V, Noble WS, Käll L. A cross-validation scheme
       for machine learning algorithms in shotgun proteomics. BMC
       Bioinformatics. 2012;13 Suppl 16(Suppl 16):S3.
       doi:10.1186/1471-2105-13-S16-S3

    Parameters
    ----------
    scores : numpy.ndarray
        The scores for each PSM.
    eval_fdr: float
        The FDR threshold to use for calibration
    desc: bool
        Are higher scores better?

    Returns
    -------
    numpy.ndarray
        An array of calibrated scores.
    """
    labels = update_labels(scores, targets, eval_fdr, desc)
    pos = labels == 1
    if not pos.sum():
        raise RuntimeError(
            "No target PSMs were below the 'eval_fdr' threshold."
        )

    target_score = np.min(scores[pos])
    decoy_score = np.median(scores[labels == -1])

    return (scores - target_score) / (target_score - decoy_score)
