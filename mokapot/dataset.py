"""
This module contains the classes and methods needed to import, validate and
normalize a collection of PSMs in PIN (Percolator INput) format.
"""
from __future__ import annotations
import logging
import random
from typing import List, Union, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import mokapot.qvalues as qvalues
import mokapot.utils as utils
from mokapot.confidence import PsmConfidence, LinearPsmConfidence

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
class PsmDataset(ABC):
    """
    Store a collection of PSMs and their features.

    :meta private:
    """
    @property
    @abstractmethod
    def targets(self) -> np.ndarray:
        """An array indicating whether each PSM is a target."""
        return

    @abstractmethod
    def assign_confidence(self, scores: np.ndarray, desc: bool) \
            -> PsmConfidence:
        """Return how to assign confidence."""
        return

    @abstractmethod
    def update_labels(self,
                      scores: np.ndarray,
                      fdr_threshold: float = 0.01,
                      desc: bool = True) \
            -> np.ndarray:
        """
        Return the label for each PSM, given it's score.

        This method is used during model training to define positive
        examples. These are traditionally the target PSMs that fall
        within a specified FDR threshold.

        Parameters
        ----------
        scores : numpy.ndarray
            The score used to rank the PSMs.

        fdr_threshold : float
            The false discovery rate threshold to use.

        desc : bool
            Are higher scores better?

        Returns
        -------
        np.ndarray
            The label of each PSM, where 1 indicates a positive example,
            -1 indicates a negative example, and 0 removes the PSM from
            training. Typically, 0 is reserved for targets, below a
            specified FDR threshold.
        """
        return

    def __init__(self,
                 psms: pd.DataFrame,
                 spectrum_columns: Union[str, Tuple[str, ...]],
                 experiment_columns: Union[str, Tuple[str, ...], None],
                 feature_columns: Union[str, Tuple[str, ...], None],
                 other_columns: Union[str, Tuple[str, ...], None]) \
            -> None:
        """Initialize an object"""
        self.data = psms

        # Set columns
        self.spectrum_columns = utils.tuplize(spectrum_columns)
        self.feature_columns = feature_columns

        if experiment_columns is not None:
            self.experiment_columns = utils.tuplize(experiment_columns)
        else:
            self.experiment_columns = ()

        if other_columns is not None:
            other_columns = utils.tuplize(other_columns)
        else:
            other_columns = ()

        # Check that all of the columns exist:
        used_columns = sum([other_columns,
                            self.spectrum_columns,
                            self.experiment_columns],
                           tuple())

        missing_columns = [c not in self.data.columns for c in used_columns]
        if not missing_columns:
            raise ValueError("The following specified columns were not found: "
                             f"{missing_columns}")

        # Get the feature columns
        if feature_columns is None:
            feature_columns = tuple(c for c in self.data.columns
                                    if c not in used_columns)
        else:
            feature_columns = tuple(feature_columns)

    @property
    def metadata_columns(self) -> Tuple[str, ...]:
        """A tuple of the metadata columns"""
        return tuple(c for c in self.data.columns
                     if c not in self.feature_columns)

    @property
    def metadata(self) -> pd.DataFrame:
        """A pandas DataFrame of the metadata."""
        return self.data.loc[:, self.metadata_columns]

    @property
    def features(self) -> pd.DataFrame:
        """A pandas DataFrame of the features."""
        return self.data.loc[:, self.feature_columns]

    @property
    def spectra(self) -> pd.DataFrame:
        """
        A pandas DataFrame of the columns that uniquely identify a
        mass spectrum.
        """
        return self.data.loc[:, self.spectrum_columns]

    @property
    def columns(self) -> List[str]:
        """The columns of the dataset."""
        return self.data.columns.tolist()

    def find_best_feature(self, fdr_threshold: float) \
            -> Tuple[str, int, np.ndarray]:
        """
        Find the best feature to separate targets from decoys at the
        specified false-discovery rate threshold.

        Parameters
        ----------
        fdr_threshold : float
            The false-discovery rate threshold used to define the
            best feature.

        Returns
        -------
        A tuple of an str, int, and numpy.ndarray
            The contains the name of the best feature, the number of
            positive examples found if that feature is used, and an
            array containing the labels defining positive and negative
            examples when that feature is used.
        """
        best_feat = None
        best_positives = 0
        new_labels = None
        for desc in (True, False):
            labs = self.features.apply(self.update_labels,
                                       fdr_threshold=fdr_threshold,
                                       desc=desc)

            num_passing = (labs == 1).sum()
            feat_idx = num_passing.idxmax()
            num_passing = num_passing[feat_idx]

            if num_passing > best_positives:
                best_positives = num_passing
                best_feat = feat_idx
                new_labels = labs.loc[:, feat_idx].values

        if best_feat is None:
            raise RuntimeError("No PSMs found below the fdr_threshold.")

        return best_feat, best_positives, new_labels

    def calibrate_scores(self, scores: np.ndarray, fdr_threshold: float,
                         desc: bool = True) -> np.ndarray:
        """Calibrate scores"""
        labels = self.update_labels(scores, fdr_threshold, desc)
        target_score = np.min(scores[labels == 1])
        decoy_score = np.median(scores[labels == -1])

        return (scores - target_score) / (target_score - decoy_score)

    def split(self, folds: int) -> Tuple[Tuple[int, ...], ...]:
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
        cols = list(self.spectrum_columns)
        scans = list(self.data.groupby(cols, sort=False).indices.values())
        np.random.shuffle(scans)
        scans = list(scans)

        # Split the data evenly
        num = len(scans) // folds
        splits = [scans[i:i+num] for i in range(0, len(scans), num)]

        if len(splits[-1]) < num:
            splits[-2] += splits[-1]
            splits = splits[:-1]

        return tuple(utils.flatten(s) for s in splits)


class LinearPsmDataset(PsmDataset):
    """
    Store and analyze a collection of PSMs

    A `PsmDataset` is intended to store a collection of PSMs from
    standard, data-dependent acquisition proteomics experiments and
    defines the necessary fields for mokapot analysis.

    Parameters
    ----------
    psms : pandas.DataFrame
        A collection of PSMs.

    target_column : str
        The column specifying whether each PSM is a target (`True`) or a
        decoy (`False`). This column will be coerced to boolean, so the
        specifying targets as `1` and decoys as `-1` will not work correctly.

    spectrum_columns : str or tuple of str
        The column(s) that collectively identify unique mass spectra.
        Multiple columns can be useful to avoid combining scans from
        multiple mass spectrometry runs.

    peptide_columns : str or tuple of str
        The column(s) that collectively define a peptide. Multiple
        columns may be useful if sequence and modifications are provided
        as separate columns.

    protein_column : str
        The column that specifies which protein(s) the detected peptide
        might have originated from. This column should contain a
        delimited list of protein identifiers that match the FASTA file
        used for database searching.

    feature_columns : str or tuple of str
        The column(s) specifying the feature(s) for mokapot analysis. If
        `None`, these are assumed to be all columns not specified in the
        previous parameters.
    """
    def __init__(self,
                 psms: pd.DataFrame,
                 target_column: str,
                 spectrum_columns: Union[str, Tuple[str, ...]] = "scan",
                 peptide_columns: Union[str, Tuple[str, ...]] = "peptide",
                 protein_column: str = "protein",
                 experiment_columns: Union[str, Tuple[str, ...], None] = None,
                 feature_columns: Union[str, Tuple[str, ...], None] = None) \
            -> None:
        """Initialize a PsmDataset object."""
        self.target_column = target_column
        self.peptide_columns = utils.tuplize(peptide_columns)
        self.protein_column = utils.tuplize(protein_column)

        # Some error checking:
        if len(self.protein_column) > 1:
            raise ValueError("Only one column can be used for "
                             "'protein_column'.")

        # Finish initialization
        other_columns = sum([utils.tuplize(self.target_column),
                             self.peptide_columns,
                             self.protein_column],
                            tuple())

        super().__init__(psms=psms,
                         spectrum_columns=spectrum_columns,
                         experiment_columns=experiment_columns,
                         feature_columns=feature_columns,
                         other_columns=other_columns)

    @property
    def targets(self) -> np.ndarray:
        """An array indicating whether each PSM is a target."""
        return self.data[self.target_column].values.astype(bool)

    def update_labels(self,
                      scores: np.ndarray,
                      fdr_threshold: float = 0.01,
                      desc: bool = True) \
            -> np.ndarray:
        """
        Return the label for each PSM, given it's score.

        This method is used during model training to define positive
        examples, which are traditionally the target PSMs that fall
        within a specified FDR threshold.

        Parameters
        ----------
        scores : numpy.ndarray
            The score used to rank the PSMs.

        fdr_threshold : float
            The false discovery rate threshold to use.

        desc : bool
            Are higher scores better?

        Returns
        -------
        np.ndarray
            The label of each PSM, where 1 indicates a positive example,
            -1 indicates a negative example, and 0 removes the PSM from
            training. Typically, 0 is reserved for targets, below a
            specified FDR threshold.
        """
        qvals = qvalues.tdc(scores, target=self.targets, desc=desc)
        unlabeled = np.logical_and(qvals > fdr_threshold, self.targets)
        new_labels = np.ones(len(qvals))
        new_labels[~self.targets] = -1
        new_labels[unlabeled] = 0
        return new_labels

    def assign_confidence(self, scores: np.ndarray, desc: bool = True) \
            -> LinearPsmConfidence:
        """Assign Confidence for stuff"""
        return LinearPsmConfidence(self, scores, desc)


class CrossLinkedPsmDataset(PsmDataset):
    """
    Store and analyze a collection of PSMs

    A `PsmDataset` is intended to store a collection of PSMs from
    standard, data-dependent acquisition proteomics experiments and
    defines the necessary fields for mokapot analysis.

    Parameters
    ----------
    psms : pandas.DataFrame
        A collection of PSMs.

    target_column : str
        The column specifying whether each PSM is a target (`True`) or a
        decoy (`False`). This column will be coerced to boolean, so the
        specifying targets as `1` and decoys as `-1` will not work correctly.

    spectrum_columns : str or tuple of str
        The column(s) that collectively identify unique mass spectra.
        Multiple columns can be useful to avoid combining scans from
        multiple mass spectrometry runs.

    peptide_columns : str or tuple of str
        The column(s) that collectively define a peptide. Multiple
        columns may be useful if sequence and modifications are provided
        as separate columns.

    protein_column : str
        The column that specifies which protein(s) the detected peptide
        might have originated from. This column should contain a
        delimited list of protein identifiers that match the FASTA file
        used for database searching.

    feature_columns : str or tuple of str
        The column(s) specifying the feature(s) for mokapot analysis. If
        `None`, these are assumed to be all columns not specified in the
        previous parameters.
    """
    def __init__(self,
                 psms: pd.DataFrame,
                 target_column: str,
                 spectrum_columns: Union[str, Tuple[str, ...]] = "scan",
                 peptide_columns: Union[str, Tuple[str, ...]] = "peptide",
                 protein_column: str = "protein",
                 experiment_columns: Union[str, Tuple[str, ...], None] = None,
                 feature_columns: Union[str, Tuple[str, ...], None] = None) \
            -> None:
        """Initialize a PsmDataset object."""
        self.target_column = target_column
        self.peptide_columns = utils.tuplize(peptide_columns)
        self.protein_column = utils.tuplize(protein_column)

        # Some error checking:
        if len(self.protein_column) > 1:
            raise ValueError("Only one column can be used for "
                             "'protein_column'.")

        # Finish initialization
        other_columns = sum([utils.tuplize(self.target_column),
                             self.peptide_columns,
                             self.protein_column],
                            tuple())

        super().__init__(psms=psms,
                         spectrum_columns=spectrum_columns,
                         experiment_columns=experiment_columns,
                         feature_columns=feature_columns,
                         other_columns=other_columns)

    @property
    def targets(self) -> np.ndarray:
        """An array indicating whether each PSM is a target."""
        return self.data[self.target_column].values.astype(bool)

    def update_labels(self,
                      scores: np.ndarray,
                      fdr_threshold: float = 0.01,
                      desc: bool = True) \
            -> np.ndarray:
        """
        Return the label for each PSM, given it's score.

        This method is used during model training to define positive
        examples, which are traditionally the target PSMs that fall
        within a specified FDR threshold.

        Parameters
        ----------
        scores : numpy.ndarray
            The score used to rank the PSMs.

        fdr_threshold : float
            The false discovery rate threshold to use.

        desc : bool
            Are higher scores better?

        Returns
        -------
        np.ndarray
            The label of each PSM, where 1 indicates a positive example,
            -1 indicates a negative example, and 0 removes the PSM from
            training. Typically, 0 is reserved for targets, below a
            specified FDR threshold.
        """
        qvals = qvalues.tdc(scores, target=self.targets, desc=desc)
        unlabeled = np.logical_and(qvals > fdr_threshold, self.targets)
        new_labels = np.ones(len(qvals))
        new_labels[~self.targets] = -1
        new_labels[unlabeled] = 0
        return new_labels

    def assign_confidence(self, scores: np.ndarray, desc: bool = True) \
            -> LinearPsmConfidence:
        """Assign Confidence for stuff"""
        return LinearPsmConfidence(self, scores, desc)
