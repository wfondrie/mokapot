"""
The :py:class:`LinearPsmDataset` and :py:class:`CrossLinkedPsmDataset`
classes are used to define collections peptide-spectrum matches. The
:py:class:`LinearPsmDataset` class is suitable for most types of
data-dependent acquisition proteomics experiments, whereas the
:py:class:`CrossLinkedPsmDataset` is specifically designed for
collections of cross-linked PSMs (CSMs) originating from
cross-linking proteomics experiments.

Although either class can be constructed from a
:py:class:`pandas.DataFrame`, it is often easier to load the PSMs directly
from a file in the `Percolator tab-delimited format
<https://github.com/percolator/percolator/wiki/Interface#tab-delimited-file-format>`_
(also known as the Percolator input format, or "PIN") using the
:py:func:`mokapot.read_pin` function.

Instances of these classes are required to train a
:py:class:`mokapot.model.Model` object, use the :py:func:`mokapot.brew`
function, or :doc:`assign confidence estimates <confidence>`.
"""
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from . import qvalues
from . import utils
from .proteins import read_fasta
from .confidence import LinearConfidence, CrossLinkedConfidence
from .proteins import FastaProteins

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
class PsmDataset(ABC):
    """
    Store a collection of PSMs and their features.

    :meta private:
    """

    @property
    @abstractmethod
    def targets(self):
        """An array indicating whether each PSM is a target."""
        return

    @abstractmethod
    def assign_confidence(self, scores, desc):
        """
        Return how to assign confidence.

        Parameters
        ----------
        scores : numpy.ndarray
            An array of scores.
        desc : bool
            Are higher scores better?
        """
        return

    @abstractmethod
    def _update_labels(self, scores, eval_fdr, desc):
        """
        Return the label for each PSM, given it's score.

        This method is used during model training to define positive
        examples. These are traditionally the target PSMs that fall
        within a specified FDR threshold.

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
        numpy.ndarray
            The label of each PSM, where 1 indicates a positive example,
            -1 indicates a negative example, and 0 removes the PSM from
            training. Typically, 0 is reserved for targets, below a
            specified FDR threshold.
        """
        return

    def __init__(
        self, psms, spectrum_columns, feature_columns, other_columns, copy_data
    ):
        """Initialize an object"""
        self._data = psms.copy(deep=copy_data).reset_index(drop=True)
        self._proteins = None

        # Set columns
        self._spectrum_columns = utils.tuplize(spectrum_columns)

        if other_columns is not None:
            other_columns = utils.tuplize(other_columns)
        else:
            other_columns = ()

        # Check that all of the columns exist:
        used_columns = sum([other_columns, self._spectrum_columns], tuple())

        missing_columns = [c not in self.data.columns for c in used_columns]
        if not missing_columns:
            raise ValueError(
                "The following specified columns were not found: "
                f"{missing_columns}"
            )

        # Get the feature columns
        if feature_columns is None:
            self._feature_columns = tuple(
                c for c in self.data.columns if c not in used_columns
            )
        else:
            self._feature_columns = utils.tuplize(feature_columns)

        LOGGER.info("Using %i features:", len(self._feature_columns))
        for i, feat in enumerate(self._feature_columns):
            LOGGER.info("  (%i)\t%s", i + 1, feat)

        LOGGER.info("Found %i PSMs.", len(self._data))

    @property
    def data(self):
        """The full collection of PSMs as a :py:class:`pandas.DataFrame`."""
        return self._data

    def __len__(self):
        """Return the number of PSMs"""
        return len(self._data.index)

    @property
    def _metadata_columns(self):
        """A list of the metadata columns"""
        return tuple(
            c for c in self.data.columns if c not in self._feature_columns
        )

    @property
    def metadata(self):
        """A :py:class:`pandas.DataFrame` of the metadata."""
        return self.data.loc[:, self._metadata_columns]

    @property
    def features(self):
        """A :py:class:`pandas.DataFrame` of the features."""
        return self.data.loc[:, self._feature_columns]

    @property
    def spectra(self):
        """
        A :py:class:`pandas.DataFrame` of the columns that uniquely
        identify a mass spectrum.
        """
        return self.data.loc[:, self._spectrum_columns]

    @property
    def columns(self):
        """The columns of the dataset."""
        return self.data.columns.tolist()

    @property
    def has_proteins(self):
        """Has a FASTA file been added?"""
        return self._proteins is not None

    def add_proteins(self, proteins, **kwargs):
        """
        Add protein information to the dataset.

        Protein sequence information is required to compute protein-level
        confidence estimates using the picked-protein approach.

        Parameters
        ----------
        proteins : a FastaProteins object or str
            The :py:class:`mokapot.FastaProteins` object defines the mapping
            of peptides to proteins and the mapping of decoy proteins to their
            corresponding target proteins. Alternatively, a string specifying
            a FASTA file can be specified which will be parsed to define
            these mappings.
        **kwargs : dict
            If `proteins` is a string, then **kwargs are keyword arguments
            passed to the :py:class:`mokapot.FastaProteins` constructor.
        """
        if not isinstance(proteins, FastaProteins):
            proteins = FastaProteins(proteins, **kwargs)

        self._proteins = proteins

    def _find_best_feature(self, eval_fdr):
        """
        Find the best feature to separate targets from decoys at the
        specified false-discovery rate threshold.

        Parameters
        ----------
        eval_fdr : float
            The false-discovery rate threshold used to define the
            best feature.

        Returns
        -------
        A tuple of an str, int, and numpy.ndarray
        best_feature : str
            The name of the best feature.
        num_passing : int
            The number of accepted PSMs using the best feature.
        labels : numpy.ndarray
            The new labels defining positive and negative examples when
            the best feature is used.
        desc : bool
            Are high scores better for the best feature?
        """
        best_feat = None
        best_positives = 0
        new_labels = None
        for desc in (True, False):
            labs = self.features.apply(
                self._update_labels, eval_fdr=eval_fdr, desc=desc
            )

            num_passing = (labs == 1).sum()
            feat_idx = num_passing.idxmax()
            num_passing = num_passing[feat_idx]

            if num_passing > best_positives:
                best_positives = num_passing
                best_feat = feat_idx
                new_labels = labs.loc[:, feat_idx].values
                best_desc = desc

        if best_feat is None:
            raise RuntimeError("No PSMs found below the 'eval_fdr'.")

        return best_feat, best_positives, new_labels, best_desc

    def _calibrate_scores(self, scores, eval_fdr, desc=True):
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
        labels = self._update_labels(scores, eval_fdr, desc)
        target_score = np.min(scores[labels == 1])
        decoy_score = np.median(scores[labels == -1])

        return (scores - target_score) / (target_score - decoy_score)

    def _split(self, folds):
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
        cols = list(self._spectrum_columns)
        scans = list(self.data.groupby(cols, sort=False).indices.values())
        np.random.shuffle(scans)
        scans = list(scans)

        # Split the data evenly
        num = len(scans) // folds
        splits = [scans[i : i + num] for i in range(0, len(scans), num)]

        if len(splits[-1]) < num:
            splits[-2] += splits[-1]
            splits = splits[:-1]

        return tuple(utils.flatten(s) for s in splits)


class LinearPsmDataset(PsmDataset):
    """Store and analyze a collection of PSMs.

    Stores a collection of PSMs from data-dependent acquisition
    proteomics experiments and defines the necessary fields
    for mokapot analysis.

    Parameters
    ----------
    psms : pandas.DataFrame
        A collection of PSMs.
    target_column : str
        The column specifying whether each PSM is a target (`True`) or a
        decoy (`False`). This column will be coerced to boolean, so the
        specifying targets as `1` and decoys as `-1` will not work
        correctly.
    spectrum_columns : str or tuple of str
        The column(s) that collectively identify unique mass spectra.
        Multiple columns can be useful to avoid combining scans from
        multiple mass spectrometry runs.
    peptide_column : str
        The column that defines a unique peptide. Modifications should
        indicated either in square brackets :code:`[]` or parentheses
        :code:`()`. The exact modification format within those entities
        does not matter, so long as it is consistent.
    protein_column : str, optional
        The column that specifies which protein(s) the detected peptide
        might have originated from. This column is not used to compute
        protein-level confidence estimates (see :py:meth:`add_fasta()`).
    feature_columns : str or tuple of str, optional
        The column(s) specifying the feature(s) for mokapot analysis. If
        `None`, these are assumed to be all columns not specified in the
        previous parameters.
    copy_data : bool, optional
        If true, a deep copy of `psms` is created, so that changes to the
        original collection of PSMs is not propagated to this object. This
        uses more memory, but is safer since it prevents accidental
        modification of the underlying data.

    Attributes
    ----------
    data : pandas.DataFrame
    metadata : pandas.DataFrame
    features : pandas.DataFrame
    spectra : pandas.DataFrame
    peptides : pandas.DataFrame
    targets : numpy.ndarray
    columns : list of str
    has_proteins : bool
    """

    def __init__(
        self,
        psms,
        target_column,
        spectrum_columns,
        peptide_column,
        protein_column,
        feature_columns=None,
        copy_data=True,
    ):
        """Initialize a PsmDataset object."""
        self._target_column = target_column
        self._peptide_column = utils.tuplize(peptide_column)

        if protein_column is not None:
            self._protein_column = utils.tuplize(protein_column)
        else:
            self._protein_column = tuple()

        # Some error checking:
        if len(self._peptide_column) > 1:
            raise ValueError(
                "Only one column can be used for " "'peptide_column'"
            )

        # Finish initialization
        other_columns = sum(
            [
                utils.tuplize(self._target_column),
                self._peptide_column,
                self._protein_column,
            ],
            tuple(),
        )

        super().__init__(
            psms=psms,
            spectrum_columns=spectrum_columns,
            feature_columns=feature_columns,
            other_columns=other_columns,
            copy_data=copy_data,
        )

        self._data[target_column] = self._data[target_column].astype(bool)
        num_targets = (self.targets).sum()
        num_decoys = (~self.targets).sum()
        LOGGER.info(
            "  - %i target PSMs and %i decoy PSMs detected.",
            num_targets,
            num_decoys,
        )

        if not num_targets:
            raise ValueError("No target PSMs were detected.")
        if not num_decoys:
            raise ValueError("No decoy PSMs were detected.")
        if not self.data.shape[0]:
            raise ValueError("No PSMs were detected.")

    def __repr__(self):
        """How to print the class"""
        return (
            f"A mokapot.dataset.LinearPsmDataset with {len(self.data)} "
            "PSMs:\n"
            f"\t- Protein confidence estimates enabled: {self.has_proteins}\n"
            f"\t- Target PSMs: {self.targets.sum()}\n"
            f"\t- Decoy PSMs: {(~self.targets).sum()}\n"
            f"\t- Unique spectra: {len(self.spectra.drop_duplicates())}\n"
            f"\t- Unique peptides: {len(self.peptides.drop_duplicates())}\n"
            f"\t- Features: {self._feature_columns}"
        )

    @property
    def targets(self):
        """An array indicating whether each PSM is a target sequence."""
        return self.data[self._target_column].values

    @property
    def peptides(self):
        """A :py:class:`pandas.DataFrame` of the peptide column."""
        return self.data.loc[:, self._peptide_column]

    def _update_labels(self, scores, eval_fdr=0.01, desc=True):
        """
        Return the label for each PSM, given it's score.

        This method is used during model training to define positive
        examples, which are traditionally the target PSMs that fall
        within a specified FDR threshold.

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
            The label of each PSM, where 1 indicates a positive example,
            -1 indicates a negative example, and 0 removes the PSM from
            training. Typically, 0 is reserved for targets, below a
            specified FDR threshold.
        """
        qvals = qvalues.tdc(scores, target=self.targets, desc=desc)
        unlabeled = np.logical_and(qvals > eval_fdr, self.targets)
        new_labels = np.ones(len(qvals))
        new_labels[~self.targets] = -1
        new_labels[unlabeled] = 0
        return new_labels

    def assign_confidence(self, scores=None, desc=True, eval_fdr=0.01):
        """
        Assign confidence to PSMs and peptides.

        Two forms of confidence estimates are calculated: q-values,
        which are the minimum false discovery rate (FDR) at which a
        given PSMs would be accepted, and posterior error probabilities
        (PEPs), which probability that the given PSM is incorrect. For
        more information see the :doc:`Confidence Estimation
        <confidence>` page.

        Parameters
        ----------
        scores : numpy.ndarray
            The scores used to rank the PSMs. The default,
            :code:`None`, uses the feature that accepts the most
            PSMs at an FDR of `eval_fdr`.
        desc : bool
            Are higher scores better?
        eval_fdr : float
            The FDR threshold at which to report and evaluate
            performance. If `scores` is not :code:`None`, this
            parameter has no affect on the analysis itself, only on
            logging messages.

        Returns
        -------
        LinearConfidence
            A :py:class:`LinearConfidence` object storing the
            confidence estimates for the collection of PSMs.
        """
        if scores is None:
            feat, _, _, desc = self._find_best_feature(eval_fdr)
            LOGGER.info("Selected %s as the best feature.", feat)
            scores = self.features[feat].values

        return LinearConfidence(self, scores, eval_fdr=eval_fdr, desc=desc)


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
    target_column : tuple of str
        The columns specifying whether each peptide of a PSM is a target
        (`True`) or a decoy (`False`) sequence. These columns will be coerced
        to boolean, so the
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

    :meta private:
    """

    def __init__(
        self,
        psms: pd.DataFrame,
        spectrum_columns,
        target_columns,
        peptide_columns,
        protein_columns,
        feature_columns=None,
    ):
        """Initialize a PsmDataset object."""
        self._target_columns = utils.tuplize(target_columns)
        self._peptide_columns = tuple(
            utils.tuplize(c) for c in peptide_columns
        )
        self._protein_columns = tuple(
            utils.tuplize(c) for c in protein_columns
        )

        # Finish initialization
        other_columns = sum(
            [
                self._target_columns,
                *self._peptide_columns,
                *self._protein_columns,
            ],
            tuple(),
        )

        super().__init__(
            psms=psms,
            spectrum_columns=spectrum_columns,
            feature_columns=feature_columns,
            other_columns=other_columns,
        )

    @property
    def targets(self):
        """An array indicating whether each PSM is a target."""
        bool_targs = self.data.loc[:, self._target_columns].astype(bool)
        return bool_targs.sum(axis=1).values

    def _update_labels(self, scores, eval_fdr=0.01, desc=True):
        """
        Return the label for each PSM, given it's score.

        This method is used during model training to define positive
        examples, which are traditionally the target PSMs that fall
        within a specified FDR threshold.

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
            The label of each PSM, where 1 indicates a positive example,
            -1 indicates a negative example, and 0 removes the PSM from
            training. Typically, 0 is reserved for targets, below a
            specified FDR threshold.
        """
        qvals = qvalues.crosslink_tdc(
            scores, num_targets=self.targets, desc=desc
        )
        unlabeled = np.logical_and(qvals > eval_fdr, self.targets)
        new_labels = np.ones(len(qvals))
        new_labels[~self.targets] = -1
        new_labels[unlabeled] = 0
        return new_labels

    def assign_confidence(self, scores, desc=True):
        """
        Assign confidence to crosslinked PSMs and peptides.

        Two forms of confidence estimates are calculated: q-values,
        which are the minimum false discovery rate at which a given PSMs
        would be accepted, and posterior error probabilities (PEPs),
        which probability that the given PSM is incorrect. For more
        information see the :doc:`PsmConfidence <confidence>` page.

        Parameters
        ----------
        scores : numpy.ndarray
            The scores used to rank the PSMs.
        desc : bool
            Are higher scores better?

        Returns
        -------
        CrossLinkedPsmConfidence
            A :py:class:`CrossLinkedPsmConfidence` object storing the
            confidence for the provided PSMs.
        """
        return CrossLinkedConfidence(self, scores, desc)
