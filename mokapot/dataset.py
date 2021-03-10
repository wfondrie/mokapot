"""The :py:class:`LinearPsmDataset` classe is used to define a collection
peptide-spectrum matches. The :py:class:`LinearPsmDataset` class is suitable for
most types of data-dependent acquisition proteomics experiments.

Although the class can be constructed from a :py:class:`pandas.DataFrame`, it
is often easier to load the PSMs directly from a file in the `Percolator
tab-delimited format
<https://github.com/percolator/percolator/wiki/Interface#tab-delimited-file-format>`_
(also known as the Percolator input format, or "PIN") using the
:py:func:`~mokapot.read_pin()` function or from a PepXML file using the
:py:func:`~mokapot.read_pepxml()` function. If protein-level confidence
estimates are desired, make sure to use the
:py:meth:`~LinearPsmDataset.add_proteins()` method.

One of more instance of this class are required to use the
:py:func:`~mokapot.brew()` function.

"""
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from . import qvalues
from . import utils
from .proteins import FastaProteins
from .confidence import (
    LinearConfidence,
    CrosslinkConfidence,
    GroupedConfidence,
)

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
        self,
        psms,
        spectrum_columns,
        feature_columns,
        group_column,
        other_columns,
        copy_data,
    ):
        """Initialize an object"""
        self._data = psms.copy(deep=copy_data).reset_index(drop=True)
        self._proteins = None

        # Set columns
        self._spectrum_columns = utils.tuplize(spectrum_columns)
        self._group_column = group_column

        if other_columns is not None:
            other_columns = utils.tuplize(other_columns)
        else:
            other_columns = ()

        if group_column is not None:
            group_column = (group_column,)
        else:
            group_column = ()

        # Check that all of the columns exist:
        used_columns = sum(
            [other_columns, self._spectrum_columns, group_column], tuple()
        )

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

        # Check that features don't have missing values:
        na_mask = self.features.isna().any(axis=0)
        if na_mask.any():
            na_idx = np.where(na_mask)[0]
            keep_idx = np.where(~na_mask)[0]
            LOGGER.warning(
                "Missing values detected in the following features:"
            )
            for col in [self._feature_columns[i] for i in na_idx]:
                LOGGER.warning("  - %s", col)

            LOGGER.warning("Dropping features with missing values...")
            self._feature_columns = tuple(
                [self._feature_columns[i] for i in keep_idx]
            )

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
    def groups(self):
        """
        A :py:class:`pandas.Series` of the groups for confidence estimation.
        """
        return self.data.loc[:, self._group_column]

    @property
    def columns(self):
        """The columns of the dataset."""
        return self.data.columns.tolist()

    @property
    def has_proteins(self):
        """Has a FASTA file been added?"""
        return self._proteins is not None

    def add_proteins(self, proteins, **kwargs):
        """Add protein information to the dataset.

        Protein sequence information is required to compute protein-level
        confidence estimates using the picked-protein approach.

        Parameters
        ----------
        proteins : a FastaProteins object or str
            The :py:class:`mokapot.proteins.FastaProteins` object defines the
            mapping of peptides to proteins and the mapping of decoy proteins
            to their corresponding target proteins. Alternatively, a string
            specifying a FASTA file can be specified which will be parsed to
            define these mappings.
        **kwargs : dict
            Keyword arguments to be passed to the
            :py:class:`mokapot.proteins.FastaProteins` constructor.
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
        pos = labels == 1
        if not pos.sum():
            raise RuntimeError(
                "No target PSMs were below the 'eval_fdr' threshold."
            )

        target_score = np.min(scores[pos])
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

    Store a collection of PSMs from data-dependent acquisition proteomics
    experiments and and pepare them for mokapot analysis.

    Parameters
    ----------
    psms : pandas.DataFrame
        A collection of PSMs, where the rows are PSMs and the columns are
        features or metadata describing them.
    target_column : str
        The column specifying whether each PSM is a target (`True`) or a decoy
        (`False`). This column will be coerced to boolean, so the specifying
        targets as `1` and decoys as `-1` will not work correctly.
    spectrum_columns : str or tuple of str
        The column(s) that collectively identify unique mass spectra. Multiple
        columns can be useful to avoid combining scans from multiple mass
        spectrometry runs.
    peptide_column : str
        The column that defines a unique peptide. Modifications should be
        indicated either in square brackets :code:`[]` or parentheses
        :code:`()`. The exact modification format within these entities does
        not matter, so long as it is consistent.
    protein_column : str, optional
        The column that specifies which protein(s) the detected peptide might
        have originated from. This column is not used to compute protein-level
        confidence estimates (see :py:meth:`add_proteins()`).
    group_column : str, optional
        A factor by which to group PSMs for grouped confidence estimation.
    feature_columns : str or tuple of str, optional
        The column(s) specifying the feature(s) for mokapot analysis. If
        :code:`None`, these are assumed to be all of the columns that were not
        specified in the previous parameters.
    copy_data : bool, optional
        If true, a deep copy of `psms` is created, so that changes to the
        original collection of PSMs is not propagated to this object. This uses
        more memory, but is safer since it prevents accidental modification of
        the underlying data.

    Attributes
    ----------
    data : pandas.DataFrame
    metadata : pandas.DataFrame
    features : pandas.DataFrame
    spectra : pandas.DataFrame
    peptides : pandas.Series
    groups : pandas.Series
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
        protein_column=None,
        group_column=None,
        feature_columns=None,
        copy_data=True,
    ):
        """Initialize a PsmDataset object."""
        self._target_column = target_column
        self._peptide_column = peptide_column
        self._protein_column = protein_column

        # Finish initialization
        other_columns = [target_column, peptide_column]
        if protein_column is not None:
            other_columns += [protein_column]

        super().__init__(
            psms=psms,
            spectrum_columns=spectrum_columns,
            feature_columns=feature_columns,
            group_column=group_column,
            other_columns=other_columns,
            copy_data=copy_data,
        )

        self._data[target_column] = self._data[target_column].astype(bool)
        if pd.isna(self._data[target_column]).values.any():
            raise ValueError(
                "The 'target_column' could not be coerced to boolean."
            )

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
        """A :py:class:`numpy.ndarray` indicating whether each PSM is a target
        sequence.
        """
        return self.data[self._target_column].values

    @property
    def peptides(self):
        """A :py:class:`pandas.Series` of the peptide column."""
        return self.data.loc[:, self._peptide_column]

    def _update_labels(self, scores, eval_fdr=0.01, desc=True):
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
        qvals = qvalues.tdc(scores, target=self.targets, desc=desc)
        unlabeled = np.logical_and(qvals > eval_fdr, self.targets)
        new_labels = np.ones(len(qvals))
        new_labels[~self.targets] = -1
        new_labels[unlabeled] = 0
        return new_labels

    def assign_confidence(self, scores=None, desc=True, eval_fdr=0.01):
        """Assign confidence to PSMs peptides, and optionally, proteins.

        Two forms of confidence estimates are calculated: q-values---the
        minimum false discovery rate (FDR) at which a given PSM would be
        accepted---and posterior error probabilities (PEPs)---the probability
        that a given PSM is incorrect. For more information see the
        :doc:`Confidence Estimation <confidence>` page.

        Parameters
        ----------
        scores : numpy.ndarray
            The scores by which to rank the PSMs. The default, :code:`None`,
            uses the feature that accepts the most PSMs at an FDR threshold of
            `eval_fdr`.
        desc : bool
            Are higher scores better?
        eval_fdr : float
            The FDR threshold at which to report and evaluate performance. If
            `scores` is not :code:`None`, this parameter has no affect on the
            analysis itself, only on logging messages.

        Returns
        -------
        LinearConfidence
            A :py:class:`~mokapot.confidence.LinearConfidence` object storing
            the confidence estimates for the collection of PSMs.
        """
        if scores is None:
            feat, _, _, desc = self._find_best_feature(eval_fdr)
            LOGGER.info("Selected %s as the best feature.", feat)
            scores = self.features[feat].values

        if self._group_column is None:
            LOGGER.info("Assigning confidence...")
            return LinearConfidence(self, scores, eval_fdr=eval_fdr, desc=desc)
        else:
            LOGGER.info("Assigning confidence within groups...")
            return GroupedConfidence(
                self, scores, eval_fdr=eval_fdr, desc=desc
            )


class CrosslinkPsmDataset(PsmDataset):
    """Store and analyze a collection of CSMs

    Stores a collection of cross-linked peptide-spectrum matches (CSMs) from
    data-dependent acquisition cross-linking mass spectrometry experiments and
    defines the necessary fields for mokapot analysis.

    Parameters
    ----------
    csms : pandas.DataFrame
        A collection of CSMs. Each row should contain information about both
        peptides for a cross-linked peptide pair.
    target_columns : tuple of str
        A tuple of two strings that indicate the columns specifying if the
        :math:{\alpha}-peptide and :math:{\beta}-peptide of a cross-linked
        peptide pair is a target (`True`) or a decoy (`False`). These columns
        will be coerced to boolean, so specifying targets as `1` and decoys as
        `-1` will not work correctly.
    spectrum_columns : str or tuple of str
        The column(s) that collectively identify unique mass spectra.
        Multiple columns can be useful to avoid combining scans from
        multiple mass spectrometry runs.
    peptide_columns : tuple of str
        A tuple of two strings that indicate which columns define a unique
        peptide for the :math:{\alpha}-peptide and :math:{\beta}-peptide,
        respectively. Modifications should indicated either in square brackets
        :code:`[]` or parentheses :code:`()`. The exact modification format
        within those entities does not matter, so long as it is consistent.
    protein_columns : tuple of str, optional
        The columns that specify which protein(s) the detected peptide pair may
        have originated from. This column is not used to compute protein-level
        confidence estimates (see :py:meth:`add_proteins()`).
    group_column : str, optional
        A factor by which to group PSMs for grouped confidence estimation.
    feature_columns : str or tuple of str, optional
        The column(s) specifying the feature(s) for mokapot analysis. If
        `None`, these are assumed to be all columns not specified in the
        previous parameters.
    copy_data : bool, optional
        If true, a deep copy of `csms` is created, so that changes to the
        original collection of CSMs is not propagated to this object. This uses
        more memory, but is safer since it prevents accidental modification of
        the underlying data.

    Attributes
    ----------
    data : pandas.DataFrame
    metadata : pandas.DataFrame
    features : pandas.DataFrame
    spectra : pandas.DataFrame
    peptides : pandas.DataFrame
    combined_peptides: pandas.Series
    groups : pandas.Series
    targets : numpy.ndarray
    columns : list of str
    has_proteins : bool
    """

    def __init__(
        self,
        csms,
        spectrum_columns,
        target_columns,
        peptide_columns,
        protein_columns=None,
        group_column=None,
        feature_columns=None,
        copy_data=True,
    ):
        """Initialize a PsmDataset object."""
        self._target_columns = utils.tuplize(target_columns)
        self._peptide_columns = utils.tuplize(peptide_columns)

        check_len(self._target_columns, "target_columns", 2)
        check_len(self._peptide_columns, "peptide_columns", 2)

        # Finish initialization
        other_columns = sum(
            [list(self._target_columns), list(self._peptide_columns),], [],
        )

        if protein_columns is not None:
            self._protein_columns = utils.tuplize(protein_columns)
            other_columns += list(self._protein_columns)
        else:
            self._proteins_columns = None

        super().__init__(
            psms=csms,
            spectrum_columns=spectrum_columns,
            feature_columns=feature_columns,
            group_column=group_column,
            other_columns=other_columns,
            copy_data=copy_data,
        )

        self._data.loc[:, self._target_columns] = self._data.loc[
            :, self._target_columns
        ].astype(bool)
        if pd.isna(self._data.loc[:, self._target_columns]).values.any():
            raise ValueError(
                "The 'target_columns' could not be coerced to boolean."
            )

        num_targets = self.data[target_columns].sum(axis=1).values
        self._num_tt = (num_targets == 2).sum()
        self._num_td = (num_targets == 1).sum()
        self._num_dd = (num_targets == 0).sum()

        self._combined_peptides = None

        LOGGER.info(
            (
                "\t- %i target-target CSMs, %i target-decoy CSMs, and %i "
                "decoy-decoy CSMs detected"
            ),
            self._num_tt,
            self._num_td,
            self._num_dd,
        )

    def __repr__(self):
        """How to print the class"""
        num_spec = len(self.spectra.drop_duplicates())
        num_pep = len(self.combined_peptides.drop_duplicates())
        return (
            f"A mokapot.dataset.CrosslinkPsmDataset with {len(self.data)} "
            "CSMs:\n"
            f"\t- Protein confidence estimates enabled: {self.has_proteins}\n"
            f"\t- Target-target CSMs:   {self._num_tt}\n"
            f"\t- Target-decoy CSMs:    {self._num_td}\n"
            f"\t- Decoy-decoy CSMs:     {self._num_dd}\n"
            f"\t- Unique spectra:       {num_spec}\n"
            f"\t- Unique peptide pairs: {num_pep}\n"
            f"\t- Features: {self._feature_columns}"
        )

    @property
    def data(self):
        """The full collection of CSMs as a :py:class:`pandas.DataFrame`."""
        return super().data

    @property
    def combined_peptides(self):
        """Both peptides of a CSM as a single string."""
        if self._combined_peptides is not None and len(
            self._combined_peptides
        ) == len(self._data):
            return self._combined_peptides

        self._combined_peptides = []
        for _, peps in self.data.loc[:, self._peptide_columns].iterrows():
            self._combined_peptides.append("-".join(sorted(peps.tolist())))

        self._combined_peptides = pd.Series(self._combined_peptides)
        return self._combined_peptides

    @property
    def targets(self):
        """A :py:class:`numpy.ndarray` containing the number of target
        peptides  in each CSM.
        """
        return self.data.loc[:, self._target_columns].sum(axis=1).values

    @property
    def peptides(self):
        """A :py:class:`pandas.DataFrame` of the peptides from each CSM"""
        return self.data.loc[:, self._peptide_columns]

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
        new_labels[~(self.targets == 2)] = -1
        new_labels[unlabeled] = 0
        return new_labels

    def assign_confidence(self, scores=None, desc=True, eval_fdr=0.01):
        """Assign confidence to crosslinked CSMs and peptides.

        Two forms of confidence estimates are calculated: q-values---the
        minimum false discovery rate (FDR) at which a given CSM would be
        accepted---and posterior error probabilities (PEPs)---the probability
        that a given CSM is incorrect. For more information see the
        :doc:`Confidence Estimation <confidence>` page.

        Parameters
        ----------
        scores : numpy.ndarray
            The scores used to rank the CSMs. The default, :code:`None`, uses
            the feature that accepts the most CSMs at an FDR of `eval_fdr`.
        desc : bool
            Are higher scores better?
        eval_fdr : float
            The FDR threshold at which to report and evaluate performance. If
            `scores` is not :code:`None`, this parameter has no affect on the
            analysis itself, only on logging messages.

        Returns
        -------
        CrosslinkConfidence
            A :py:class:`CrosslinkConfidence` object storing the
            confidence for the provided CSMs.

        """
        if scores is None:
            feat, _, _, desc = self._find_best_feature(eval_fdr)
            LOGGER.info("Selected %s as the best feature.", feat)
            scores = self.features[feat].values

        if self._group_column is None:
            LOGGER.info("Assigning confidence...")
            return CrosslinkConfidence(
                self, scores, eval_fdr=eval_fdr, desc=desc
            )
        else:
            LOGGER.info("Assigin confidence withing groups...")
            return GroupedConfidence(
                self, scores, eval_fdr=eval_fdr, desc=desc
            )


# Utility Functions -----------------------------------------------------------
def check_len(var, label, expected=2):
    """
    Check that a variable is the expected length.

    Parameters
    ----------
    var : anything with a `__len__() method`
        The variable to assess.
    label : str
        The string to put in the error message.
    expected : int
        The expected length.
    """
    if len(var) != expected:
        raise ValueError(f"'{label}' must be of length {expected}")
