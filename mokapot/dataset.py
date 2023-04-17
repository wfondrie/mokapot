"""The :py:class:`LinearPsmDataset` class is used to define a collection
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
from .parsers.fasta import read_fasta
from .proteins import Proteins

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
        rng,
    ):
        """Initialize an object"""
        self._data = psms.copy(deep=copy_data).reset_index(drop=True)
        self._proteins = None
        self.rng = rng

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

    @property
    def rng(self):
        """The random number generator for model training."""
        return self._rng

    @rng.setter
    def rng(self, rng):
        """Set the random number generator"""
        self._rng = np.random.default_rng(rng)

    def add_proteins(self, proteins, **kwargs):
        """Add protein information to the dataset.

        Protein sequence information is required to compute protein-level
        confidence estimates using the picked-protein approach.

        Parameters
        ----------
        proteins : a Proteins object or str
            The :py:class:`~mokapot.proteins.Proteins` object defines the
            mapping of peptides to proteins and the mapping of decoy proteins
            to their corresponding target proteins. Alternatively, a string
            specifying a FASTA file can be specified which will be parsed to
            define these mappings.
        **kwargs : dict
            Keyword arguments to be passed to the
            :py:class:`mokapot.read_fasta()` function.
        """
        if not isinstance(proteins, Proteins):
            proteins = read_fasta(proteins, **kwargs)

        self._proteins = proteins

    def _targets_count_by_feature(self, desc, eval_fdr):
        """
        iterate over features and count the number of positive examples

        :param desc: bool
            Are high scores better for the best feature?
        :param eval_fdr: float
            The false discovery rate threshold to use.
        :return: pd.Series
            The number of positive examples for each feature.
        """
        return pd.Series(
            [
                (
                    self._update_labels(
                        self.data.loc[:, col],
                        eval_fdr=eval_fdr,
                        desc=desc,
                    )
                    == 1
                ).sum()
                for col in self._feature_columns
            ],
            index=self._feature_columns,
        )

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
            num_passing = self._targets_count_by_feature(desc, eval_fdr)
            feat_idx = num_passing.idxmax()
            num_passing = num_passing[feat_idx]

            if num_passing > best_positives:
                best_positives = num_passing
                best_feat = feat_idx
                new_labels = self._update_labels(
                    self.data.loc[:, feat_idx], eval_fdr=eval_fdr, desc=desc
                )
                best_desc = desc

        if best_feat is None:
            raise RuntimeError("No PSMs found below the 'eval_fdr'.")

        return best_feat, best_positives, new_labels, best_desc

    def _calibrate_scores(self, scores, eval_fdr, desc=True):
        calibrate_scores(
            scores=scores, eval_fdr=eval_fdr, desc=desc, targets=self.targets
        )


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
        specified in the other parameters.
    filename_column : str, optional
        The column specifying the mass spectrometry data file (e.g. mzML)
        containing each spectrum. This is required for some output formats,
        such as mzTab and FlashLFQ.
    scan_column : str, optional
        The column specifying the scan number for each spectrum. Each value
        in the column should be an integer. This is required for some output
        formats, such as mzTab.
    calcmass_column : str, optional
        The column specifying the theoretical monoisotopic mass of each
        peptide. This is required for some output formats, such as mzTab and
        FlashLFQ.
    expmass_column : str, optional
        The column specifying the measured neutral precursor mass. This is
        required for the some ouput formats, such as mzTab.
    rt_column : str, optional
        The column specifying the retention time of each spectrum, in seconds.
        This is required for some output formats, such as mzTab and FlashLFQ.
    charge_column : str, optional
        The column specifying the charge state of each PSM. This is required
        for some output formats, such as mzTab and FlashLFQ.
    copy_data : bool, optional
        If true, a deep copy of `psms` is created, so that changes to the
        original collection of PSMs is not propagated to this object. This uses
        more memory, but is safer since it prevents accidental modification of
        the underlying data.
    rng : int or np.random.Generator, optional
        A seed or generator used for cross-validation split creation and to
        break ties, or ``None`` to use the default random number generator
        state.
    enforce_checks : bool, optional
        If True, it is checked whether decoys and targets exist and an error is thrown
        when this is not the case. Per default this check is True, but for prediction
        for example this can be optionally turned off.

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
    rng : numpy.random.Generator
       The random number generator.
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
        filename_column=None,
        scan_column=None,
        calcmass_column=None,
        expmass_column=None,
        rt_column=None,
        charge_column=None,
        copy_data=True,
        rng=None,
        enforce_checks=True,
    ):
        """Initialize a PsmDataset object."""
        self._target_column = target_column
        self._peptide_column = peptide_column
        self._protein_column = protein_column

        self._optional_columns = {
            "filename": filename_column,
            "scan": scan_column,
            "calcmass": calcmass_column,
            "expmass": expmass_column,
            "rt": rt_column,
            "charge": charge_column,
        }

        # Finish initialization
        other_columns = [target_column, peptide_column]
        if protein_column is not None:
            other_columns.append(protein_column)

        for _, opt_column in self._optional_columns.items():
            if opt_column is not None:
                other_columns.append(opt_column)

        super().__init__(
            psms=psms,
            spectrum_columns=spectrum_columns,
            feature_columns=feature_columns,
            group_column=group_column,
            other_columns=other_columns,
            copy_data=copy_data,
            rng=rng,
        )

        self._data[target_column] = self._data[target_column].astype(bool)
        num_targets = (self.targets).sum()
        num_decoys = (~self.targets).sum()

        if not self.data.shape[0]:
            raise ValueError("No PSMs were detected.")
        elif enforce_checks:
            if not num_targets:
                raise ValueError("No target PSMs were detected.")
            if not num_decoys:
                raise ValueError("No decoy PSMs were detected.")

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
        return _update_labels(scores=scores, targets=self.targets, desc=desc)


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


class OnDiskPsmDataset:
    def __init__(
        self,
        filename,
        columns,
        target_column,
        spectrum_columns,
        peptide_column,
        protein_column,
        group_column,
        feature_columns,
        metadata_columns,
        filename_column,
        scan_column,
        specId_column,
        calcmass_column,
        expmass_column,
        rt_column,
        charge_column,
        spectra_dataframe,
    ):
        """Initialize a PsmDataset object."""
        self.filename = filename
        self.columns = columns
        self.target_column = target_column
        self.peptide_column = peptide_column
        self.protein_column = protein_column
        self.spectrum_columns = spectrum_columns
        self.group_column = group_column
        self.feature_columns = feature_columns
        self.metadata_columns = metadata_columns
        self.filename_column = filename_column
        self.scan_column = scan_column
        self.calcmass_column = calcmass_column
        self.expmass_column = expmass_column
        self.rt_column = rt_column
        self.charge_column = charge_column
        self.specId_column = specId_column
        self.spectra_dataframe = spectra_dataframe

    def calibrate_scores(self, scores, eval_fdr, desc=True):
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
        targets = read_file(
            self.filename,
            use_cols=self.target_column,
            target_column=self.target_column,
        )
        labels = _update_labels(scores, targets, eval_fdr, desc)
        pos = labels == 1
        if not pos.sum():
            raise RuntimeError(
                "No target PSMs were below the 'eval_fdr' threshold."
            )

        target_score = np.min(scores[pos])
        decoy_score = np.median(scores[labels == -1])

        return (scores - target_score) / (target_score - decoy_score)

    def _targets_count_by_feature(self, column, eval_fdr, desc):
        df = read_file(
            file_name=self.filename,
            use_cols=[column] + [self.target_column],
            target_column=self.target_column,
        )

        return (
            _update_labels(
                df.loc[:, self.columns],
                targets=df.loc[:, self.target_column],
                eval_fdr=eval_fdr,
                desc=desc,
            )
            == 1
        ).sum()

    def find_best_feature(self, eval_fdr):
        best_feat = None
        best_positives = 0
        new_labels = None
        for desc in (True, False):
            num_passing = pd.Series(
                [
                    self._targets_count_by_feature(
                        eval_fdr=eval_fdr,
                        column=c,
                        desc=desc,
                    )
                    for c in self.feature_columns
                ],
                index=self.feature_columns,
            )

            feat_idx = num_passing.idxmax()
            num_passing = num_passing[feat_idx]

            if num_passing > best_positives:
                best_positives = num_passing
                best_feat = feat_idx
                df = read_file(
                    file_name=self.filename,
                    use_cols=[best_feat, self.target_column],
                )

                new_labels = _update_labels(
                    scores=df.loc[:, best_feat],
                    targets=df[self.target_column],
                    eval_fdr=eval_fdr,
                    desc=desc,
                )
                best_desc = desc

        if best_feat is None:
            raise RuntimeError("No PSMs found below the 'eval_fdr'.")

        return best_feat, best_positives, new_labels, best_desc

    def update_labels(self, scores, target_column, eval_fdr=0.01, desc=True):
        df = read_file(
            file_name=self.filename,
            use_cols=[target_column],
            target_column=target_column,
        )
        return _update_labels(
            scores=scores,
            targets=df[target_column],
            eval_fdr=eval_fdr,
            desc=desc,
        )

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
        self.spectra_dataframe = self.spectra_dataframe[self.spectrum_columns]
        scans = list(
            self.spectra_dataframe.groupby(
                list(self.spectra_dataframe.columns), sort=False
            ).indices.values()
        )
        self.spectra_dataframe = None
        for indices in scans:
            rng.shuffle(indices)
        rng.shuffle(scans)
        scans = list(scans)

        # Split the data evenly
        num = len(scans) // folds
        splits = [scans[i : i + num] for i in range(0, len(scans), num)]

        if len(splits[-1]) < num:
            splits[-2] += splits[-1]
            splits = splits[:-1]

        return tuple(utils.flatten(s) for s in splits)


def _update_labels(scores, targets, eval_fdr=0.01, desc=True):
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
    qvals = qvalues.tdc(scores, target=targets, desc=desc)
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
    labels = _update_labels(scores, targets, eval_fdr, desc)
    pos = labels == 1
    if not pos.sum():
        raise RuntimeError(
            "No target PSMs were below the 'eval_fdr' threshold."
        )

    target_score = np.min(scores[pos])
    decoy_score = np.median(scores[labels == -1])

    return (scores - target_score) / (target_score - decoy_score)


def update_labels(file_name, scores, target_column, eval_fdr=0.01, desc=True):
    df = read_file(
        file_name=file_name,
        use_cols=[target_column],
        target_column=target_column,
    )
    return _update_labels(
        scores=scores,
        targets=df[target_column],
        eval_fdr=eval_fdr,
        desc=desc,
    )


def read_file(file_name, use_cols=None, target_column=None):
    with utils.open_file(file_name) as f:
        df = pd.read_csv(
            f, sep="\t", usecols=use_cols, index_col=False, on_bad_lines="skip"
        ).apply(pd.to_numeric, errors="ignore")
    if target_column:
        return utils.convert_targets_column(df, target_column)
    else:
        return df
