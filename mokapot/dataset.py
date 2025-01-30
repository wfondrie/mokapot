"""The :py:class:`LinearPsmDataset` class is used to define a collection
peptide-spectrum matches. The :py:class:`LinearPsmDataset` class is suitable
for most types of data-dependent acquisition proteomics experiments.

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
from abc import ABC
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd
from typeguard import typechecked

from mokapot import utils
from mokapot.parsers.fasta import read_fasta
from mokapot.proteins import Proteins
from mokapot.stats.algorithms import QvalueAlgorithm
from mokapot.tabular_data import TabularDataReader

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
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


@typechecked
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
    targets : numpy.ndarray
    columns : list of str
    has_proteins : bool
    rng : numpy.random.Generator
       The random number generator.
    """  # noqa: E501

    def __init__(
        self,
        psms,
        target_column,
        spectrum_columns,
        peptide_column,
        protein_column=None,
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
        """Initialize a LinearPsmDataset object."""
        super().__init__(rng=rng)
        self._data = psms.copy(deep=copy_data).reset_index(drop=True)

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
                "The following specified columns were not found: " f"{missing_columns}"
            )

        # Get the feature columns
        if feature_columns is None:
            self._feature_columns = tuple(
                c for c in self.data.columns if c not in used_columns
            )
        else:
            self._feature_columns = utils.tuplize(feature_columns)

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

    @property
    def data(self):
        """The full collection of PSMs as a :py:class:`pandas.DataFrame`."""
        return self._data

    def __len__(self):
        """Return the number of PSMs"""
        return len(self._data.index)

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
        return self.data[self._target_column].values == 1

    @property
    def peptides(self):
        """A :py:class:`pandas.Series` of the peptide column."""
        return self.data.loc[:, self._peptide_column]

    def _update_labels(
        self, scores: np.ndarray[float], eval_fdr: float = 0.01, desc: bool = True
    ):
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
        return _update_labels(
            scores=scores,
            targets=self.targets,
            eval_fdr=eval_fdr,
            desc=desc,
        )

    @property
    def _metadata_columns(self):
        """A list of the metadata columns"""
        return tuple(c for c in self.data.columns if c not in self._feature_columns)

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
        num_passed_per_col = []
        for col in self._feature_columns:
            scores = self.data.loc[:, col].values.astype(float)
            labels = self._update_labels(scores, eval_fdr=eval_fdr, desc=desc)
            num_passed = (labels == 1).sum()
            num_passed_per_col.append(num_passed)
        return pd.Series(num_passed_per_col, index=self._feature_columns)

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
        LOGGER.debug(
            f"\t- Finding best feature in {len(self._feature_columns)} feature columns."
        )
        best_feat = None
        best_positives = 0
        new_labels = None
        for desc in (True, False):
            num_passing = self._targets_count_by_feature(desc, eval_fdr)
            for col, num in num_passing.items():
                LOGGER.debug(
                    f"\t  - Column {col} desc={desc}: found {num} passing PSMs."
                )
            feat_idx = num_passing.idxmax()
            num_passing = num_passing[feat_idx]

            if num_passing > best_positives:
                best_positives = num_passing
                best_feat = feat_idx
                feat_scores = self.data.loc[:, feat_idx].values.astype(float)
                new_labels = self._update_labels(
                    feat_scores, eval_fdr=eval_fdr, desc=desc
                )
                best_desc = desc

        if best_feat is None:
            raise RuntimeError(f"No PSMs found below the 'eval_fdr' {eval_fdr}.")

        return best_feat, best_positives, new_labels, best_desc

    def _calibrate_scores(self, scores, eval_fdr, desc=True):
        calibrate_scores(
            scores=scores, eval_fdr=eval_fdr, desc=desc, targets=self.targets
        )


@typechecked
class OnDiskPsmDataset(PsmDataset):
    def __init__(
        self,
        filename_or_reader: Path | TabularDataReader,
        columns,
        target_column,
        spectrum_columns,
        peptide_column,
        protein_column,
        feature_columns,
        metadata_columns,
        metadata_column_types,
        level_columns,
        filename_column,
        scan_column,
        specId_column,
        calcmass_column,
        expmass_column,
        rt_column,
        charge_column,
        spectra_dataframe,
    ):
        """Initialize an OnDiskPsmDataset object."""
        super().__init__(rng=None)
        if isinstance(filename_or_reader, TabularDataReader):
            self.reader = filename_or_reader
        else:
            self.reader = TabularDataReader.from_path(filename_or_reader)

        self.columns = columns
        self.target_column = target_column
        self.peptide_column = peptide_column
        self.protein_column = protein_column
        self.spectrum_columns = spectrum_columns
        self.feature_columns = feature_columns
        self.metadata_columns = metadata_columns
        self.metadata_column_types = metadata_column_types
        self.level_columns = level_columns
        self.filename_column = filename_column
        self.scan_column = scan_column
        self.calcmass_column = calcmass_column
        self.expmass_column = expmass_column
        self.rt_column = rt_column
        self.charge_column = charge_column
        self.specId_column = specId_column
        self.spectra_dataframe = spectra_dataframe

        columns = self.reader.get_column_names()

        # todo: nice to have: here reader.file_name should be something like
        #   reader.user_repr() which tells the user where to look for the
        #   error, however, we cannot expect the reader to have a file_name
        def check_column(column):
            if column and column not in columns:
                file_name = getattr(self.reader, "file_name", "<unknown file>")
                raise ValueError(
                    f"Column '{column}' not found in data columns of file"
                    f" '{file_name}' ({columns})"
                )

        def check_columns(columns):
            if columns:
                for column in columns:
                    check_column(column)

        check_columns(self.columns)
        check_column(self.target_column)
        check_column(self.peptide_column)
        check_column(self.protein_column)
        check_columns(self.spectrum_columns)
        check_columns(self.feature_columns)
        check_columns(self.metadata_columns)
        check_columns(self.level_columns)
        check_column(self.filename_column)
        check_column(self.scan_column)
        check_column(self.calcmass_column)
        check_column(self.expmass_column)
        check_column(self.rt_column)
        check_column(self.charge_column)
        check_column(self.specId_column)

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
        targets = self.read_data(columns=self.target_column)
        targets = utils.convert_targets_column(targets, self.target_column)
        labels = _update_labels(scores, targets, eval_fdr, desc)
        pos = labels == 1
        if not pos.sum():
            raise RuntimeError("No target PSMs were below the 'eval_fdr' threshold.")

        target_score = np.min(scores[pos])
        decoy_score = np.median(scores[labels == -1])

        return (scores - target_score) / (target_score - decoy_score)

    def _targets_count_by_feature(self, column, eval_fdr, desc):
        df = self.read_data(
            columns=[column] + [self.target_column],
        )
        df = utils.convert_targets_column(df, self.target_column)
        return (
            _update_labels(
                df.loc[:, column],
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
                df = self.read_data(
                    columns=[best_feat, self.target_column],
                )

                new_labels = _update_labels(
                    scores=df.loc[:, best_feat],
                    targets=df[self.target_column],
                    eval_fdr=eval_fdr,
                    desc=desc,
                )
                best_desc = desc

        if best_feat is None:
            raise RuntimeError(f"No PSMs found below the 'eval_fdr' {eval_fdr}.")

        return best_feat, best_positives, new_labels, best_desc

    @staticmethod
    def _hash_row(x: np.ndarray) -> int:
        """
        Hash array for splitting of test/training sets.

        Parameters
        ----------
        x : np.ndarray
            Input array to be hashed.

        Returns
        -------
        int
            Computed hash of the input array.
        """

        def to_base_val(v):
            """Return base python value also for numpy types"""
            try:
                return v.item()
            except AttributeError:
                return v

        tup = tuple(to_base_val(x) for x in x)
        return crc32(str(tup).encode())

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
        spectra = self.spectra_dataframe[self.spectrum_columns].values
        del self.spectra_dataframe
        spectra = np.apply_along_axis(OnDiskPsmDataset._hash_row, 1, spectra)

        # sort values to get start position of unique hashes
        spectra_idx = np.argsort(spectra)
        spectra = spectra[spectra_idx]
        idx_start_unique = np.unique(spectra, return_index=True)[1]
        del spectra

        fold_size = len(spectra_idx) // folds
        remainder = len(spectra_idx) % folds
        start_split_indices = []
        start_idx = 0
        for i in range(folds - 1):
            end_idx = start_idx + fold_size + (1 if i < remainder else 0)
            start_split_indices.append(end_idx)
            start_idx = end_idx

        # search for smallest index bigger of equal to split index in start
        # indexes of unique groups
        idx_split = idx_start_unique[
            np.searchsorted(idx_start_unique, start_split_indices)
        ]
        del idx_start_unique
        spectra_idx = np.split(spectra_idx, idx_split)
        for indices in spectra_idx:
            rng.shuffle(indices)
        return spectra_idx

    def read_data(self, columns=None, chunk_size=None):
        if chunk_size:
            return self.reader.get_chunked_data_iterator(
                chunk_size=chunk_size, columns=columns
            )
        else:
            return self.reader.read(columns=columns)


@typechecked
def _update_labels(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    eval_fdr: float = 0.01,
    desc: bool = True,
) -> np.ndarray[float]:
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
        The label of each PSM, where 1 indicates a positive example (probably
        true target), -1 indicates a negative example (decoy), and 0 removes
        he PSM from training (probably false target). Typically, 0 is reserved
        for targets, below a specified FDR threshold.
    """

    qvals = QvalueAlgorithm.eval(scores, targets=targets, desc=desc)
    # not_passing = np.logical_and(qvals > eval_fdr, targets)

    new_labels = np.ones(len(qvals))
    new_labels[qvals > eval_fdr] = 0
    new_labels[~targets] = -1
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
        raise RuntimeError("No target PSMs were below the 'eval_fdr' threshold.")

    target_score = np.min(scores[pos])
    decoy_score = np.median(scores[labels == -1])

    return (scores - target_score) / (target_score - decoy_score)


@typechecked
def update_labels(
    reader: TabularDataReader,
    scores: np.ndarray[float],
    target_column: str,
    eval_fdr: float = 0.01,
    desc: bool = True,
):
    df = reader.read(columns=[target_column])
    return _update_labels(
        scores=scores,
        targets=df[target_column].values == 1,
        eval_fdr=eval_fdr,
        desc=desc,
    )
