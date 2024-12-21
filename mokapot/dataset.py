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

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from zlib import crc32
from typing import Generator, Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from typeguard import typechecked

from mokapot import qvalues
from mokapot import utils
from mokapot.column_defs import ColumnGroups, OptionalColumns
from .tabular_data import TabularDataReader, DataFrameReader

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------


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
    spectra_dataframe : pandas.DataFrame
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
        column_groups: ColumnGroups,
        copy_data: bool = True,
        rng: int | np.random.Generator | None = None,
        enforce_checks: bool = True,
    ):
        """Initialize a LinearPsmDataset object."""
        super().__init__(rng=rng)
        self._data = psms.copy(deep=copy_data).reset_index(drop=True)
        self._column_groups = column_groups
        self.make_bool_trarget()
        num_targets = (self.targets).sum()
        num_decoys = (~self.targets).sum()

        if not self.data.shape[0]:
            raise ValueError("No PSMs were detected.")
        elif enforce_checks:
            if not num_targets:
                raise ValueError("No target PSMs were detected.")
            if not num_decoys:
                raise ValueError("No decoy PSMs were detected.")

    @staticmethod
    def new(
        psms,
        target_column: str,
        peptide_column: str,
        spectrum_columns: list[str] | tuple[str, ...],
        feature_columns: list[str] | tuple[str, ...] | None = None,
        extra_confidence_level_columns: list[str]
        | tuple[str, ...]
        | None = None,
        copy_data: bool = True,
        rng: int | np.random.Generator | None = None,
        enforce_checks: bool = True,
        *,
        filename_column: str | None = None,
        scan_column: str | None = None,
        calcmass_column: str | None = None,
        expmass_column: str | None = None,
        rt_column: str | None = None,
        charge_column: str | None = None,
        protein_column: str | None = None,
    ) -> LinearPsmDataset:
        """Initialize a PsmDataset object from a ColumnGroups object."""

        if extra_confidence_level_columns is None:
            extra_confidence_level_columns = tuple()
        else:
            extra_confidence_level_columns = utils.tuplize(
                extra_confidence_level_columns
            )

        spectrum_columns = utils.tuplize(spectrum_columns)
        used_columns = (target_column, peptide_column)
        used_columns += spectrum_columns
        used_columns += extra_confidence_level_columns

        if feature_columns is not None:
            # Note ... I dont think this is needed anymore, since we are
            # typechecking the element passed as a list.
            used_columns += utils.tuplize(feature_columns)

        missing_columns = [
            c for c in set(used_columns) if c not in psms.columns
        ]

        if missing_columns:
            raise ValueError(
                "The following specified columns were not found: "
                f"{missing_columns}"
            )

        # Get the feature columns
        if feature_columns is None:
            feature_columns = []
            nonfeat_columns = []
            for c in psms.columns:
                # Add only if its not a string (int, float, bool)
                if c in used_columns:
                    continue
                elif not isinstance(psms[c], str):
                    feature_columns.append(c)
                else:
                    nonfeat_columns.append(c)
            feature_columns = tuple(feature_columns)
            LOGGER.info(
                f"Found {len(feature_columns)} feature columns: "
                f"{feature_columns}"
            )
            LOGGER.info(
                f"Found {len(nonfeat_columns)} non-feature columns: "
                f"{nonfeat_columns}"
            )
        else:
            feature_columns = utils.tuplize(feature_columns)

        column_groups = ColumnGroups(
            columns=psms.columns,
            target_column=target_column,
            peptide_column=peptide_column,
            spectrum_columns=spectrum_columns,
            feature_columns=feature_columns,
            extra_confidence_level_columns=extra_confidence_level_columns,
            optional_columns=OptionalColumns(
                filename=filename_column,
                scan=scan_column,
                calcmass=calcmass_column,
                expmass=expmass_column,
                rt=rt_column,
                charge=charge_column,
                protein=protein_column,
            ),
        )
        return LinearPsmDataset(
            psms=psms,
            column_groups=column_groups,
            copy_data=copy_data,
            rng=rng,
            enforce_checks=enforce_checks,
        )

    @property
    def reader(self) -> TabularDataReader:
        return DataFrameReader(self.data)

    @property
    def column_groups(self) -> ColumnGroups:
        return self._column_groups

    def get_column_names(self) -> tuple[str, ...]:
        return utils.tuplize(list(self.data.columns))

    def _split(self, folds, rng):
        inds = self.spectra_dataframe.index.to_numpy()
        splits = rng.integers(0, high=folds, size=len(inds))
        out = tuple([inds[splits == i] for i in range(folds)])
        return out

    def make_bool_trarget(self):
        out = self._make_bool_trarget(self._data[self.target_column])
        self._data[self.target_column] = out

    @staticmethod
    def _make_bool_trarget(target_column: pd.Series):
        """Convert target column to boolean if possible.

        1. If its a 0-1 col convert to bool
        2. If its a -1,1 values > 0 becomes True, rest False
        """
        if target_column.dtype == bool:
            return target_column

        if target_column.dtype == int or target_column.dtype == float:
            # Check if all values are 0 or 1
            uniq_vals = target_column.unique().tolist()
            if uniq_vals == [0, 1]:
                # If so, we can just cast to bool
                return target_column.astype(bool)
            elif uniq_vals == [-1, 1]:
                return target_column > 0
            else:
                raise ValueError(
                    f"Target column "
                    "has values that are not boolean or 0/1 or -1/1."
                    f"Please check and fix. (found {uniq_vals})"
                )

        # If not raise an error ... most likely the cast will be
        # Something the user does not want.
        raise ValueError(
            "Target column "
            "has values that are not boolean, 0/1 or -1/1,"
            f" please check and fix. ({target_column})"
        )

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
            f"\t- Target PSMs: {self.targets.sum()}\n"
            f"\t- Decoy PSMs: {(~self.targets).sum()}\n"
            "\t- Unique spectra: "
            f"{len(self.spectra_dataframe.drop_duplicates())}\n"
            f"\t- Unique peptides: {len(self.peptides.drop_duplicates())}\n"
            f"\t- Features: {self.feature_columns}"
        )

    @property
    def targets(self) -> np.ndarray[bool]:
        """A :py:class:`numpy.ndarray` indicating whether each PSM is a target
        sequence.
        """
        return self.data[self.target_column].values

    @property
    def peptides(self):
        """A :py:class:`pandas.Series` of the peptide column."""
        return self.data.loc[:, self._peptide_column]

    def _update_labels(self, scores, eval_fdr=0.01, desc=True):
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
            scores=scores, targets=self.targets, eval_fdr=eval_fdr, desc=desc
        )

    @property
    def features(self):
        """A :py:class:`pandas.DataFrame` of the features."""
        return self.data.loc[:, self.feature_columns]

    @property
    def spectra_dataframe(self):
        """
        A :py:class:`pandas.DataFrame` of the columns that uniquely
        identify a mass spectrum.
        """
        return self.data[self.spectrum_columns]

    @property
    def columns(self):
        """The columns of the dataset."""
        return self.data.columns.tolist()

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
                for col in self.feature_columns
            ],
            index=self.feature_columns,
        )

    def find_best_feature(self, eval_fdr: float) -> LabeledBestFeature:
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
            raise RuntimeError(
                f"No PSMs found below the 'eval_fdr' {eval_fdr}."
            )

        out = LabeledBestFeature(
            feature=BestFeatureProperties(
                name=str(best_feat),
                positives=int(best_positives),
                descending=best_desc,
                fdr=eval_fdr,
            ),
            new_labels=new_labels,
        )
        return out

    def _calibrate_scores(self, scores, eval_fdr, desc=True):
        calibrate_scores(
            scores=scores, eval_fdr=eval_fdr, desc=desc, targets=self.targets
        )

    @staticmethod
    def _yield_data_chunked(
        data: pd.DataFrame,
        chunk_size: int,
    ) -> Generator[pd.DataFrame, None, None]:
        if chunk_size:
            start = 0
            while True:
                end = start + chunk_size
                chunk = data.iloc[start:end]
                if len(chunk) == 0:
                    break
                yield chunk
                start = end

    def read_data(
        self,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        data = self.data
        if columns is not None:
            data = data[columns]

        return self.data

    def read_data_chunked(
        self,
        *,
        chunk_size: int,
        columns: list[str] | None = None,
    ) -> Generator[pd.DataFrame, None, None]:
        data = self.data
        if columns is not None:
            data = data[columns]

        return self._yield_data_chunked(data, chunk_size)

    def get_default_extension(self) -> str:
        return ".csv"

    @property
    def scores(self) -> np.ndarray | None:
        if not hasattr(self, "_scores"):
            return None
        return self._scores

    @scores.setter
    def scores(self, scores: np.ndarray | None):
        self._scores = scores


@typechecked
class OnDiskPsmDataset(PsmDataset):
    # Q: can we have a docstring here?
    def __init__(
        self,
        filename_or_reader: Path | TabularDataReader,
        column_groups: ColumnGroups,
        spectra_dataframe: pd.DataFrame,
    ):
        """Initialize an OnDiskPsmDataset object."""
        super().__init__(rng=None)
        if isinstance(filename_or_reader, TabularDataReader):
            self._reader = filename_or_reader
        else:
            self._reader = TabularDataReader.from_path(filename_or_reader)

        self._column_groups = column_groups
        self._spectra_dataframe = spectra_dataframe

        # todo: nice to have: here reader.file_name should be something like
        #   reader.user_repr() which tells the user where to look for the
        #   error, however, we cannot expect the reader to have a file_name
        def check_column(column):
            reader_columns = self.reader.get_column_names()
            if column and column not in reader_columns:
                file_name = getattr(self.reader, "file_name", "<unknown file>")
                raise ValueError(
                    f"Column '{column}' not found in data columns of file"
                    f" '{file_name}' ({reader_columns})"
                )

        def check_columns(columns):
            if columns:
                for column in columns:
                    check_column(column)

        check_columns(self.column_groups.columns)
        check_column(self.target_column)
        check_column(self.peptide_column)
        check_columns(self.spectrum_columns)
        check_columns(self.feature_columns)
        check_columns(self.metadata_columns)
        check_columns(self.confidence_level_columns)
        check_columns(self.extra_confidence_level_columns)

    @staticmethod
    def new(
        filename_or_reader: Path | TabularDataReader,
        *,
        target_column: str,
        peptide_column: str,
        spectrum_columns: list[str],
        feature_columns: list[str],
        extra_confidence_level_columns: list[str],
        spectra_dataframe: pd.DataFrame,
        protein_column: str | None = None,
        filename_column: str | None = None,
        scan_column: str | None = None,
        calcmass_column: str | None = None,
        expmass_column: str | None = None,
        rt_column: str | None = None,
        charge_column: str | None = None,
    ):
        """Initialize an OnDiskPsmDataset object."""
        if isinstance(filename_or_reader, TabularDataReader):
            reader = filename_or_reader
        else:
            reader = TabularDataReader.from_path(filename_or_reader)

        columns = reader.get_column_names()

        column_groups = ColumnGroups(
            columns=utils.tuplize(columns),
            target_column=target_column,
            peptide_column=peptide_column,
            spectrum_columns=utils.tuplize(spectrum_columns),
            feature_columns=utils.tuplize(feature_columns),
            extra_confidence_level_columns=utils.tuplize(
                extra_confidence_level_columns
            ),
            optional_columns=OptionalColumns(
                filename=filename_column,
                scan=scan_column,
                calcmass=calcmass_column,
                expmass=expmass_column,
                rt=rt_column,
                charge=charge_column,
                protein=protein_column,
            ),
        )

        return OnDiskPsmDataset(
            reader,
            column_groups=column_groups,
            spectra_dataframe=spectra_dataframe,
        )

    def get_default_extension(self) -> str:
        return self.reader.get_default_extension()

    @property
    def reader(self) -> TabularDataReader:
        return self._reader

    @property
    def column_groups(self) -> ColumnGroups:
        return self._column_groups

    @property
    def spectra_dataframe(self) -> pd.DataFrame:
        return self._spectra_dataframe

    @spectra_dataframe.deleter
    def spectra_dataframe(self):
        del self._spectra_dataframe

    def get_column_names(self) -> tuple[str, ...]:
        columns = self.reader.get_column_names()
        return utils.tuplize(columns)

    def __repr__(self) -> str:
        spec_sec = "\tUnset"
        try:
            spec_sec = self.spectra_dataframe
        except AttributeError:
            pass

        rep = "OnDiskPsmDataset object\n"
        rep += f"Reader: {self.reader}\n"
        rep += f"Spectrum columns: {self.spectrum_columns}\n"
        rep += f"Peptide column: {self.peptide_column}\n"
        rep += f"Feature columns: {self.feature_columns}\n"
        rep += f"Level columns: {self.confidence_level_columns}\n"
        rep += f"Spectra DF: \n{spec_sec}\n"
        return rep

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
        targets = self.read_data(columns=[self.target_column])
        targets = utils.convert_targets_column(targets, self.target_column)
        targets_series = targets[self.target_column]
        if not isinstance(targets_series, pd.Series):
            raise RuntimeError(
                f"Mokapot expected a series but got: {type(targets_series)}"
                " Please report this as a bug."
            )
        labels = _update_labels(
            scores,
            targets_series,
            eval_fdr,
            desc,
        )
        pos = labels == 1
        if not pos.sum():
            raise RuntimeError(
                "No target PSMs were below the 'eval_fdr' threshold."
            )

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

    def find_best_feature(self, eval_fdr: float) -> LabeledBestFeature:
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
            raise RuntimeError(
                f"No PSMs found below the 'eval_fdr' {eval_fdr}."
            )

        out = LabeledBestFeature(
            feature=BestFeatureProperties(
                name=best_feat,
                positives=best_positives,
                descending=best_desc,
                fdr=eval_fdr,
            ),
            new_labels=new_labels,
        )
        return out

    def update_labels(self, scores, target_column, eval_fdr=0.01, desc=True):
        df = self.read_data(columns=target_column)
        if target_column:
            df = utils.convert_targets_column(df, target_column)
        return _update_labels(
            scores=scores,
            targets=df[target_column],
            eval_fdr=eval_fdr,
            desc=desc,
        )

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
        if not all(
            c in self.spectra_dataframe.columns for c in self.spectrum_columns
        ):
            raise ValueError(
                f"Columns {self.spectrum_columns} not found in spectra"
                " dataframe"
                f" Available columns: {self.spectra_dataframe.columns}"
            )
        spectra = self.spectra_dataframe[list(self.spectrum_columns)].values
        # Q: Why is this deleted here?
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

    def read_data(
        self,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.reader.read(columns=columns)

    def read_data_chunked(
        self,
        *,
        chunk_size: int,
        columns: list[str] | None = None,
    ) -> Generator[pd.DataFrame, None, None]:
        return self.reader.get_chunked_data_iterator(
            chunk_size=chunk_size, columns=columns
        )

    @property
    def scores(self) -> np.ndarray | None:
        if not hasattr(self, "_scores"):
            return None
        return self._scores

    @scores.setter
    def scores(self, scores: np.ndarray | None):
        self._scores = scores


@typechecked
def _update_labels(
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


@typechecked
def update_labels(
    dataset: OnDiskPsmDataset | LinearPsmDataset,
    scores,
    eval_fdr=0.01,
    desc=True,
):
    if isinstance(dataset, OnDiskPsmDataset):
        targets = dataset.reader.read(columns=[dataset.target_column])[
            dataset.target_column
        ]
    elif isinstance(dataset, LinearPsmDataset):
        targets = dataset.data[dataset.target_column]
    else:
        msg = f"Unknown dataset type of type {type(dataset)}"
        raise NotImplementedError(msg)
    return _update_labels(
        scores=scores,
        targets=targets,
        eval_fdr=eval_fdr,
        desc=desc,
    )
