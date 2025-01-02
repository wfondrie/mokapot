from __future__ import annotations

import logging
from typing import Generator

import numpy as np
import pandas as pd

from .. import utils
from ..column_defs import ColumnGroups, OptionalColumns
from ..tabular_data import DataFrameReader, TabularDataReader
from .base import (
    BestFeatureProperties,
    LabeledBestFeature,
    PsmDataset,
    calibrate_scores,
    update_labels,
)

LOGGER = logging.getLogger(__name__)

# Q: should I add typechecking here?


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
        target_column: str | None = None,
        peptide_column: str | None = None,
        spectrum_columns: list[str] | tuple[str, ...] | None = None,
        feature_columns: list[str] | tuple[str, ...] | None = None,
        extra_confidence_level_columns: list[str]
        | tuple[str, ...]
        | None = None,
        copy_data: bool = True,
        rng: int | np.random.Generator | None = None,
        enforce_checks: bool = True,
        *,
        id_column: str | None = None,
        filename_column: str | None = None,
        scan_column: str | None = None,
        calcmass_column: str | None = None,
        expmass_column: str | None = None,
        rt_column: str | None = None,
        charge_column: str | None = None,
        protein_column: str | None = None,
        column_groups: ColumnGroups | None = None,
    ):
        """Initialize a PsmDataset object from a ColumnGroups object."""
        super().__init__(rng=rng)
        all_columns = {c: psms[c].dtype for c in psms.columns}
        cgroup_kwargs = {
            "target_column": target_column,
            "peptide_column": peptide_column,
            "spectrum_columns": spectrum_columns,
            "feature_columns": feature_columns,
            "extra_confidence_level_columns": extra_confidence_level_columns,
            "id_column": id_column,
            "filename_column": filename_column,
            "scan_column": scan_column,
            "calcmass_column": calcmass_column,
            "expmass_column": expmass_column,
            "rt_column": rt_column,
            "charge_column": charge_column,
            "protein_column": protein_column,
        }
        if column_groups is None:
            column_groups = LinearPsmDataset._build_column_groups(
                all_columns,
                **cgroup_kwargs,
            )
        else:
            column_groups = column_groups.update(**cgroup_kwargs)

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
    def _build_column_groups(
        all_columns: dict[str, np.dtype],
        target_column: str,
        peptide_column: str,
        spectrum_columns: list[str] | tuple[str, ...],
        feature_columns: list[str] | tuple[str, ...] | None = None,
        extra_confidence_level_columns: list[str]
        | tuple[str, ...]
        | None = None,
        id_column: str | None = None,
        filename_column: str | None = None,
        scan_column: str | None = None,
        calcmass_column: str | None = None,
        expmass_column: str | None = None,
        rt_column: str | None = None,
        charge_column: str | None = None,
        protein_column: str | None = None,
    ) -> ColumnGroups:
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
            c for c in set(used_columns) if c not in all_columns
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
            for c in all_columns:
                # Add only if its not a string (int, float, bool)
                if c in used_columns:
                    continue
                elif not isinstance(all_columns[c], str):
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
            columns=tuple(all_columns.keys()),
            target_column=target_column,
            peptide_column=peptide_column,
            spectrum_columns=spectrum_columns,
            feature_columns=feature_columns,
            extra_confidence_level_columns=extra_confidence_level_columns,
            optional_columns=OptionalColumns(
                id=id_column,
                filename=filename_column,
                scan=scan_column,
                calcmass=calcmass_column,
                expmass=expmass_column,
                rt=rt_column,
                charge=charge_column,
                protein=protein_column,
            ),
        )
        return column_groups

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
        out = utils.make_bool_trarget(self._data[self.target_column])
        self._data[self.target_column] = out

    @property
    def target_values(self) -> np.ndarray[bool]:
        return self._data[self.target_column]

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
        return self.data.loc[:, self.peptide_column]

    @property
    def features(self):
        """A :py:class:`pandas.DataFrame` of the features."""
        return self.data.loc[:, list(self.feature_columns)]

    @property
    def spectra_dataframe(self):
        """
        A :py:class:`pandas.DataFrame` of the columns that uniquely
        identify a mass spectrum.
        """
        return self.data[list(self.spectrum_columns)]

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

        Returns
        -------
        list of BestFeatureProperties
            The number of positive examples for each feature and the
            associated properties.
        """

        outs = []
        for col in self.feature_columns:
            labs = update_labels(
                self.data.loc[:, col],
                self.data.loc[:, self.target_column].values,
                eval_fdr=eval_fdr,
                desc=desc,
            )
            tmp = BestFeatureProperties(
                name=col,
                positives=(labs == 1).sum(),
                fdr=eval_fdr,
                descending=desc,
            )
            outs.append(tmp)
        return outs

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
        feat_stats = []
        for desc in (True, False):
            num_passing = self._targets_count_by_feature(desc, eval_fdr)
            feat_stats.extend(num_passing)

        feat_stats.sort(key=lambda x: x.positives, reverse=True)
        best = feat_stats[0]

        if best.positives == 0:
            raise RuntimeError(
                f"No PSMs found below the 'eval_fdr' {eval_fdr}"
                " for any feature."
            )

        new_labels = update_labels(
            self.data.loc[:, best.name],
            self.data.loc[:, self.target_column].values,
            eval_fdr=eval_fdr,
            desc=best.descending,
        )

        out = LabeledBestFeature(
            feature=best,
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
            data = data.loc[:, columns]

        return data

    def read_data_chunked(
        self,
        *,
        chunk_size: int,
        columns: list[str] | None = None,
    ) -> Generator[pd.DataFrame, None, None]:
        data = self.data
        if columns is not None:
            data = data.loc[:, columns]

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
