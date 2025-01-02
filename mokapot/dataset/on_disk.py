from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator
from zlib import crc32

import numpy as np
import pandas as pd
from typeguard import typechecked

from .. import utils
from ..column_defs import ColumnGroups, OptionalColumns
from ..tabular_data import TabularDataReader
from .base import (
    BestFeatureProperties,
    LabeledBestFeature,
    PsmDataset,
    update_labels,
)

LOGGER = logging.getLogger(__name__)


@typechecked
class OnDiskPsmDataset(PsmDataset):
    # Q: can we have a docstring here?
    def __init__(
        self,
        filename_or_reader: Path | TabularDataReader,
        spectra_dataframe: pd.DataFrame,
        *,
        target_column: str | None = None,
        peptide_column: str | None = None,
        spectrum_columns: list[str] | None = None,
        feature_columns: list[str] | None = None,
        extra_confidence_level_columns: list[str] | None = None,
        id_column: str | None = None,
        protein_column: str | None = None,
        filename_column: str | None = None,
        scan_column: str | None = None,
        calcmass_column: str | None = None,
        expmass_column: str | None = None,
        rt_column: str | None = None,
        charge_column: str | None = None,
        column_groups: ColumnGroups | None = None,
    ):
        """Initialize an OnDiskPsmDataset object."""
        super().__init__(rng=None)
        if isinstance(filename_or_reader, TabularDataReader):
            reader = filename_or_reader
        else:
            reader = TabularDataReader.from_path(filename_or_reader)

        columns = reader.get_column_names()

        cgroup_kwargs = {
            "target_column": target_column,
            "peptide_column": peptide_column,
            "spectrum_columns": spectrum_columns,
            "feature_columns": feature_columns,
            "extra_confidence_level_columns": extra_confidence_level_columns,
            "filename_column": filename_column,
            "scan_column": scan_column,
            "calcmass_column": calcmass_column,
            "expmass_column": expmass_column,
            "rt_column": rt_column,
            "charge_column": charge_column,
            "protein_column": protein_column,
            "id_column": id_column,
        }

        if column_groups is None:
            column_groups = OnDiskPsmDataset._buil_column_groups(
                columns=columns,
                **cgroup_kwargs,
            )
        else:
            column_groups = column_groups.update(**cgroup_kwargs)

        self._reader = reader
        self._column_groups = column_groups
        spectra_dataframe[column_groups.target_column] = (
            utils.make_bool_trarget(
                spectra_dataframe[column_groups.target_column]
            )
        )
        self._spectra_dataframe = spectra_dataframe
        self._check_columns()

    @staticmethod
    def _buil_column_groups(
        *,
        columns,
        target_column,
        peptide_column,
        spectrum_columns,
        feature_columns,
        extra_confidence_level_columns,
        filename_column,
        scan_column,
        calcmass_column,
        expmass_column,
        rt_column,
        charge_column,
        protein_column,
        id_column,
    ):
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

    def _check_columns(self):
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

    @property
    def target_values(self) -> np.ndarray[bool]:
        return self._spectra_dataframe.loc[:, self.target_column].values

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
        targets.loc[:, self.target_column] = utils.make_bool_trarget(
            targets.loc[:, self.target_column]
        )
        targets_series = targets[self.target_column]
        if not isinstance(targets_series, pd.Series):
            raise RuntimeError(
                f"Mokapot expected a series but got: {type(targets_series)}"
                " Please report this as a bug."
            )
        labels = update_labels(
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
        df[self.target_column] = utils.make_bool_trarget(
            df[self.target_column]
        )
        return (
            update_labels(
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

                new_labels = update_labels(
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
        return update_labels(
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
        # Q: Why is this deleted here? I am assuming memory but its not really
        #    a massive source of memory usage.
        # del self.spectra_dataframe
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
