"""
This module contains the classes and methods needed to import, validate and
normalize a collection of PSMs in PIN (Percolator INput) format.
"""
import logging
import gzip
import random
import itertools
from typing import List, Union, Tuple

import numpy as np
import pandas as pd

from .qvalues import tdc

# Classes ---------------------------------------------------------------------
class PsmDataset():
    """
    Store a collection of PSMs.
    """
    def __init__(self, psm_data = pd.DataFrame) -> None:
        """Initialize a PsmDataset object."""

        self.data = psm_data
        # Change key cols to lowercase for consistency
        self.data.columns = self.data.columns.str.lower()

        # Verify necessary columns are present
        required_cols = {"label", "scannr", "peptide", "proteins"}
        if not required_cols <= set(self.columns):
            raise ValueError("Required columns are missing."
                             f" These are {required_cols} and are case "
                             "insensitive.")

        # Initialize scores attribute
        self._scores = None

        # Columns that define PSMs, Peptides, and Proteins
        self.psm_cols = ["scannr"]
        if "expmass" in self.columns:
            self.psm_cols += ["expmass"]
        if "fileidx" in self.columns:
            self.psm_cols += ["fileidx"]
        if "file" in self.columns:
            self.psm_cols += ["file"]

        self.peptide_cols = ["peptide"]
        self.protein_cols = ["proteins"]


    @property
    def columns(self) -> List[str]:
        """Get the columns of the PIN files in lower case."""
        return self.data.columns.tolist()

    @property
    def scores(self) -> np.ndarray:
        """Get the scores assigned to the PSMs of this dataset"""
        return self._scores

    @scores.setter
    def scores(self, score_array: np.ndarray, test_fdr: float = 0.01) -> None:
        """
        Add scores to the PsmDataset.

        Scores are calibrated such that by subtracting the score at the
        specified 'test_fdr' and dividing by the median of the decoy scores.
        Note that assigning new scores replaces any previous scores that were
        stored.
        """
        psm_df = self.data[self.psm_cols+["label"]]

        if len(score_array) != len(psm_df):
            raise ValueError("Length of 'score_array' must be equal to the"
                             " number of PSMs, %i", len(psm_df))

        psm_df["score"] = score_array
        psm_idx = psm_df.groupby(self.psm_cols).score.idxmax()
        psms = psm_df.loc[psm_idx, :]

        labels = (psms.label.values+1)/2
        qvals = tdc(psms.score.values, target=labels)

        decoy_med = qvals[~labels].median()
        test_score = psms.score.values[qvals <= test_fdr].min()

        self._scores = (score_array - test_score) / decoy_med

    @property
    def dual(self) -> bool:
        """Get the best dual setting for sklearn.svm.LinearSVC()"""
        dual = False
        if self.data.shape[0] <= self.data.shape[1]:
            dual = True
            logging.warning("The number of features is greater than the number"
                            " of PSMs.")

        return dual

    @property
    def features(self) -> pd.DataFrame:
        """Get the features of the PsmDataset"""
        cols = self.columns
        feat_end = cols.index("peptide")

        # Crux adds "ExpMass" and "CalcMass" columns after "ScanNr"
        if "calcmass" in self.columns:
            feat_start = cols.index("calcmass") + 1
        else:
            feat_start = cols.index("scannr") + 1

        return self.data.iloc[:, feat_start:feat_end]

    @property
    def label(self) -> np.ndarray:
        """Get the data PSM labels."""
        return self.data.label.values

    def find_best_feature(self, fdr: float) -> None:
        """Find the best feature to separate targets from decoys."""
        qvals = self.features.apply(tdc, target=(self.label+1)/2)
        targ_qvals = qvals[self.label == 1]
        num_passing = (targ_qvals <= fdr).sum()
        best_feat = num_passing.idxmax()
        unlabeled = np.logical_and(qvals[best_feat].values > fdr, self.label == 1)

        target = self.label.copy()
        target[unlabeled] = 0

        return best_feat, num_passing[best_feat], target

    def split(self, folds): #-> Tuple[Tuple[PsmDataset]]:
        """Split into cross-validation folds"""
        # Group by scan columns and shuffle
        scans = [df for _, df in self.data.groupby(self.psm_cols)]
        random.shuffle(scans)

        # Split the data evenly
        num = len(scans) // folds
        splits = [scans[i:i+num] for i in range(0, len(scans), num)]

        if len(splits[-1]) < num:
            splits[-2] += splits[-1]
            splits = splits[:-1]

        # This part is slow :/
        splits = [pd.concat(s, copy=False) for s in splits]

        # Assign train and test sets
        train = []
        test = []
        for idx, test_split in enumerate(splits):
            train_split = pd.concat(splits[:idx]+splits[idx+1:],
                                    ignore_index=True)

            train.append(PsmDataset(train_split))
            test.append(PsmDataset(test_split))

        return (tuple(train), tuple(test))

    def get_results(self, feature: str = None, desc: bool = True) \
        -> Tuple[pd.DataFrame]:
        """Get the PSMs and peptides with FDR estimates. Proteins to come"""
        if feature is None:
            score_feat = self.scores
            if score_feat is None:
                raise ValueError("No Mokapot scores have been assigned.")

        else:
            if feature not in self.features.columns:
                raise ValueError("Feature not found in self.features.")

            score_feat = self.features[feature].values

        psm_df = self.data[self.psm_cols+self.peptide_cols+self.protein_cols+["label"]]
        psm_df["score"] = score_feat

        if not desc:
            psm_df.score = -psm_df.score

        # PSM
        psm_idx = psm_df.groupby(self.psm_cols).score.idxmax()
        psms = psm_df.loc[psm_idx, :]

        # Peptides
        peptide_idx = psms.groupby(self.peptide_cols).score.idxmax()
        peptides = psms.loc[peptide_idx, :]

        # TODO: Protein level FDR.

        out_list = []
        cols = self.psm_cols + ["score", "mokapot q-values"] \
            + self.peptide_cols + self.protein_cols

        for dat in (psms, peptides):
            dat["mokapot q-values"] = tdc(dat.score.values,
                                          target=(dat.label.values+1)/2)

            dat.sort_values("score", ascending=(not desc), inplace=True)
            dat.reset_index(drop=True, inplace=True)
            dat=dat[cols]

            if feature is None:
                dat = dat.rename({"score": "mokapot score"})
            else:
                dat = dat.rename({"score": feature})

            out_list.append(dat)

        return tuple(out_list)



# Functions -------------------------------------------------------------------
def read_pin(pin_files: Union[str, Tuple[str]]) -> PsmDataset:
    """Read a Percolator pin file to a PsmDataset"""
    if isinstance(pin_files, str):
        pin_files = (pin_files,)

    psm_data = pd.concat([_read_pin(f) for f in pin_files])
    return PsmDataset(psm_data)

def read_mpin(mpin_files: Union[str, Tuple[str]]) -> PsmDataset:
    """Read a Mokapot input (mpin) file to a PsmDataset"""
    pass

# Utility Functions -----------------------------------------------------------
def _read_pin(pin_file):
    """Parse a Percolator INput formatted file."""
    if pin_file.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open

    with fopen(pin_file, "r") as pin:
        header = pin.readline()
        header = header.replace("\n", "").split("\t")
        rows = [l.replace("\n", "").split("\t", len(header)-1) for l in pin]

    pin_df = pd.DataFrame(columns=header, data=rows)
    return pin_df.apply(pd.to_numeric, errors="ignore")
