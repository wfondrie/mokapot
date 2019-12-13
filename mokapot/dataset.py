"""
This module contains the classes and methods needed to import, validate and
normalize a collection of PSMs in PIN (Percolator INput) format.
"""
from __future__ import annotations
import logging
import gzip
import random
import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import List, Union, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from .qvalues import tdc
if TYPE_CHECKING: # Needed to avoid circular imports for type hints :P
    from .model import MokapotModel

# Classes ---------------------------------------------------------------------
class PsmDataset():
    """
    Store a collection of PSMs.
    """
    def __init__(self, psm_data: pd.DataFrame, metadata: Tuple[str] = None,
                 normalization_fdr: float = 0.01) -> None:
        """Initialize a PsmDataset object."""
        self.data = psm_data

        # Change key cols to lowercase for consistency
        #self.data.columns = self.data.columns.str.lower()
        cols = self.data.columns #.str.lower()

        # Verify necessary columns are present
        required_cols = {"label", "specid", "scannr", "peptide", "proteins"}
        if not required_cols <= set(cols.str.lower()):
            raise ValueError("Required columns are missing."
                             f" These are {required_cols} and are case "
                             "insensitive.")

        # Verify additional specified metadata is present
        if metadata is not None:
            self._metadata_cols = metadata
            if set(metadata) <= set(cols):
                raise ValueError("One or more columns specified by 'metadata' "
                                 "is not present. Are they capitalized "
                                 "correctly?")
        else:
            metadata = []
            self._metadata_cols = []

        # Initialize scores attributes
        self._raw_scores = None
        self._scores = None
        self._normalization_fdr = normalization_fdr

        # Columns that define metadata, PSMs, Peptides, and Proteins
        psm_cols = ["scannr", "expmass"]
        peptide_cols = ["peptide"]
        protein_cols = ["proteins"]
        label_col = ["label"]
        specid_col = ["specid"]
        other_cols = ["calcmass"]
        nonfeat_cols = sum([psm_cols, peptide_cols, protein_cols, label_col,
                            specid_col, other_cols, metadata], [])

        self._psm_cols = [c for c in cols if c.lower() in psm_cols]
        self._peptide_cols = [c for c in cols if c.lower() in peptide_cols]
        self._protein_cols = [c for c in cols if c.lower() in protein_cols]
        self._label_col = [c for c in cols if c.lower() in label_col]
        self._feature_cols = [c for c in cols if c.lower() not in nonfeat_cols]
        self._specid_col = [c for c in cols if c.lower() in specid_col]

    @property
    def columns(self) -> List[str]:
        """Get the columns of the PIN files in lower case."""
        return self.data.columns.tolist()

    @property
    def normalization_fdr(self) -> float:
        """Get the normalization_fdr"""
        return self._normalization_fdr

    @normalization_fdr.setter
    def normalization_fdr(self, fdr: float):
        """Change the FDR threshold used for score normalization"""
        self._normalization_fdr = fdr
        if self._raw_scores is not None:
            self.scores = self._raw_scores

    @property
    def scores(self) -> np.ndarray:
        """Get the scores assigned to the PSMs of this dataset"""
        return self._scores

    @scores.setter
    def scores(self, score_array: np.ndarray) -> None:
        """
        Add scores to the PsmDataset.

        Scores are normalized such that by subtracting the score at the
        specified 'test_fdr' and dividing by the median of the decoy scores.
        Note that assigning new scores replaces any previous scores that were
        stored.
        """
        self._raw_scores = score_array
        fdr = self._normalization_fdr
        psm_df = self.data.loc[:, self._psm_cols + self._label_col]

        if len(score_array) != len(psm_df):
            raise ValueError(f"Length of 'score_array' must be equal to the"
                             " number of PSMs, {len(psm_df)}")

        psm_df["score"] = score_array
        psm_idx = _groupby_max(psm_df, self._psm_cols, "score")
        psms = psm_df.loc[psm_idx, :]
        labels = (psms[self._label_col[0]].values + 1) / 2
        qvals = tdc(psms.score.values, target=labels)
        decoy_med = np.median(psms.score.values[~labels.astype(bool)])
        test_score = np.min(psms.score.values[qvals <= fdr])

        self._scores = (score_array - test_score) / (test_score - decoy_med)

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
        return self.data.loc[:, self._feature_cols]

    @property
    def label(self) -> np.ndarray:
        """Get the data PSM labels."""
        return self.data[self._label_col[0]].values

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

    def split(self, folds) -> Tuple[Tuple[PsmDataset]]:
        """Split into cross-validation folds"""
        scans = list(self.data.groupby(self._psm_cols, sort=False).indices.values())
        random.shuffle(scans)

        # Split the data evenly
        num = len(scans) // folds
        splits = [scans[i:i+num] for i in range(0, len(scans), num)]

        if len(splits[-1]) < num:
            splits[-2] += splits[-1]
            splits = splits[:-1]

        split_idx = [_flatten(s) for s in splits]
        splits = [self.data.loc[i, :] for i in split_idx]

        # Assign train and test sets
        train = []
        test = []
        for idx, test_split in enumerate(splits):
            train_split = pd.concat(splits[:idx]+splits[idx+1:],
                                    ignore_index=True)

            train.append(PsmDataset(train_split))
            test.append(PsmDataset(test_split))

        return (tuple(train), tuple(test), tuple(splits))

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

        cols = sum([self._psm_cols, self._peptide_cols, self._protein_cols,
                    self._label_col, self._metadata_cols, self._specid_col],
                   [])

        psm_df = self.data.loc[:, cols]
        psm_df["score"] = score_feat

        if not desc:
            psm_df.score = -psm_df.score

        # PSMs
        logging.info("Conducting target-decoy competition.")
        logging.info("Selecting one PSM per %s...", "+".join(self._psm_cols))
        psm_idx = _groupby_max(psm_df, self._psm_cols, "score")
        psms = psm_df.loc[psm_idx, :]

        # Peptides
        peptide_idx = _groupby_max(psms, self._peptide_cols, "score")
        peptides = psms.loc[peptide_idx, :]

        # TODO: Protein level FDR.

        out_list = []
        keep = sum([self._metadata_cols, self._specid_col,
                    ["score", "q-value"], self._peptide_cols,
                    self._protein_cols], [])

        labs = ("PSMs", "peptides")
        for dat, lab in zip((psms, peptides), labs):
            targ = (dat[self._label_col[0]].values + 1) / 2
            dat["q-value"] = tdc(dat.score.values, targ)
            dat = dat.loc[targ.astype(bool), :] # Keep only targets
            dat = dat.sort_values("score", ascending=(not desc))
            dat = dat.reset_index(drop=True)
            dat = dat[keep]

            num_found = (dat["q-value"] <= self._normalization_fdr).sum()
            logging.info("-> Found %i %s at %2.f%% FDR.", num_found, lab,
                         self._normalization_fdr*100)

            dat = dat.rename(columns={self._specid_col[0]: "PSMId"})
            if feature is None:
                dat = dat.rename(columns={"score": "mokapot score"})
            else:
                dat = dat.rename(columns={"score": feature})

            out_list.append(dat)

        return tuple(out_list)

    def percolate(self, estimator: MokapotModel, train_fdr: float = 0.01,
                  test_fdr: float = 0.01, max_iter: int = 10,
                  folds: int = 3, max_workers: int = None,
                  **kwargs) -> None:
        """Run the tradiational Percolator algorithm with cross-validation"""
        logging.info("Splitting PSMs into %i folds...", folds)
        train_sets, test_sets, test_idx = self.split(folds)

        # Need args for map:
        map_args = [estimator._fit, train_sets, [train_fdr]*folds,
                    [max_iter]*folds, [False]*folds, [kwargs]*folds]

        # Train models in parallel:
        estimator.estimator = []
        logging.info("Training SVM models by %i-fold cross-validation...\n", folds)
        with ProcessPoolExecutor(max_workers=max_workers) as prc:
            for split, results in enumerate(prc.map(*map_args)):
                _fold_msg(results[3], results[1], results[2], split+1)
                estimator.estimator.append(results[0])

        estimator.estimator = tuple(estimator.estimator)
        del train_sets # clear some memory

        logging.info("Scoring PSMs...")
        scores = estimator.predict(test_sets)

        # Add scores to test sets
        for test_set, score in zip(test_sets, scores):
            test_set.normalization_fdr = test_fdr
            test_set.scores = score

        scores = np.concatenate([s.scores for s in test_sets])
        test_idx = np.concatenate(test_idx)
        self.scores = scores[test_idx]

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


def merge(psms: Tuple[PsmDataset]):
    """Merge multiple PsmDataset objects into one."""
    psm_data = pd.concat([p.data for p in psms], ignore_index=True)
    new_psms = PsmDataset(psm_data)
    scores = [p.scores for p in psms]
    scores_exist = [s is not None for s in scores]

    if all(scores_exist):
        new_psms.scores = np.concatenate(scores)
    elif any(scores_exist):
        logging.warning("One or more PsmDataset did not have scores "
                        "assigned. Scores were reset with merge.")

    return new_psms

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


def _flatten(split):
    """Get the indices from split"""
    return list(itertools.chain.from_iterable(split))


def _groupby_max(df, by, max_col):
    """Quickly get the indices for the maximum value of col"""
    return df.sort_values(by+[max_col]).drop_duplicates(by, keep="last").index


def _best_feat_msg(best_feat, num_pass):
    """Log the best feature and the number of positive PSMs."""
    logging.info("Selected feature '%s' as initial direction.", best_feat)
    logging.info("Could separate %i training set positives in that direction.",
                 num_pass)


def _fold_msg(best_feat, best_feat_pass, num_pass, fold=None):
    """Logging messages for each fold"""
    if fold is not None:
        logging.info("Fold %i", fold)
        _best_feat_msg(best_feat, best_feat_pass)

    logging.info("Positive PSMs by iteration:")
    num_passed = [f"{n}" for i, n in enumerate(num_pass)]
    num_passed = "->".join(num_passed)
    logging.info("%s\n", num_passed)

