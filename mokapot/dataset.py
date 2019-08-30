"""
This module contains the classes and methods needed to import, validate and
normalize a collection of PSMs in PIN (Percolator INput) format.
"""
import logging
import gzip
from typing import List, Union, Tuple

import numpy as np
import pandas as pd

from mokapot.qvalues import tdc

# Classes ---------------------------------------------------------------------
class PsmDataset():
    """
    Store a collection of PSMs.
    """
    def __init__(self, pin_files: Union[str, Tuple[str]]) -> None:
        """Initialize a PsmDataset object."""
        if isinstance(pin_files, str):
            pin_files = (pin_files,)

        self.data = pd.concat([_read_pin(f) for f in pin_files])
        cols = self.columns

        # Verify necessary columns are present.
        required_cols = {"label", "scannr", "peptide", "proteins"}
        if not required_cols <= set(cols):
            raise ValueError("Required columns are missing from the pin file."
                             f" These are {required_cols} and are case "
                             "insensitive.")

        # Change key cols to lowercase for consistency
        df_cols = self.data.columns.tolist()
        for col_name in required_cols:
            df_cols[cols.index(col_name)] = col_name

        self.data.columns = df_cols

    @property
    def columns(self) -> List[str]:
        """Get the columns of the PIN files in lower case."""
        return [c.lower() for c in self.data.columns.tolist()]

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

        print(num_passing)
        print(qvals[best_feat].values > fdr)
        print(self.label == 1)
        unlabeled = np.logical_and(qvals[best_feat].values > fdr, self.label == 1)

        target = self.label
        target[unlabeled] = 0

        return best_feat, num_passing[best_feat], target


# Functions -------------------------------------------------------------------
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
