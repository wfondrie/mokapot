"""
A class to estimate and store confidence information about a collection
of PSMs
"""
from __future__ import annotations
import logging
import random
from typing import Tuple, Union, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mokapot.qvalues as qvalues
if TYPE_CHECKING:
    from mokapot.dataset import PsmDataset, LinearPsmDataset

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
class PsmConfidence():
    """
    Estimate and store the statistical confidence for a collection of
    PSMs.
    """
    def __init__(self, psms: PsmDataset, scores: np.ndarray) -> None:
        """
        Initialize a PsmConfidence object.
        """
        self.data = psms.metadata
        self.data[len(psms.columns)] = scores
        self.score_column = psms.columns[-1]

        # This attribute holds the results as DataFrames:
        self.qvalues: Dict[str, pd.DataFrame] = {}

    @property
    def levels(self) -> Tuple[str, ...]:
        """The available confidence levels (i.e. PSMs, peptides, proteins)"""
        return tuple(self.qvalues.keys())

    def to_txt(self, fileroot: str, sep: str = "\t"):
        """Save the results to files"""
        for level, qvals in self.qvalues.items():
            pd.to_csv(qvals, f"{fileroot}.{level}.txt", sep="\t")

    def _perform_tdc(self, psm_columns: Tuple[str, ...]) -> None:
        """Conduct TDC, stuff"""
        psm_idx = _groupby_max(self.data, psm_columns, self.score_column)
        self.data = self.data.loc[psm_idx]

    def plot(self, level: str, theshold: float = 0.1,
             ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """
        Plot the accepted number of PSMs, peptides, or proteins over
        a range of q-values.

        Parameters
        ----------
        level : str, optional
            The level of q-values to report. Can be one of `"psms"`,
            `"peptides"`, or `"proteins"`.

        threshold : float, optional
            Indicates the maximum q-value to plot.

        ax : matplotlib.pyplot.Axes, optional
            The matplotlib Axes on which to plot. If `None` the current
            Axes instance is used.

        **kwargs : dict, optional
            Arguments passed to matplotlib.pyplot.plot()

        Returns
        -------
        matplotlib.pyplot.Axes
            A plot of the cumulative number of accepted target PSMs,
            peptides, or proteins.
        """
        pass


class LinearPsmConfidence(PsmConfidence):
    """Assign confidence to a set of linear PSMs"""
    def __init__(self, psms: LinearPsmDataset, scores: np.ndarray,
                 protein_database: None = None,
                 desc: bool = True) -> None:
        """Initialize a a LinearPsmConfidence object"""
        super().__init__(psms, scores)
        self.data[len(self.data.columns)] = psms.targets
        self.target_column = self.data.columns[-1]
        self.psm_columns = psms.spectrum_columns + psms.experiment_columns
        self.peptide_columns = psms.peptide_columns + psms.experiment_columns

        self._perform_tdc(self.psm_columns)

        if protein_database is None:
            self._assign_confidence(protein=False, desc=desc)
        else:
            # TODO picked-protein grouping.
            self._assign_confidence(protein=True, desc=desc)

    def _assign_confidence(self, protein: bool, desc: bool) -> None:
        """
        Assign confidence to PSMs
        """
        peptide_idx = _groupby_max(self.data, self.peptide_columns,
                                   self.score_column)

        peptides = self.data.loc[peptide_idx]

        for level, data in zip(("psms", "peptides"), (self.data, peptides)):
            scores = self.data[self.score_column]
            targets = self.data[self.target_column]
            data["mokapot q-value"] = qvalues.tdc(scores, targets, desc)

            # TODO: Add PEP estimation here
            data["mokapot PEP"] = 0

            data = data.loc[targets, :] \
                       .sort_values(self.score_column, ascending=(not desc)) \
                       .reset_index(drop=True) \
                       .drop(self.target_column) \
                       .rename(columns={"mokapot score": self.score_column})

            self.qvalues[level] = data

        if protein:
            # TODO picked-protein FDR.
            pass


# Functions -------------------------------------------------------------------
def _groupby_max(df: pd.DataFrame, by: Tuple[str, ...], max_col: str):
    """Quickly get the indices for the maximum value of col"""
    return df.sort_values(by+(max_col,)).drop_duplicates(by, keep="last").index
