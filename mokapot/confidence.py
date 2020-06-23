"""
One of the primary purposes of mokapot is to assign confidence estimates to PSMs.
This task is accomplished by ranking PSMs according to a score or metric and
using an appropriate confidence estimation procedure for the type of data
(currently, linear and crosslinked PSMs are supported). In either case,
mokapot can provide confidence estimates based any score, regardless of
whether it was the result of a learned :py:func:`~mokapot.model.Model`
instance or provided independently.

The following classes store the confidence estimates for a dataset based on the
provided score. In either case, they provide utilities to access, save, and
plot these estimates for the various relevant levels (i.e. PSMs, peptides, and
proteins). The :py:func:`LinearPsmConfidence` class is appropriate for most
proteomics datasets, whereas the :py:func:`CrossLinkedPsmConfidence` is
specifically designed for crosslinked peptides.
"""
from __future__ import annotations
import logging
import random
from typing import Tuple, Union, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from triqler import qvality

from . import qvalues
if TYPE_CHECKING:
    from .dataset import PsmDataset, LinearPsmDataset, CrossLinkedPsmDataset

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
class PsmConfidence():
    """
    Estimate and store the statistical confidence for a collection of
    PSMs.

    :meta private:
    """
    def __init__(self, psms: PsmDataset, scores: np.ndarray) -> None:
        """
        Initialize a PsmConfidence object.
        """
        #self.data = psms.metadata.sample(frac=1)
        self.data = psms.metadata
        self.data[len(psms.columns)] = scores
        self.score_column = self.data.columns[-1]

        # This attribute holds the results as DataFrames:
        self.qvalues: Dict[str, pd.DataFrame] = {}

    @property
    def levels(self) -> Tuple[str, ...]:
        """The available confidence levels (i.e. PSMs, peptides, proteins)"""
        return tuple(self.qvalues.keys())

    def to_txt(self, fileroot: str, sep: str = "\t"):
        """Save the results to files"""
        for level, qvals in self.qvalues.items():
            qvals.to_csv(f"{fileroot}.{level}.txt", sep="\t", index=False)

    def _perform_tdc(self, psm_columns: Tuple[str, ...]) -> None:
        """Conduct TDC, stuff"""
        psm_idx = _groupby_max(self.data, psm_columns, self.score_column)
        self.data = self.data.loc[psm_idx]

    def plot(self, level: str, threshold: float = 0.1,
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
        level_labs = {"psms": "PSMs",
                      "peptides": "Peptides",
                      "proteins": "Proteins"}

        if ax is None:
            ax = plt.gca()
        elif not isinstance(ax, plt.Axes):
            raise ValueError("'ax' must be a matplotlib Axes instance.")

        # Calculate cumulative targets at each q-value
        qvals = self.qvalues[level].loc[:, ["mokapot q-value"]]
        qvals = qvals.sort_values(by="mokapot q-value", ascending=True)
        qvals["target"] = 1
        qvals["num"] = qvals["target"].cumsum()
        qvals = qvals.groupby(["mokapot q-value"]).max().reset_index()
        qvals = qvals[["mokapot q-value", "num"]]

        zero = pd.DataFrame({"mokapot q-value": qvals["mokapot q-value"][0],
                             "num": 0}, index=[-1])
        qvals = pd.concat([zero, qvals], sort=True).reset_index(drop=True)

        xmargin = threshold * 0.05
        ymax = qvals.num[qvals["mokapot q-value"] <= (threshold + xmargin)].max()
        ymargin = ymax * 0.05

        # Set margins
        curr_ylims = ax.get_ylim()
        if curr_ylims[1] < ymax + ymargin:
            ax.set_ylim(0 - ymargin, ymax + ymargin)

        ax.set_xlim(0 - xmargin, threshold + xmargin)
        ax.set_xlabel("q-value")
        ax.set_ylabel(f"Accepted {level_labs[level]}")

        return ax.step(qvals["mokapot q-value"].values,
                       qvals.num.values, where="post", **kwargs)


class LinearPsmConfidence(PsmConfidence):
    """Assign confidence to a set of linear PSMs"""
    def __init__(self, psms: LinearPsmDataset, scores: np.ndarray,
                 desc: bool = True) -> None:
        """Initialize a a LinearPsmConfidence object"""
        super().__init__(psms, scores)
        self.data[len(self.data.columns)] = psms.targets
        self.target_column = self.data.columns[-1]
        self.psm_columns = psms.spectrum_columns + psms.experiment_columns
        self.peptide_columns = psms.peptide_columns + psms.experiment_columns

        self._perform_tdc(self.psm_columns)
        self._assign_confidence(desc=desc)


    def _assign_confidence(self, desc: bool) -> None:
        """
        Assign confidence to PSMs
        """
        peptide_idx = _groupby_max(self.data, self.peptide_columns,
                                   self.score_column)

        peptides = self.data.loc[peptide_idx]

        for level, data in zip(("psms", "peptides"), (self.data, peptides)):
            scores = data.loc[:, self.score_column].values
            targets = data.loc[:, self.target_column].astype(bool).values
            data["mokapot q-value"] = qvalues.tdc(scores, targets, desc)

            data = data.loc[targets, :] \
                       .sort_values(self.score_column, ascending=(not desc)) \
                       .reset_index(drop=True) \
                       .drop(self.target_column, axis=1) \
                       .rename(columns={self.score_column: "mokapot score"})

            target_scores = scores[targets]
            decoy_scores = scores[~targets]
            _, data["mokapot PEP"] = qvality.getQvaluesFromScores(scores[targets],
                                                                  scores[~targets])

            self.qvalues[level] = data


class CrossLinkedPsmConfidence(PsmConfidence):
    """Assign confidence to a set of CrossLinked PSMs"""
    def __init__(self, psms: LinearPsmDataset, scores: np.ndarray,
                 desc: bool = True) -> None:
        """Initialize a a LinearPsmConfidence object"""
        super().__init__(psms, scores)
        self.data[len(self.data.columns)] = psms.targets
        self.target_column = self.data.columns[-1]
        self.psm_columns = psms.spectrum_columns + psms.experiment_columns
        self.peptide_columns = psms.peptide_columns + psms.experiment_columns

        self._perform_tdc(self.psm_columns)
        self._assign_confidence(desc=desc)

    def _assign_confidence(self, desc: bool) -> None:
        """
        Assign confidence to PSMs
        """
        peptide_idx = _groupby_max(self.data, self.peptide_columns,
                                   self.score_column)

        peptides = self.data.loc[peptide_idx]

        for level, data in zip(("psms", "peptides"), (self.data, peptides)):
            scores = data.loc[:, self.score_column].values
            targets = data.loc[:, self.target_column].astype(bool).values
            data["mokapot q-value"] = qvalues.crosslink_tdc(scores, targets,
                                                            desc)

            data = data.loc[targets, :] \
                       .sort_values(self.score_column, ascending=(not desc)) \
                       .reset_index(drop=True) \
                       .drop(self.target_column, axis=1) \
                       .rename(columns={self.score_column: "mokapot score"})

            target_scores = scores[targets]
            decoy_scores = scores[~targets]
            _, data["mokapot PEP"] = qvality.getQvaluesFromScores(scores[targets == 2],
                                                                  scores[~targets])

            self.qvalues[level] = data



# Functions -------------------------------------------------------------------
def _groupby_max(df: pd.DataFrame, by_cols: Tuple[str, ...], max_col: str):
    """Quickly get the indices for the maximum value of col"""
    idx = df.sample(frac=1) \
            .sort_values(list(by_cols)+[max_col], axis=0) \
            .drop_duplicates(list(by_cols), keep="last") \
            .index

    return idx
