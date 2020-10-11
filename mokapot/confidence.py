"""
One of the primary purposes of mokapot is to assign confidence estimates to PSMs.
This task is accomplished by ranking PSMs according to a score or metric and
using an appropriate confidence estimation procedure for the type of data
(currently, linear and cross-linked PSMs are supported). In either case,
mokapot can provide confidence estimates based any score, regardless of
whether it was the result of a learned :py:func:`mokapot.model.Model`
instance or provided independently.

The following classes store the confidence estimates for a dataset based on the
provided score. In either case, they provide utilities to access, save, and
plot these estimates for the various relevant levels (i.e. PSMs, peptides, and
proteins). The :py:func:`LinearConfidence` class is appropriate for most
proteomics datasets, whereas the :py:func:`CrossLinkedConfidence` is
specifically designed for crosslinked peptides.
"""
import os
import logging

import pandas as pd
import matplotlib.pyplot as plt
from triqler import qvality

from . import qvalues

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
class Confidence:
    """
    Estimate and store the statistical confidence for a collection of
    PSMs.

    :meta private:
    """

    _level_labs = {
        "psms": "PSMs",
        "peptides": "Peptides",
        "proteins": "Proteins",
        "csms": "Cross-Linked PSMs",
        "peptide_pairs": "Peptide Pairs",
    }

    def __init__(self, psms, scores, desc):
        """
        Initialize a PsmConfidence object.
        """
        self._data = psms.metadata
        self._score_column = _new_column("score", self._data)
        self._has_proteins = psms.has_proteins
        if psms.has_proteins:
            self._proteins = psms._proteins
        else:
            self._proteins = None

        # Flip sign of scores if not descending
        self._data[self._score_column] = scores * (desc * 2 - 1)

        # This attribute holds the results as DataFrames:
        self._confidence_estimates = {}

    def __getattr__(self, attr):
        try:
            return self._confidence_estimates[attr]
        except KeyError:
            raise AttributeError

    @property
    def levels(self):
        """
        The available levels for confidence estimates.
        """
        return list(self._confidence_estimates.keys())

    def to_txt(self, dest_dir=None, file_root=None, sep="\t"):
        """
        Save confidence estimates to delimited text files.

        Parameters
        ----------
        dest_dir : str or None, optional
            The directory in which to save the files. `None` will use the
            current working directory.
        file_root : str or None, optional
            An optional prefix for the confidence estimate files. The
            suffix will always be `mokapot.psms.txt` and
            `mokapot.peptides.txt`.
        sep : str
            The delimiter to use.

        Returns
        -------
        list of str
            The paths to the saved files.
        """
        file_base = "mokapot"
        if file_root is not None:
            file_base = file_root + "." + file_base
        if dest_dir is not None:
            file_base = os.path.join(dest_dir, file_base)

        out_files = []
        for level, qvals in self._confidence_estimates.items():
            out_file = file_base + f".{level}.txt"
            qvals.to_csv(out_file, sep=sep, index=False)
            out_files.append(out_file)

        return out_files

    def _perform_tdc(self, psm_columns):
        """
        Perform target-decoy competition.

        Parameters
        ----------
        psm_columns : str or list of str
            The columns that define a PSM.
        """
        psm_idx = _groupby_max(self._data, psm_columns, self._score_column)
        self._data = self._data.loc[psm_idx, :]

    def plot_qvalues(self, level="psms", threshold=0.1, ax=None, **kwargs):
        """
        Plot the cumulative number of discoveries over range of q-values.

        The available levels can be found using
        :py:meth:`~mokapot.confidence.Confidence.levels` attribute.

        Parameters
        ----------
        level : str, optional
            The level of q-values to report.
        threshold : float, optional
            Indicates the maximum q-value to plot.
        ax : matplotlib.pyplot.Axes, optional
            The matplotlib Axes on which to plot. If `None` the current
            Axes instance is used.
        **kwargs : dict, optional
            Arguments passed to :py:func:`matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.pyplot.Axes
            An :py:class:`matplotlib.axes.Axes` with the cumulative
            number of accepted target PSMs or peptides.
        """
        ax = plot_qvalues(
            self._confidence_estimates[level]["mokapot q-value"],
            threshold=threshold,
            ax=ax,
            **kwargs,
        )
        ax.set_xlabel("q-value")
        ax.set_ylabel(f"Accepted {self._level_labs[level]}")

        return ax


class LinearConfidence(Confidence):
    """
    Assign confidence estimates to a set of PSMs

    Estimate q-values and posterior error probabilities (PEPs) for PSMs
    and peptides when ranked by the provided scores.

    Parameters
    ----------
    psms : LinearPsmDataset object
        A collection of PSMs.
    scores : np.ndarray
        A vector containing the score of each PSM.
    desc : bool
        Are higher scores better?
    eval_fdr : float
        The FDR threshold at which to report performance. This parameter
        has no affect on the analysis itself, only logging messages.

    Attributes
    ----------
    levels : list of str
    psms : pandas.DataFrame
        Confidence estimates for PSMs in the dataset.
    peptides : pandas.DataFrame
        Confidence estimates for peptides in the dataset.
    """

    def __init__(self, psms, scores, desc=True, eval_fdr=0.01):
        """Initialize a a LinearPsmConfidence object"""
        LOGGER.info("=== Assigning Confidence ===")
        super().__init__(psms, scores, desc)
        self._target_column = _new_column("target", self._data)
        self._data[self._target_column] = psms.targets
        self._psm_columns = psms._spectrum_columns
        self._peptide_column = psms._peptide_column
        self._eval_fdr = eval_fdr

        LOGGER.info("Performing target-decoy competition...")
        LOGGER.info(
            "Keeping the best match per %s columns...",
            "+".join(self._psm_columns),
        )

        self._perform_tdc(self._psm_columns)
        LOGGER.info("\t- Found %i PSMs from unique spectra.", len(self._data))

        self._assign_confidence(desc=desc)

        self.accepted = {}
        for level in self.levels:
            self.accepted[level] = self._num_accepted(level)

    def __repr__(self):
        """How to print the class"""
        base = (
            "A mokapot.confidence.LinearConfidence object:\n"
            f"\t- PSMs at q<={self._eval_fdr:g}: {self.accepted['psms']}\n"
            f"\t- Peptides at q<={self._eval_fdr:g}: "
            f"{self.accepted['peptides']}\n"
        )

        if self._has_proteins:
            base += (
                f"\t- Protein groups at q<={self._eval_fdr:g}: "
                f"{self.accepted['proteins']}\n"
            )

        return base

    def _num_accepted(self, level):
        """Calculate the number of accepted discoveries"""
        disc = self._confidence_estimates[level]
        return (disc["mokapot q-value"] <= self._eval_fdr).sum()

    def _assign_confidence(self, desc=True):
        """
        Assign confidence to PSMs and peptides.

        Parameters
        ----------
        desc : bool
            Are higher scores better?
        """
        levels = ["PSMs", "peptides"]
        peptide_idx = _groupby_max(
            self._data, self._peptide_column, self._score_column
        )

        peptides = self._data.loc[peptide_idx]
        LOGGER.info("\t- Found %i unique peptides.", len(peptides))

        level_data = [self._data, peptides]

        if self._has_proteins:
            proteins = self._picked_protein(peptides)
            levels += ["proteins"]
            level_data += [proteins]
            LOGGER.info("\t- Found %i unique protein groups.", len(proteins))

        for level, data in zip(levels, level_data):
            scores = data.loc[:, self._score_column].values
            targets = data.loc[:, self._target_column].astype(bool).values

            # Estimate q-values and assign to dataframe
            LOGGER.info("Assiging q-values to %s...", level)
            data["mokapot q-value"] = qvalues.tdc(scores, targets, desc)

            # Make output tables pretty
            data = (
                data.loc[targets, :]
                .sort_values(self._score_column, ascending=(not desc))
                .reset_index(drop=True)
                .drop(self._target_column, axis=1)
                .rename(columns={self._score_column: "mokapot score"})
            )

            # Set scores to be the correct sign again:
            data["mokapot score"] = data["mokapot score"] * (desc * 2 - 1)

            # Logging update on q-values
            LOGGER.info(
                "\t- Found %i %s with q<=%g",
                (data["mokapot q-value"] <= self._eval_fdr).sum(),
                level,
                self._eval_fdr,
            )

            # Calculate PEPs
            LOGGER.info("Assiging PEPs to %s...", level)
            _, pep = qvality.getQvaluesFromScores(
                scores[targets], scores[~targets]
            )
            data["mokapot PEP"] = pep
            self._confidence_estimates[level.lower()] = data

    def _picked_protein(self, peptides):
        """
        Calculate protein-level confidence estimates.

        Returns
        -------
        pandas.DataFrame
            The protein confidence estimates.
        """
        keep = [
            self._target_column,
            self._peptide_column[0],
            self._score_column,
        ]

        prots = peptides.loc[:, keep].rename(
            columns={self._peptide_column[0]: "best peptide"}
        )

        prots["stripped sequence"] = (
            prots["best peptide"]
            .str.replace(r"[\[\(].*?[\]\)]", "")
            .str.replace(r"^.*?\.", "")
            .str.replace(r"\..*?$", "")
        )

        prots["mokapot protein group"] = prots["stripped sequence"].map(
            self._proteins.peptide_map.get
        )

        unmatched = pd.isna(prots["mokapot protein group"])
        unmatched_prots = prots.loc[unmatched, :]
        shared = unmatched_prots["stripped sequence"].isin(
            self._proteins.shared_peptides
        )

        shared_unmatched = (~shared).sum()
        num_shared = len(shared) - shared_unmatched
        LOGGER.debug(
            "%i out of %i peptides were discarded as shared peptides.",
            num_shared,
            len(prots),
        )

        if shared_unmatched:
            LOGGER.warning(
                "%i out of %i peptides could not be mapped."
                "Check your digest settings.",
                shared_unmatched,
                len(prots),
            )

        prots = prots.loc[~unmatched, :]

        prots["decoy"] = (
            prots["mokapot protein group"]
            .str.split(",", expand=True)[0]
            .map(lambda x: self._proteins.protein_map.get(x, x))
        )

        prot_idx = _groupby_max(prots, ["decoy"], self._score_column)
        final_cols = [
            "mokapot protein group",
            "best peptide",
            "stripped sequence",
            self._score_column,
            self._target_column,
        ]

        return prots.loc[prot_idx, final_cols]


class CrossLinkedConfidence(Confidence):
    """
    Assign confidence estimates to a set of cross-linked PSMs

    Estimate q-values and posterior error probabilities (PEPs) for
    cross-linked PSMs (CSMs) and the peptide pairs when ranked by the
    provided scores.

    Parameters
    ----------
    psms : CrossLinkedPsmDataset object
        A collection of cross-linked PSMs.
    scores : np.ndarray
        A vector containing the score of each PSM.
    desc : bool
        Are higher scores better?

    Attributes
    ----------
    csms : pandas.DataFrame
        Confidence estimates for cross-linked PSMs in the dataset.
    peptide_pairs : pandas.DataFrame
        Confidence estimates for peptide pairs in the dataset.
    """

    def __init__(self, psms, scores, desc=True):
        """Initialize a CrossLinkedConfidence object"""
        super().__init__(psms, scores, desc)
        self._data[len(self._data.columns)] = psms.targets
        self._target_column = self._data.columns[-1]
        self._psm_columns = psms._spectrum_columns
        self._peptide_column = psms._peptide_column

        self._perform_tdc(self._psm_columns)
        self._assign_confidence(desc=desc)

    def _assign_confidence(self, desc=True):
        """
        Assign confidence to PSMs and peptides.

        Parameters
        ----------
        desc : bool
            Are higher scores better?
        """
        peptide_idx = _groupby_max(
            self._data, self._peptide_columns, self._score_column
        )

        peptides = self._data.loc[peptide_idx]
        levels = ("csms", "peptide_pairs")

        for level, data in zip(levels, (self._data, peptides)):
            scores = data.loc[:, self._score_column].values
            targets = data.loc[:, self._target_column].astype(bool).values
            data["mokapot q-value"] = qvalues.crosslink_tdc(
                scores, targets, desc
            )

            data = (
                data.loc[targets, :]
                .sort_values(self._score_column, ascending=(not desc))
                .reset_index(drop=True)
                .drop(self._target_column, axis=1)
                .rename(columns={self._score_column: "mokapot score"})
            )

            _, pep = qvality.getQvaluesFromScores(
                scores[targets == 2], scores[~targets]
            )
            data["mokapot PEP"] = pep
            self._confidence_estimates[level] = data


# Functions -------------------------------------------------------------------
def plot_qvalues(qvalues, threshold=0.1, ax=None, **kwargs):
    """
    Plot the cumulative number of discoveries over range of q-values.

    Parameters
    ----------
    qvalues : numpy.ndarray
        The q-values to plot.
    threshold : float, optional
        Indicates the maximum q-value to plot.
    ax : matplotlib.pyplot.Axes, optional
        The matplotlib Axes on which to plot. If `None` the current
        Axes instance is used.
    **kwargs : dict, optional
        Arguments passed to :py:func:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.pyplot.Axes
        An :py:class:`matplotlib.axes.Axes` with the cumulative
        number of accepted target PSMs or peptides.
    """
    if ax is None:
        ax = plt.gca()

    # Calculate cumulative targets at each q-value
    qvals = pd.Series(qvalues, name="qvalue")
    qvals = qvals.sort_values(ascending=True).to_frame()
    qvals["target"] = 1
    qvals["num"] = qvals["target"].cumsum()
    qvals = qvals.groupby(["qvalue"]).max().reset_index()
    qvals = qvals[["qvalue", "num"]]

    zero = pd.DataFrame({"qvalue": qvals["qvalue"][0], "num": 0}, index=[-1])
    qvals = pd.concat([zero, qvals], sort=True).reset_index(drop=True)

    xmargin = threshold * 0.05
    ymax = qvals.num[qvals["qvalue"] <= (threshold + xmargin)].max()
    ymargin = ymax * 0.05

    # Set margins
    curr_ylims = ax.get_ylim()
    if curr_ylims[1] < ymax + ymargin:
        ax.set_ylim(0 - ymargin, ymax + ymargin)

    ax.set_xlim(0 - xmargin, threshold + xmargin)
    ax.set_xlabel("q-value")
    ax.set_ylabel(f"Discoveries")

    ax.step(qvals["qvalue"].values, qvals.num.values, where="post", **kwargs)

    return ax


def _groupby_max(df, by_cols, max_col):
    """Quickly get the indices for the maximum value of col"""
    idx = (
        df.sample(frac=1)
        .sort_values(list(by_cols) + [max_col], axis=0)
        .drop_duplicates(list(by_cols), keep="last")
        .index
    )

    return idx


def _new_column(name, df):
    """Add a new column, ensuring a unique name"""
    new_name = name
    cols = set(df.columns)
    i = 0
    while new_name in cols:
        new_name = name + "_" + str(i)
        i += 1

    return new_name
