"""One of the primary purposes of mokapot is to assign confidence estimates to
PSMs. This task is accomplished by ranking PSMs according to a score and using
an appropriate confidence estimation procedure for the type of data. mokapot
can provide confidence estimates based any score, regardless of whether it was
the result of a learned :py:func:`~mokapot.model.Model` instance or provided
independently.

The following classes store the confidence estimates for a dataset based on the
provided score. They provide utilities to access, save, and plot these
estimates for the various relevant levels (i.e. PSMs, peptides, and proteins).
The :py:func:`LinearConfidence` class is appropriate for most data-dependent
acquisition proteomics datasets.

We recommend using the :py:func:`~mokapot.brew()` function or the
:py:meth:`~mokapot.LinearPsmDataset.assign_confidence()` method to obtain these
confidence estimates, rather than initializing the classes below directly.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from triqler import qvality
from joblib import Parallel, delayed
from typeguard import typechecked

from . import qvalues
from .peps import peps_from_scores
from .utils import (
    create_chunks,
    groupby_max,
    convert_targets_column,
    merge_sort,
    map_columns_to_indices
)
from .dataset import read_file, PsmDataset, OnDiskPsmDataset
from .picked_protein import picked_protein
from .writers import to_flashlfq, to_txt
from .parsers.pin import read_file_in_chunks
from .constants import CONFIDENCE_CHUNK_SIZE

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
class GroupedConfidence:
    """Perform grouped confidence estimation for a collection of PSMs.

    Groups are defined by the :py:class:`~mokapot.dataset.LinearPsmDataset`.
    Confidence estimates for each group can be retrieved by using the group
    name as an attribute, or from the
    :py:meth:`~GroupedConfidence.group_confidence_estimates` property.

    Parameters
    ----------
    psms : OnDiskPsmDataset
        A collection of PSMs.
    rng : int or np.random.Generator, optional
        A seed or generator used for cross-validation split creation and to
        break ties, or ``None`` to use the default random number generator
        state.
    scores : np.ndarray
        A vector containing the score of each PSM.
    desc : bool
        Are higher scores better?
    eval_fdr : float
        The FDR threshold at which to report performance. This parameter
        has no affect on the analysis itself, only logging messages.
    dest_dir : str or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    sep : str, optional
        The delimiter to use.
    decoys : bool, optional
        Save decoys confidence estimates as well?
    combine : bool, optional
            Should groups be combined into a single file?
    """

    @typechecked
    def __init__(
        self,
        psms,
        scores,
        max_workers,
        desc=True,
        eval_fdr=0.01,
        decoys=False,
        dest_dir: Path | None = None,
        sep="\t",
        proteins=None,
        combine=False,
        prefixes=None,
        rng=0,
        peps_error=False,
    ):
        """Initialize a GroupedConfidence object"""
        data = read_file(psms.filename, use_cols=list(psms.columns))
        self.group_column = psms.group_column
        psms.group_column = None
        # Do TDC to eliminate multiples PSMs for a spectrum that may occur
        # in different groups.
        keep = "last" if desc else "first"
        scores = (
            pd.Series(scores, index=data.index).sample(frac=1).sort_values()
        )

        idx = (
            data.loc[scores.index, :]
            .drop_duplicates(psms.spectrum_columns, keep=keep)
            .index
        )
        append_to_group = False
        group_file = dest_dir / f"{prefixes}group_psms.csv"
        for group, group_df in data.groupby(self.group_column):
            LOGGER.info("Group: %s == %s", self.group_column, group)
            tdc_winners = group_df.index.intersection(idx)
            group_psms = group_df.loc[tdc_winners, :]
            group_scores = scores.loc[group_psms.index].values

            group_psms.to_csv(group_file, sep="\t", index=False)
            psms.filename = group_file
            assign_confidence(
                [psms],
                max_workers,
                [group_scores],
                descs=[desc],
                eval_fdr=eval_fdr,
                dest_dir=dest_dir,
                sep=sep,
                decoys=decoys,
                proteins=proteins,
                group_column=group,
                combine=combine,
                prefixes=prefixes,
                append_to_output_file=append_to_group,
                rng=rng,
                peps_error=peps_error,
            )
            if combine:
                append_to_group = True
            group_file.unlink()

    @property
    def group_confidence_estimates(self):
        """A dictionary of the confidence estimates for each group."""
        return self._group_confidence_estimates

    @property
    def groups(self):
        """The groups for confidence estimation"""
        return list(self._group_confidence_estimates.keys())

    def to_txt(
        self,
        dest_dir=None,
        file_root=None,
        sep="\t",
        decoys=False,
        combine=False,
    ):
        """Save confidence estimates to delimited text files.

        Parameters
        ----------
        dest_dir : str or None, optional
            The directory in which to save the files. `None` will use the
            current working directory.
        file_root : str or None, optional
            An optional prefix for the confidence estimate files. The suffix
            will be "mokapot.{level}.txt", where "{level}" indicates the level
            at which confidence estimation was performed (i.e. PSMs, peptides,
            proteins) if :code:`combine=True`. If :code:`combine=False` (the
            default), additionally the group value is prepended, yeilding a
            suffix "{group}.mokapot.{level}.txt".
        sep : str, optional
            The delimiter to use.
        decoys : bool, optional
            Save decoys confidence estimates as well?
        combine : bool, optional
            Should groups be combined into a single file?

        Returns
        -------
        list of str
            The paths to the saved files.

        """
        if combine:
            res = self.group_confidence_estimates.values()
            ret_files = to_txt(
                res,
                dest_dir=dest_dir,
                file_root=file_root,
                sep=sep,
                decoys=decoys,
            )
            return ret_files

        ret_files = []
        for group, res in self.group_confidence_estimates.items():
            prefix = file_root + f".{group}"
            new_files = res.to_txt(
                dest_dir=dest_dir, file_root=prefix, sep=sep, decoys=decoys
            )
            ret_files.append(new_files)

        return ret_files

    def __repr__(self):
        """Nice printing."""
        ngroups = len(self.group_confidence_estimates)
        lines = [
            "A mokapot.confidence.GroupedConfidence object with "
            f"{ngroups} groups:\n"
        ]

        for group, conf in self.group_confidence_estimates.items():
            lines += [f"Group: {self.group_column} == {group}"]
            lines += ["-" * len(lines[-1])]
            lines += [str(conf)]

        return "\n".join(lines)

    def __getattr__(self, attr):
        """Make groups accessible easily"""
        try:
            return self.group_confidence_estimates[attr]
        except KeyError:
            raise AttributeError

    def __len__(self):
        """Report the number of groups"""
        return len(self.group_confidence_estimates)


class Confidence(object):
    """Estimate and store the statistical confidence for a collection of PSMs.

    :meta private:
    """

    _level_labs = {
        "psms": "PSMs",
        "peptides": "Peptides",
        "proteins": "Proteins",
        "csms": "Cross-Linked PSMs",
        "peptide_pairs": "Peptide Pairs",
    }

    def __init__(self, psms, proteins=None, rng=0):
        """Initialize a PsmConfidence object."""
        self._score_column = "score"
        self._target_column = psms.target_column
        self._protein_column = "proteinIds"
        self._rng = rng
        self._group_column = psms.group_column
        self._metadata_column = psms.metadata_columns

        self.scores = None
        self.targets = None
        self.qvals = None
        self.peps = None

        self._proteins = proteins

        # This attribute holds the results as DataFrames:
        self.confidence_estimates = {}
        self.decoy_confidence_estimates = {}

    def __getattr__(self, attr):
        if attr.startswith("__"):
            return super().__getattr__(attr)

        try:
            return self.confidence_estimates[attr]
        except KeyError:
            raise AttributeError

    @property
    def levels(self):
        """
        The available levels for confidence estimates.
        """
        return list(self.confidence_estimates.keys())

    def to_txt(self, data_path, columns, level, decoys, sep, out_paths):
        """Save confidence estimates to delimited text files.
        Parameters
        ----------
        data_path : Path
            File of unique psms or peptides.
        columns : List
            columns that will be used
        level : str
            the level at which confidence estimation was performed
        sep : str, optional
            The delimiter to use.
        decoys : bool, optional
            Save decoys confidence estimates as well?
        out_paths : List(Path)
            The output files where the results will be written

        Returns
        -------
        list of str
            The paths to the saved files.

        """
        # The columns here are usually the metadata_columns from confidence.assign_confidence
        # which are usually ['PSMId', 'Label', 'peptide', 'proteinIds', 'score']
        # Since, those are exactly the columns that are written there to the csv
        # files, it's not exactly clear, why they are passed along here anyway
        # (but let's assert that here)
        assert all(pd.read_csv(data_path, sep="\t", nrows=0).columns == columns)

        in_columns = [i for i in columns if i != self._target_column]
        chunked_csv_file_iterator = read_file_in_chunks(
            file=data_path,
            chunk_size=CONFIDENCE_CHUNK_SIZE,
            use_cols=in_columns
        )

        chunked = lambda list: create_chunks(list, chunk_size=CONFIDENCE_CHUNK_SIZE)

        protein_column = self._protein_column

        for data_chunk, qvals_chunk, peps_chunk, targets_chunk in zip(
            chunked_csv_file_iterator, chunked(self.qvals), chunked(self.peps), chunked(self.targets) ):
            data_chunk["qvals"] = qvals_chunk
            data_chunk["PEP"] = peps_chunk
            if level != "proteins" and protein_column is not None:
                # EZ: seems to move the proteinIds column to the back (last col)
                # todo: we should rather have an out_columns where the the
                # protein column is moved to last position and then we reindex
                # using the out_columns
                data_chunk[protein_column] = data_chunk.pop(protein_column)

            # EZ: the definitions of the columns are to be found in
            # assign_confidence (that's where the file headers are written)
            data_chunk.loc[targets_chunk, :].to_csv(
                out_paths[0], sep=sep, index=False, mode="a", header=None
            )
            if decoys:
                data_chunk.loc[~targets_chunk, :].to_csv(
                    out_paths[1], sep=sep, index=False, mode="a", header=None
                )
        return out_paths

    def _perform_tdc(self, psms, psm_columns):
        """Perform target-decoy competition.

        Parameters
        ----------
        psms : Dataframe

            Dataframe of percolator with metadata columns
            [SpecId, Label, ScanNr, ExpMass, Peptide, score, Proteins].

        psm_columns : str or list of str
            The columns that define a PSM.
        """
        psm_idx = groupby_max(psms, psm_columns, self._score_column)
        return psms.loc[psm_idx, :]

    def plot_qvalues(self, level="psms", threshold=0.1, ax=None, **kwargs):
        """Plot the cumulative number of discoveries over range of q-values.

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
        qvals = self.confidence_estimates[level]["mokapot q-value"]
        if qvals is None:
            raise ValueError(f"{level}-level estimates are unavailable.")

        ax = plot_qvalues(qvals, threshold=threshold, ax=ax, **kwargs)
        ax.set_xlabel("q-value")
        ax.set_ylabel(f"Accepted {self._level_labs[level]}")

        return ax


class LinearConfidence(Confidence):
    """Assign confidence estimates to a set of PSMs

    Estimate q-values and posterior error probabilities (PEPs) for PSMs and
    peptides when ranked by the provided scores.

    Parameters
    ----------
    psms : OnDiskPsmDataset
        A collection of PSMs.
    rng : int or np.random.Generator, optional
        A seed or generator used for cross-validation split creation and to
        break ties, or ``None`` to use the default random number generator
        state.
    level_paths : List(Path)
            Files with unique psms and unique peptides.
    levels : List(str)
        Levels at which confidence estimation was performed
    out_paths : List(Path)
        The output files where the results will be written
    desc : bool
        Are higher scores better?
    eval_fdr : float
        The FDR threshold at which to report performance. This parameter
        has no affect on the analysis itself, only logging messages.
    sep : str, optional
        The delimiter to use.
    decoys : bool, optional
        Save decoys confidence estimates as well?
    """

    def __init__(
        self,
        psms,
        level_paths,
        levels,
        out_paths,
        desc=True,
        eval_fdr=0.01,
        decoys=None,
        do_rollup=True,
        proteins=None,
        peps_error=False,
        sep="\t",
        rng=0,
        peps_algorithm="qvality",
        qvalue_algorithm="tdc",
    ):
        """Initialize a a LinearPsmConfidence object"""
        super().__init__(psms, proteins, rng)
        self._target_column = psms.target_column
        self._peptide_column = "peptide"
        self._protein_column = "proteinIds"
        self._eval_fdr = eval_fdr
        self.do_rollup = do_rollup

        self._assign_confidence(
            level_paths=level_paths,
            levels=levels,
            out_paths=out_paths,
            desc=desc,
            decoys=decoys,
            sep=sep,
            peps_error=peps_error,
            peps_algorithm=peps_algorithm,
            qvalue_algorithm=qvalue_algorithm
        )

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

        if self._proteins:
            base += (
                f"\t- Protein groups at q<={self._eval_fdr:g}: "
                f"{self.accepted['proteins']}\n"
            )

        return base

    def _num_accepted(self, level):
        """Calculate the number of accepted discoveries"""
        disc = self.confidence_estimates[level]
        if disc is not None:
            return (disc["q-value"] <= self._eval_fdr).sum()
        else:
            return None

    def _assign_confidence(
        self,
        level_paths,
        levels,
        out_paths,
        desc=True,
        decoys=False,
        peps_error=False,
        sep="\t",
        peps_algorithm="qvality",
        qvalue_algorithm="tdc",
    ):
        """
        Assign confidence to PSMs and peptides.

        Parameters
        ----------
        level_paths : List(Path)
            Files with unique psms and unique peptides.
        levels : List(str)
            the levels at which confidence estimation was performed
        out_paths : List(Path)
            The output files where the results will be written
        desc : bool
            Are higher scores better?
        sep : str, optional
            The delimiter to use.
        decoys : bool, optional
            Save decoys confidence estimates as well?
        """

        if self._proteins:
            data = read_file(level_paths[1])
            data = data.apply(pd.to_numeric, errors="ignore")
            convert_targets_column(
                data=data, target_column=self._target_column
            )
            proteins = picked_protein(
                data,
                self._target_column,
                self._peptide_column,
                self._score_column,
                self._proteins,
                self._rng,
            )
            proteins = proteins.sort_values(
                by=self._score_column, ascending=False
            ).reset_index(drop=True)
            proteins_path = level_paths[-1]
            proteins.to_csv(proteins_path, index=False, sep=sep)
            assert levels[-1] == "proteins" # todo
            out_paths += [
                _psms.with_suffix(".proteins")
                for _psms in out_paths[0]
            ]

            LOGGER.info("\t- Found %i unique protein groups.", len(proteins))
        for level, data_path, out_path in zip(levels, level_paths, out_paths):
            data = read_file(data_path, target_column=self._target_column)
            data_columns = list(data.columns)
            self.scores = data.loc[:, self._score_column].values
            self.targets = data.loc[:, self._target_column].astype(bool).values
            del data
            if all(self.targets):
                LOGGER.warning(
                    "No decoy PSMs remain for confidence estimation. "
                    "Confidence estimates may be unreliable."
                )

            # Estimate q-values and assign to dataframe
            LOGGER.info("Assigning q-values to %s (using %s algorithm) ...", level, qvalue_algorithm)
            self.qvals = qvalues.qvalues_from_scores(self.scores, self.targets, qvalue_algorithm)

            # Set scores to be the correct sign again:
            self.scores = self.scores * (desc * 2 - 1)
            # Logging update on q-values
            LOGGER.info(
                "\t- Found %i %s with q<=%g",
                (self.qvals[self.targets] <= self._eval_fdr).sum(),
                level,
                self._eval_fdr,
            )

            # Calculate PEPs

            LOGGER.info("Assigning PEPs to %s (using %s algorithm) ...", level, peps_algorithm)
            try:
                self.peps = peps_from_scores(self.scores, self.targets, peps_algorithm)
            except SystemExit as msg:
                if "no decoy hits available for PEP calculation" in str(msg):
                    self.peps = 0
                else:
                    raise
            if peps_error and all(self.peps == 1):
                raise ValueError("PEP values are all equal to 1.")

            logging.info(f"Writing {level} results...")

            self.to_txt(
                data_path,
                data_columns,
                level.lower(),
                decoys,
                sep,
                out_path,
            )

    def to_flashlfq(self, out_file="mokapot.flashlfq.txt"):
        """Save confidenct peptides for quantification with FlashLFQ.

        `FlashLFQ <https://github.com/smith-chem-wisc/FlashLFQ>`_ is an
        open-source tool for label-free quantification. For mokapot to save
        results in a compatible format, a few extra columns are required to
        be present, which specify the MS data file name, the theoretical
        peptide monoisotopic mass, the retention time, and the charge for each
        PSM. If these are not present, saving to the FlashLFQ format is
        disabled.

        Note that protein grouping in the FlashLFQ results will be more
        accurate if proteins were added for analysis with mokapot.

        Parameters
        ----------
        out_file : str, optional
            The output file to write.

        Returns
        -------
        str
            The path to the saved file.

        """
        return to_flashlfq(self, out_file)


class CrossLinkedConfidence(Confidence):
    """
    Assign confidence estimates to a set of cross-linked PSMs

    Estimate q-values and posterior error probabilities (PEPs) for
    cross-linked PSMs (CSMs) and the peptide pairs when ranked by the
    provided scores.

    Parameters
    ----------
    psms : OnDiskPsmDataset
        A collection of PSMs.
    level_paths : List(Path)
            Files with unique psms and unique peptides.
    out_paths : List(Path)
            The output files where the results will be written
    desc : bool
        Are higher scores better?
    """

    def __init__(
        self,
        psms,
        level_paths,
        out_paths,
        desc=True,
        decoys=None,
        peps_error=False,
        sep="\t",
    ):
        """Initialize a CrossLinkedConfidence object"""
        super().__init__(psms)
        self._target_column = psms.target_column
        self._peptide_column = "peptide"

        self._assign_confidence(
            level_paths=level_paths,
            out_paths=out_paths,
            desc=desc,
            decoys=decoys,
            sep=sep,
            peps_error=peps_error,
        )

    def _assign_confidence(
        self,
        level_paths,
        out_paths,
        desc=True,
        decoys=False,
        peps_error=False,
        sep="\t",
    ):
        """
        Assign confidence to PSMs and peptides.

        Parameters
        ----------
        level_paths : List(Path)
            Files with unique psms and unique peptides.
        out_paths : List(Path)
            The output files where the results will be written
        desc : bool
            Are higher scores better?
        sep : str, optional
            The delimiter to use.
        decoys : bool, optional
            Save decoys confidence estimates as well?
        """

        levels = ("csms", "peptide_pairs")

        for level, data_path, out_path in zip(levels, level_paths, out_paths):
            data = read_file(
                data_path,
                use_cols=self._metadata_column + ["score"],
                target_column=self._target_column,
            )
            data_columns = list(data.columns)
            self.scores = data.loc[:, self._score_column].values
            self.targets = data.loc[:, self._target_column].astype(bool).values
            self.qvals = qvalues.crosslink_tdc(self.scores, self.targets, desc)

            _, self.peps = qvality.getQvaluesFromScores(
                self.scores[self.targets == 2], self.scores[~self.targets]
            )
            if peps_error and all(self.peps == 1):
                raise ValueError("PEP values are all equal to 1.")
            logging.info(f"Writing {level} results...")
            self.to_txt(data_path, data_columns, level.lower(), decoys, sep)


# Functions -------------------------------------------------------------------


@typechecked
def assign_confidence(
    psms,
    max_workers,
    scores=None,
    descs=None,
    eval_fdr=0.01,
    dest_dir : Path | None =None,
    file_root : str = "",
    sep="\t",
    prefixes : list[str|None] | None = None,
    decoys=False,
    do_rollup=True,
    proteins=None,
    group_column=None,
    combine=False,
    append_to_output_file=False,
    rng=0,
    peps_error=False,
    peps_algorithm="qvality",
    qvalue_algorithm="tdc",
):
    """Assign confidence to PSMs peptides, and optionally, proteins.

    Parameters
    ----------
    max_workers
    psms : OnDiskPsmDataset
        A collection of PSMs.
    rng : int or np.random.Generator, optional
        A seed or generator used for cross-validation split creation and to
        break ties, or ``None`` to use the default random number generator
        state.
    scores : numpy.ndarray
        The scores by which to rank the PSMs. The default, :code:`None`,
        uses the feature that accepts the most PSMs at an FDR threshold of
        `eval_fdr`.
    descs : [bool]
        Are higher scores better?
    eval_fdr : float
        The FDR threshold at which to report and evaluate performance. If
        `scores` is not :code:`None`, this parameter has no affect on the
        analysis itself, but does affect logging messages and the FDR
        threshold applied for some output formats, such as FlashLFQ.
    dest_dir : Path or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    sep : str, optional
        The delimiter to use.
    prefixes : [str]
        The prefixes added to all output file names.

    decoys : bool, optional
        Save decoys confidence estimates as well?
    do_rollup: bool
        do we apply rollup on peptides, modified peptides etc.?
    proteins: Proteins, optional
        collection of proteins
    combine : bool, optional
            Should groups be combined into a single file?
    group_column : str, optional
        A factor to by which to group PSMs for grouped confidence
        estimation.
    append_to_output_file: bool
        do we append results to file ?

    Returns
    -------
    LinearConfidence
        A :py:class:`~mokapot.confidence.LinearConfidence` object storing
        the confidence estimates for the collection of PSMs.
    """

    if scores is None:
        scores = []
        for _psms in psms:
            feat, _, _, desc = _psms.find_best_feature(eval_fdr)
            LOGGER.info("Selected %s as the best feature.", feat)
            scores.append(
                read_file(file_name=_psms.filename, use_cols=[feat])[
                    feat
                ].values
            )

    curr_psms = psms[0] # just take the first one for info (need to make sure the other are the same)

    # todo: maybe use a collections.namedtuple for all the level info instead of all the ?
    # from collections import namedtuple
    # LevelInfo = namedtuple('LevelInfo', ['name', 'data_path', 'deduplicate', 'colnames', 'colindices'])

    levels = ["psms"]
    level_data_path = [dest_dir / f"{file_root}psms.csv"]
    level_dedup = [False]
    level_input_columns = [curr_psms.spectrum_columns]

    if do_rollup:
        level_columns = curr_psms.level_columns
        level_columns = level_columns[:1] # strip for the moment to peptides, todo: remove this later

        for level_column in level_columns:
            level = level_column.lower() + "s"  # e.g. Peptide to peptides
            levels.append(level)
            level_data_path.append(dest_dir / f"{file_root}{level}.csv")
            level_dedup.append(True)
            level_input_columns.append(level_column)

    level_input_colidx = map_columns_to_indices(level_input_columns, curr_psms.columns)

    if proteins:
        # if do_rollup:
        #     # Rather: both don't work together
        #     raise NotImplementedError("Does not work for proteins yet")
        levels.append("proteins")
        level_data_path.append(dest_dir / f"{file_root}proteins.csv")
        level_dedup.append("I don't know")
        level_input_columns.append(curr_psms.protein_column)

    out_columns_psms_peps = [
        "PSMId",
        "peptide",
        "score",
        "q-value",
        "posterior_error_prob",
        "proteinIds",
    ]
    out_columns_proteins = [
        "mokapot protein group",
        "best peptide",
        "stripped sequence",
        "score",
        "q-value",
        "posterior_error_prob",
    ]
    output_columns_map = {
        "psms":     out_columns_psms_peps,
        "peptides": out_columns_psms_peps,
        "proteins": out_columns_proteins,
    }

    for _psms, score, desc, prefix in zip(psms, scores, descs, prefixes):
        metadata_columns = [
            "PSMId",
            _psms.target_column,
            "peptide",
            "proteinIds",
            "score",
        ]
        if _psms.group_column is None:
            file_prefix = file_root
            if prefix:
                file_prefix = f"{file_prefix}{prefix}."

            out_files = []
            for level in levels:
                output_columns = output_columns_map[level]
                if group_column is not None and not combine:
                    file_prefix_group = f"{file_prefix}{group_column}."
                else:
                    file_prefix_group = file_prefix
                outfile_t = dest_dir / f"{file_prefix_group}targets.{level}"
                outfile_d = dest_dir / f"{file_prefix_group}decoys.{level}"
                if not append_to_output_file:
                    with open(outfile_t, "w") as fp:
                        fp.write(f"{sep.join(output_columns)}\n")
                out_files.append([outfile_t])
                if decoys:
                    if not append_to_output_file:
                        with open(outfile_d, "w") as fp:
                            fp.write(f"{sep.join(output_columns)}\n")
                    out_files[-1].append(outfile_d)

            # Read from the input psms (PsmDataset) and write into smaller
            # sorted files, by

            # a) Create a reader that only reads columns given in
            #    psms.metadata_columns in chunks of size CONFIDENCE_CHUNK_SIZE
            reader = read_file_in_chunks(
                file=_psms.filename,
                chunk_size=CONFIDENCE_CHUNK_SIZE,
                use_cols=_psms.metadata_columns,
            )

            # b) Split the scores in chunks of the same size
            scores_slices = create_chunks(
                score, chunk_size=CONFIDENCE_CHUNK_SIZE
            )

            # c) Write those chunks in parallel, where the columns are given
            #    by psms.metadata plus the "scores" column
            #    (NB: after the last change the columns are now indeed in the
            #     order given by metadata_columns and not by file header order)
            Parallel(n_jobs=max_workers, require="sharedmem")(
                delayed(save_sorted_metadata_chunks)(
                    chunk_metadata,
                    score_chunk,
                    _psms,
                    do_rollup,
                    i,
                    sep,
                    dest_dir,
                    f"{file_prefix}scores_metadata"
                )
                for chunk_metadata, score_chunk, i in zip(
                    reader, scores_slices, range(len(scores_slices))
                )
            )
            reader_columns = _psms.metadata_columns + ["score"]


            scores_metadata_paths = list(dest_dir.glob(f"{file_prefix}scores_metadata_*"))
            iterable_sorted = merge_sort(
                scores_metadata_paths, col_score="score", sep=sep
            )
            LOGGER.info("Assigning confidence...")
            LOGGER.info("Performing target-decoy competition...")
            LOGGER.info(
                "Keeping the best match per %s columns...",
                "+".join(_psms.spectrum_columns),
            )

            with open(level_data_path[0], "w") as f_psm:
                f_psm.write(f"{sep.join(metadata_columns)}\n")

            input_columns = [
                _psms.specId_column,
                _psms.target_column,
                _psms.spectrum_columns[0], # scannr todo: (this is ugly)
                _psms.spectrum_columns[1], # expmass todo: (this also, needs to change)
                _psms.peptide_column,
                _psms.protein_column,
                "score",
            ]
            input_colidx = map_columns_to_indices(input_columns, reader_columns)

            if do_rollup:
                with open(level_data_path[1], "w") as f_peptide:
                    f_peptide.write(f"{sep.join(metadata_columns)}\n")

                (
                    unique_psms,
                    unique_peptides,
                ) = get_unique_psms_and_peptides(
                    iterable=iterable_sorted,
                    input_colidx=input_colidx,
                    out_psms=level_data_path[0],
                    out_peptides=level_data_path[1],
                    sep=sep,
                )
                LOGGER.info(
                    "\t- Found %i PSMs from unique spectra.", unique_psms
                )
                LOGGER.info("\t- Found %i unique peptides.", unique_peptides)
            else:
                (specid_idx, label_idx, scannr_idx, expmass_idx, peptide_idx,
                 proteins_idx, score_idx) = tuple(input_colidx)
                output_colidx = [specid_idx, label_idx, peptide_idx,
                                 proteins_idx, score_idx]

                n_psms = 0
                for row in iterable_sorted:
                    n_psms += 1
                    row = np.array(row)
                    num_cols = len(row)
                    # input_colidx = np.array([0, 1, num_cols-3, num_cols-2, num_cols-1]) # todo: get from bla
                    with open(level_data_path[0], "a") as f_psm:
                        f_psm.write(sep.join(row[output_colidx]))
                LOGGER.info("\t- Found %i PSMs.", n_psms)

            # xno-commit
            # [os.remove(sc_path) for sc_path in scores_metadata_paths]

            LinearConfidence(
                psms=_psms,
                levels=levels,
                level_paths=level_data_path,
                out_paths=out_files,
                eval_fdr=eval_fdr,
                desc=desc,
                sep=sep,
                decoys=decoys,
                do_rollup=do_rollup,
                proteins=proteins,
                rng=rng,
                peps_error=peps_error,
                peps_algorithm=peps_algorithm,
                qvalue_algorithm=qvalue_algorithm,
            )
            if prefix is None:
                append_to_output_file = True
        else:
            LOGGER.info("Assigning confidence within groups...")
            GroupedConfidence(
                _psms,
                score,
                max_workers,
                eval_fdr=eval_fdr,
                desc=desc,
                dest_dir=dest_dir,
                sep=sep,
                decoys=decoys,
                proteins=proteins,
                combine=combine,
                prefixes=[prefix],
                rng=rng,
                peps_error=peps_error,
            )


@typechecked
def save_sorted_metadata_chunks(
    chunk_metadata : pd.DataFrame, score_chunk, psms, do_rollup, i, sep, dest_dir : Path, file_prefix : str
):
    chunk_metadata = convert_targets_column(
        data=chunk_metadata.apply(pd.to_numeric, errors="ignore"),
        target_column=psms.target_column,
    )
    chunk_metadata["score"] = score_chunk
    chunk_metadata.sort_values(by="score", ascending=False, inplace=True)

    if do_rollup:
        chunk_metadata = chunk_metadata.drop_duplicates(psms.spectrum_columns)

    chunk_metadata.to_csv(
        dest_dir / f"{file_prefix}_{i}.csv",
        sep=sep,
        index=False,
        mode="w",
    )


def get_unique_psms_and_peptides(iterable, input_colidx, out_psms, out_peptides, sep):
    seen_psm = set()
    seen_peptide = set()
    f_psm = open(out_psms, "a")
    f_peptide = open(out_peptides, "a")

    specid_idx, label_idx, scannr_idx, expmass_idx, peptide_idx, proteins_idx, score_idx = tuple(input_colidx)
    output_colidx = [specid_idx, label_idx, peptide_idx, proteins_idx, score_idx]

    for line_list in iterable:
        line_hash_psm = tuple([int(line_list[scannr_idx]), float(line_list[expmass_idx])])
        line_hash_peptide = line_list[peptide_idx]
        line = (np.array(line_list))[output_colidx]
        if line_hash_psm not in seen_psm:
            seen_psm.add(line_hash_psm)
            f_psm.write(f"{sep.join(line)}")
            if line_hash_peptide not in seen_peptide:
                seen_peptide.add(line_hash_peptide)
                f_peptide.write(f"{sep.join(line)}")
    f_psm.close()
    f_peptide.close()
    return [len(seen_psm), len(seen_peptide)]


@typechecked
def get_unique_peptides_from_psms(
    iterable, peptide_col_index, out_peptides : Path, sep
):
    f_peptide = open(out_peptides, "a")
    seen_peptide = set()
    for line_list in iterable:
        line_hash_peptide = line_list[peptide_col_index]
        if line_hash_peptide not in seen_peptide:
            seen_peptide.add(line_hash_peptide)
            f_peptide.write(f"{sep.join(line_list[:4] + [line_list[-1]])}")

    f_peptide.close()
    return len(seen_peptide)


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
    ax.set_ylabel("Discoveries")

    ax.step(qvals["qvalue"].values, qvals.num.values, where="post", **kwargs)

    return ax


def _new_column(name, df):
    """Add a new column, ensuring a unique name"""
    new_name = name
    cols = set(df.columns)
    i = 0
    while new_name in cols:
        new_name = name + "_" + str(i)
        i += 1

    return new_name
