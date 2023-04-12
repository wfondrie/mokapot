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
import os
import glob
from pathlib import Path

import logging
import pandas as pd
import matplotlib.pyplot as plt
from triqler import qvality
from joblib import Parallel, delayed

from . import qvalues
from .utils import (
    create_chunks,
    groupby_max,
    convert_targets_column,
    merge_sort,
    get_unique_psms_and_peptides,
)
from .dataset import read_file
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

    def __init__(
        self,
        psms,
        scores,
        desc=True,
        eval_fdr=0.01,
        decoys=False,
        dest_dir=None,
        sep="\t",
        proteins=None,
        combine=False,
        prefixes=None,
    ):
        """Initialize a GroupedConfidence object"""
        data = read_file(
            psms.filename,
            use_cols=list(psms.feature_columns) + list(psms.metadata_columns),
        )
        self.group_column = psms.group_column
        psms.group_column = None
        scores = scores * (desc * 2 - 1)

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

        self._group_confidence_estimates = {}
        append_to_group = False
        group_file = "group_psms.csv"
        for group, group_df in data.groupby(self.group_column):
            LOGGER.info("Group: %s == %s", self.group_column, group)
            tdc_winners = group_df.index.intersection(idx)
            group_psms = group_df.loc[tdc_winners, :]
            group_scores = scores.loc[group_psms.index].values + 1
            group_psms.to_csv(group_file, sep="\t", index=False)
            psms.filename = group_file
            assign_confidence(
                [psms],
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
            )
            if combine:
                append_to_group = True
            os.remove(group_file)

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

    def __init__(self, psms, proteins=None):
        """Initialize a PsmConfidence object."""
        self._score_column = "score"
        self._target_column = psms.target_column
        self._protein_column = "proteinIds"
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
        reader = read_file_in_chunks(
            file=data_path,
            chunk_size=CONFIDENCE_CHUNK_SIZE,
            use_cols=[i for i in columns if i != self._target_column],
        )

        self.scores = create_chunks(
            self.scores, chunk_size=CONFIDENCE_CHUNK_SIZE
        )
        self.qvals = create_chunks(
            self.qvals, chunk_size=CONFIDENCE_CHUNK_SIZE
        )
        self.peps = create_chunks(self.peps, chunk_size=CONFIDENCE_CHUNK_SIZE)
        self.targets = create_chunks(
            self.targets, chunk_size=CONFIDENCE_CHUNK_SIZE
        )

        for data_in, score_in, qvals_in, pep_in, target_in in zip(
            reader, self.scores, self.qvals, self.peps, self.targets
        ):
            data_in = data_in.apply(pd.to_numeric, errors="ignore")
            data_in["score"] = score_in
            data_in["qvals"] = qvals_in
            data_in["PEP"] = pep_in
            if level != "proteins" and self._protein_column is not None:
                data_in[self._protein_column] = data_in.pop(
                    self._protein_column
                )
            data_in.loc[target_in, :].to_csv(
                out_paths[0], sep=sep, index=False, mode="a", header=None
            )
            if decoys:
                data_in.loc[~target_in, :].to_csv(
                    out_paths[1], sep=sep, index=False, mode="a", header=None
                )
        os.remove(data_path)
        return out_paths

    def _perform_tdc(self, psms, psm_columns):
        """Perform target-decoy competition.

        Parameters
        ----------
        psms : Dataframe
            Dataframe of percolator with metadata columns [SpecId, Label, ScanNr, ExpMass, Peptide, score, Proteins].
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
        deduplication=True,
        proteins=None,
        sep="\t",
    ):
        """Initialize a a LinearPsmConfidence object"""
        super().__init__(psms, proteins)
        self._target_column = psms.target_column
        self._psm_columns = psms.spectrum_columns
        self._peptide_column = psms.peptide_column
        self._protein_column = "proteinIds"
        self._eval_fdr = eval_fdr
        self.deduplication = deduplication

        self._assign_confidence(
            level_paths=level_paths,
            levels=levels,
            out_paths=out_paths,
            desc=desc,
            decoys=decoys,
            sep=sep,
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
        sep="\t",
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
                self.rng,
            )
            proteins_path = "proteins.csv"
            proteins.to_csv(proteins_path, index=False, sep=sep)
            levels += ["proteins"]
            level_paths += [proteins_path]
            out_paths += [
                os.path.sep.join(
                    _psms.split(os.path.sep)[:-1]
                    + [
                        ".".join(
                            _psms.split(os.path.sep)[-1].split(".")[:-1]
                            + ["proteins"]
                        )
                    ]
                )
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
            LOGGER.info("Assiging q-values to %s...", level)
            self.qvals = qvalues.tdc(self.scores, self.targets, desc=True)

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
            LOGGER.info("Assiging PEPs to %s...", level)
            try:
                _, self.peps = qvality.getQvaluesFromScores(
                    self.scores[self.targets],
                    self.scores[~self.targets],
                    includeDecoys=True,
                )
            except SystemExit as msg:
                if "no decoy hits available for PEP calculation" in str(msg):
                    self.peps = 0
                else:
                    raise

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
        sep="\t",
    ):
        """Initialize a CrossLinkedConfidence object"""
        super().__init__(psms)
        self._target_column = psms.target_column
        self._psm_columns = psms.spectrum_columns
        self._peptide_column = psms.peptide_column

        self._assign_confidence(
            level_paths=level_paths,
            out_paths=out_paths,
            desc=desc,
            decoys=decoys,
            sep=sep,
        )

    def _assign_confidence(
        self,
        level_paths,
        out_paths,
        desc=True,
        decoys=False,
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
            logging.info(f"Writing {level} results...")
            self.to_txt(data_path, data_columns, level.lower(), decoys, sep)


# Functions -------------------------------------------------------------------
def assign_confidence(
    psms,
    scores=None,
    descs=None,
    eval_fdr=0.01,
    dest_dir=None,
    sep="\t",
    prefixes=None,
    decoys=False,
    deduplication=True,
    proteins=None,
    group_column=None,
    combine=False,
    append_to_output_file=False,
):
    """Assign confidence to PSMs peptides, and optionally, proteins.

    Parameters
    ----------
    psms : OnDiskPsmDataset
        A collection of PSMs.
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
    dest_dir : str or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    sep : str, optional
        The delimiter to use.
    prefixes : [str]
        The prefixes added to all output file names.
    decoys : bool, optional
        Save decoys confidence estimates as well?
    deduplication: bool
        do we apply deduplication on peptides?
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
                read_file(file_name=_psms.filename, use_cols=[feat]).values
            )

    psms_path = "psms.csv"
    peptides_path = "peptides.csv"
    levels = ["psms"]
    level_data_path = [psms_path]
    if deduplication:
        levels.append("peptides")
        level_data_path.append(peptides_path)
    if proteins:
        levels.append("proteins")

    metadata_columns = ["PSMId", "Label", "peptide", "proteinIds", "score"]
    output_columns = [
        "PSMId",
        "peptide",
        "score",
        "q-value",
        "posterior_error_prob",
        "proteinIds",
    ]

    for _psms, score, desc, prefix in zip(psms, scores, descs, prefixes):
        if _psms.group_column is None:
            out_files = []
            for level in levels:
                dest_dir_prefix = dest_dir
                if prefix is not None:
                    dest_dir_prefix = dest_dir_prefix + f"{prefix}."
                if group_column is not None and not combine:
                    dest_dir_prefix = f"{dest_dir_prefix}{group_column}."
                outfile_t = str(dest_dir_prefix) + f"targets.{level}"
                outfile_d = str(dest_dir_prefix) + f"decoys.{level}"
                if not append_to_output_file:
                    with open(outfile_t, "w") as fp:
                        fp.write(f"{sep.join(output_columns)}\n")
                out_files.append([outfile_t])
                if decoys:
                    if not append_to_output_file:
                        with open(outfile_d, "w") as fp:
                            fp.write(f"{sep.join(output_columns)}\n")
                    out_files[-1].append(outfile_d)
            reader = read_file_in_chunks(
                file=_psms.filename,
                chunk_size=CONFIDENCE_CHUNK_SIZE,
                use_cols=_psms.metadata_columns,
            )
            scores_slices = create_chunks(
                score, chunk_size=CONFIDENCE_CHUNK_SIZE
            )

            Parallel(n_jobs=-1, require="sharedmem")(
                delayed(save_sorted_metadata_chunks)(
                    chunk_metadata,
                    score_chunk,
                    _psms,
                    deduplication,
                    i,
                    sep,
                )
                for chunk_metadata, score_chunk, i in zip(
                    reader, scores_slices, range(len(scores_slices))
                )
            )

            scores_metadata_paths = glob.glob("scores_metadata_*")
            iterable_sorted = merge_sort(
                scores_metadata_paths, col_score="score", sep=sep
            )
            LOGGER.info("Assigning confidence...")
            LOGGER.info("Performing target-decoy competition...")
            LOGGER.info(
                "Keeping the best match per %s columns...",
                "+".join(_psms.spectrum_columns),
            )

            with open(psms_path, "w") as f_psm:
                f_psm.write(f"{sep.join(metadata_columns)}\n")

            if deduplication:
                with open(peptides_path, "w") as f_peptide:
                    f_peptide.write(f"{sep.join(metadata_columns)}\n")

                (
                    unique_psms,
                    unique_peptides,
                ) = get_unique_psms_and_peptides(
                    iterable=iterable_sorted,
                    out_psms="psms.csv",
                    out_peptides="peptides.csv",
                    sep=sep,
                )
                LOGGER.info(
                    "\t- Found %i PSMs from unique spectra.", unique_psms
                )
                LOGGER.info("\t- Found %i unique peptides.", unique_peptides)
            else:
                n_psms = 0
                for row in iterable_sorted:
                    n_psms += 1
                    with open(psms_path, "a") as f_psm:
                        f_psm.write(
                            sep.join(
                                [row[0], row[1], row[-3], row[-2], row[-1]]
                            )
                        )
                LOGGER.info("\t- Found %i PSMs.", n_psms)

            [os.remove(sc_path) for sc_path in scores_metadata_paths]

            LinearConfidence(
                psms=_psms,
                levels=levels,
                level_paths=level_data_path,
                out_paths=out_files,
                eval_fdr=eval_fdr,
                desc=desc,
                sep=sep,
                decoys=decoys,
                deduplication=deduplication,
                proteins=proteins,
            )
            if prefix is None:
                append_to_output_file = True
        else:
            LOGGER.info("Assigning confidence within groups...")
            GroupedConfidence(
                _psms,
                score,
                eval_fdr=eval_fdr,
                desc=desc,
                dest_dir=dest_dir,
                sep=sep,
                decoys=decoys,
                proteins=proteins,
                combine=combine,
                prefixes=[prefix],
            )


def save_sorted_metadata_chunks(
    chunk_metadata, score_chunk, psms, deduplication, i, sep
):
    chunk_metadata["score"] = score_chunk
    chunk_metadata.sort_values(by="score", ascending=False, inplace=True)
    if deduplication:
        chunk_metadata = chunk_metadata.drop_duplicates(psms.spectrum_columns)
    chunk_metadata.to_csv(
        f"scores_metadata_{i}.csv",
        sep=sep,
        index=False,
        mode="w",
    )


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
