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

from __future__ import annotations

import logging
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from typeguard import typechecked
import os

from . import qvalues
from .peps import peps_from_scores
from .utils import (
    create_chunks,
    groupby_max,
    convert_targets_column,
    merge_sort,
    get_dataframe_from_records,
)
from .dataset import OnDiskPsmDataset
from .picked_protein import picked_protein
from .writers import to_flashlfq
from .tabular_data import (
    TabularDataWriter,
    TabularDataReader,
    get_score_column_type,
)
from .confidence_writer import write_confidences
from .constants import CONFIDENCE_CHUNK_SIZE

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
class Confidence(object):
    """Estimate and store the statistical confidence for a collection of PSMs.

    :meta private:
    """

    _level_labs = {
        "psms": "PSMs",
        "peptides": "Peptides",
        "proteins": "Proteins",
        "peptide_pairs": "Peptide Pairs",
    }

    def __init__(self, psms, proteins=None, rng=0):
        """Initialize a PsmConfidence object."""
        self._score_column = "score"
        self._target_column = psms.target_column
        self._protein_column = "proteinIds"
        self._rng = rng
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

    def write_to_disk(
        self,
        data_path,
        columns,
        level,
        decoys,
        out_paths,
        sqlite_path=None,
    ):
        """Save confidence estimates to delimited text files.
        Parameters
        ----------
        data_path : Path
            File of unique psms or peptides.
        columns : List
            columns that will be used
        level : str
            the level at which confidence estimation was performed
        decoys : bool, optional
            Save decoys confidence estimates as well?
        out_paths : List(Path)
            The output files where the results will be written

        Returns
        -------
        list of str
            The paths to the saved files.

        """
        # The columns here are usually the metadata_columns from
        # `confidence.assign_confidence`
        # which are usually:
        #   ['PSMId', 'Label', 'peptide', 'proteinIds', 'score']
        # Since, those are exactly the columns that are written there to the
        # csv files, it's not exactly clear, why they are passed along
        # here anyway (but let's assert that here)
        reader = TabularDataReader.from_path(data_path)
        assert reader.get_column_names() == columns

        in_columns = [i for i in columns if i != self._target_column]
        chunked_data_iterator = reader.get_chunked_data_iterator(
            CONFIDENCE_CHUNK_SIZE, in_columns
        )

        # Note: the out_columns need to match those in assign_confidence
        #   (out_files)
        qvalue_column = "q_value"
        pep_column = "posterior_error_prob"
        out_columns = in_columns + [qvalue_column, pep_column]
        protein_column = self._protein_column
        if level != "proteins" and protein_column is not None:
            # Move the "proteinIds" column to the back for dubious reasons
            # todo: rather than this fiddeling here, we should have a column
            #  mapping that does this
            out_columns.remove(protein_column)
            out_columns.append(protein_column)

        def chunked(list):
            return create_chunks(list, chunk_size=CONFIDENCE_CHUNK_SIZE)

        # Replacing csv target and decoys results path with sqlite db path
        if sqlite_path:
            out_paths = [sqlite_path]

        write_confidences(
            chunked_data_iterator,
            chunked(self.qvals),
            chunked(self.peps),
            chunked(self.targets),
            out_paths,
            decoys,
            level,
            out_columns,
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
        psm_idx = groupby_max(
            psms, psm_columns, self._score_column, rng=self._rng
        )
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


@typechecked
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
    levels : list[str]
        Levels at which confidence estimation was performed
    level_paths : list[Path]
            Files with unique psms and unique peptides.
    out_paths : list[list[Path]]
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
        level_paths: list[Path],
        levels: list[str],
        out_paths: list[list[Path]],
        desc=True,
        eval_fdr=0.01,
        decoys=None,
        deduplication=True,
        do_rollup=True,
        proteins=None,
        peps_error=False,
        sep="\t",
        rng=0,
        peps_algorithm="qvality",
        qvalue_algorithm="tdc",
        sqlite_path=None,
    ):
        """Initialize a a LinearPsmConfidence object"""
        super().__init__(psms, proteins, rng)
        self._target_column = psms.target_column
        self._peptide_column = "peptide"
        self._protein_column = "proteinIds"
        self._eval_fdr = eval_fdr
        self.deduplication = deduplication
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
            qvalue_algorithm=qvalue_algorithm,
            sqlite_path=sqlite_path,
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
        sqlite_path=None,
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
            data = TabularDataReader.from_path(level_paths[1]).read()
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
            assert levels[-1] == "proteins"
            assert len(levels) == len(level_paths)
            proteins_path = level_paths[-1]
            proteins.to_csv(proteins_path, index=False, sep=sep)
            out_paths += [
                _psms.with_suffix(".proteins") for _psms in out_paths[0]
            ]
            LOGGER.info("\t- Found %i unique protein groups.", len(proteins))

        for level, data_path, out_path in zip(levels, level_paths, out_paths):
            data = TabularDataReader.from_path(data_path).read()
            if self._target_column:
                data = convert_targets_column(data, self._target_column)
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
            LOGGER.info(
                "Assigning q-values to %s (using %s algorithm) ...",
                level,
                qvalue_algorithm,
            )
            self.qvals = qvalues.qvalues_from_scores(
                self.scores, self.targets, qvalue_algorithm
            )

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
            LOGGER.info(
                "Assigning PEPs to %s (using %s algorithm) ...",
                level,
                peps_algorithm,
            )
            try:
                self.peps = peps_from_scores(
                    self.scores, self.targets, peps_algorithm
                )
            except SystemExit as msg:
                if "no decoy hits available for PEP calculation" in str(msg):
                    self.peps = 0
                else:
                    raise
            if peps_error and all(self.peps == 1):
                raise ValueError("PEP values are all equal to 1.")

            logging.info(f"Writing {level} results...")
            self.write_to_disk(
                data_path,
                data_columns,
                level.lower(),
                decoys,
                out_path,
                sqlite_path,
            )
            if sqlite_path:
                [os.unlink(path) for path in out_path]
            os.unlink(data_path)

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


# Functions -------------------------------------------------------------------
@typechecked
def assign_confidence(
    psms: list[OnDiskPsmDataset],
    max_workers,
    scores=None,
    descs=None,
    eval_fdr=0.01,
    dest_dir: Path | None = None,
    file_root: str = "",
    sep="\t",
    prefixes: list[str | None] | None = None,
    decoys=False,
    deduplication=True,
    do_rollup=True,
    proteins=None,
    combine=False,
    append_to_output_file=False,
    rng=0,
    peps_error=False,
    peps_algorithm="qvality",
    qvalue_algorithm="tdc",
    sqlite_path=None,
):
    """Assign confidence to PSMs peptides, and optionally, proteins.

    Parameters
    ----------
    max_workers
    psms : list[OnDiskPsmDataset]
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
    deduplication: bool
        Are we performing deduplication on the psm level?
    do_rollup: bool
        do we apply rollup on peptides, modified peptides etc.?
    proteins: Proteins, optional
        collection of proteins
    combine : bool, optional
            Should groups be combined into a single file?
    append_to_output_file: bool
        do we append results to file ?
    sqlite_path: Path to the sqlite database to write mokapot results

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
            scores.append(_psms.read_data(columns=[feat])[feat].values)

    # just take the first one for info (and make sure the other are the same)
    curr_psms = psms[0]
    file_ext = os.path.splitext(curr_psms.filename)[-1]
    for _psms in psms[1:]:
        assert _psms.columns == curr_psms.columns

    # todo: maybe use a collections.namedtuple for all
    # the level info instead of all the ?
    # from collections import namedtuple
    # LevelInfo = namedtuple('LevelInfo',
    #     ['name', 'data_path', 'deduplicate', 'colnames', 'colindices'])

    # Level data for psm level
    level = "psms"
    levels = [level]
    level_data_path = {level: dest_dir / f"{file_root}{level}{file_ext}"}
    level_hash_columns = {level: curr_psms.spectrum_columns}

    # Level data for higher rollup levels
    extra_output_columns = []
    if do_rollup:
        level_columns = curr_psms.level_columns

        for level_column in level_columns:
            level = level_column.lower() + "s"  # e.g. Peptide to peptides
            levels.append(level)
            level_data_path[level] = dest_dir / f"{file_root}{level}{file_ext}"
            level_hash_columns[level] = [level_column]
            if level not in ["psms", "peptides", "proteins"]:
                extra_output_columns.append(level_column)

    levels_or_proteins = levels
    if proteins:
        level = "proteins"
        levels_or_proteins = [*levels, level]
        level_data_path[level] = dest_dir / f"{file_root}{level}{file_ext}"
        level_hash_columns[level] = curr_psms.protein_column

    # fixme: the output header and data do not fit, when the
    #   `extra_output_columns` are in a different place. Fix that.
    out_columns_psms_peps = [
        "PSMId",
        "peptide",
        *extra_output_columns,
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

    for _psms, score, desc, prefix in zip(psms, scores, descs, prefixes):
        out_metadata_columns = [
            "PSMId",
            # fixme: Why is this the only column where we take
            #   the name from the input?
            _psms.target_column,
            "peptide",
            *extra_output_columns,
            "proteinIds",
            "score",
        ]
        in_metadata_columns = [
            _psms.specId_column,
            _psms.target_column,
            _psms.peptide_column,
            *extra_output_columns,
            _psms.protein_column,
            "score",
        ]

        input_output_column_mapping = {
            i: j for i, j in zip(in_metadata_columns, out_metadata_columns)
        }

        file_prefix = file_root
        if prefix:
            file_prefix = f"{file_prefix}{prefix}."

        out_files = {}
        for level in levels_or_proteins:
            if level == "proteins":
                output_columns = out_columns_proteins
            else:
                output_columns = out_columns_psms_peps

            outfile_targets = dest_dir / f"{file_prefix}targets.{level}"
            writer = TabularDataWriter.from_suffix(
                outfile_targets, output_columns
            )
            if not append_to_output_file:
                writer.initialize()
            out_files[level] = [outfile_targets]

            if decoys:
                outfile_decoys = dest_dir / f"{file_prefix}decoys.{level}"
                writer = TabularDataWriter.from_suffix(
                    outfile_decoys, output_columns
                )
                if not append_to_output_file:
                    writer.initialize()
                out_files[level].append(outfile_decoys)

        with create_sorted_file_iterator(
            _psms,
            dest_dir,
            file_prefix,
            do_rollup,
            max_workers,
            score,
        ) as sorted_file_iterator:
            LOGGER.info("Assigning confidence...")
            LOGGER.info("Performing target-decoy competition...")
            LOGGER.info(
                "Keeping the best match per %s columns...",
                "+".join(_psms.spectrum_columns),
            )

            # The columns we get from the sorted file iterator
            iterator_columns = _psms.metadata_columns + ["score"]
            iterator_column_types = _psms.metadata_column_types + [
                get_score_column_type(file_ext)
            ]
            output_column_types = [
                iterator_column_types[iterator_columns.index(i)]
                for i in in_metadata_columns
            ]

            handles = {
                level: TabularDataWriter.from_suffix(
                    level_data_path[level],
                    columns=out_metadata_columns,
                    column_types=output_column_types,
                    buffer_size=0,
                )
                for level in levels
            }
            for writer in handles.values():
                writer.initialize()

            seen_level_entities = {level: set() for level in levels}

            psm_count = 0
            batches = {level: [] for level in levels}
            batch_counts = {level: 0 for level in levels}
            for data_row in sorted_file_iterator:
                psm_count += 1
                for level in levels:
                    if level != "psms" or deduplication:
                        psm_hash = str(
                            [
                                data_row.get(col)
                                for col in level_hash_columns[level]
                            ]
                        )
                        if psm_hash in seen_level_entities[level]:
                            if level == "psms":
                                break
                            continue
                        seen_level_entities[level].add(psm_hash)
                    batches[level].append(data_row)
                    batch_counts[level] += 1
                    if batch_counts[level] == CONFIDENCE_CHUNK_SIZE:
                        df = get_dataframe_from_records(
                            batches[level],
                            in_metadata_columns,
                            input_output_column_mapping,
                            target_column=_psms.target_column,
                        )
                        handles[level].append_data(df)
                        batch_counts[level] = 0
                        batches[level] = []
            for level, batch in batches.items():
                df = get_dataframe_from_records(
                    batch,
                    in_metadata_columns,
                    input_output_column_mapping,
                    target_column=_psms.target_column,
                )
                handles[level].append_data(df)

            for level in levels:
                count = len(seen_level_entities[level])
                handles[level].finalize()
                if level == "psms":
                    if deduplication:
                        LOGGER.info(
                            f"\t- Found {count} PSMs from unique spectra."
                        )
                    else:
                        LOGGER.info(f"\t- Found {psm_count} PSMs.")
                else:
                    LOGGER.info(f"\t- Found {count} unique {level}.")

        LinearConfidence(
            psms=_psms,
            levels=levels_or_proteins,
            level_paths=[
                level_data_path[level] for level in levels_or_proteins
            ],
            out_paths=[out_files[level] for level in levels_or_proteins],
            eval_fdr=eval_fdr,
            desc=desc,
            sep=sep,
            decoys=decoys,
            deduplication=deduplication,
            do_rollup=do_rollup,
            proteins=proteins,
            rng=rng,
            peps_error=peps_error,
            peps_algorithm=peps_algorithm,
            qvalue_algorithm=qvalue_algorithm,
            sqlite_path=sqlite_path,
        )
        if not prefix:
            append_to_output_file = True


@contextmanager
@typechecked
def create_sorted_file_iterator(
    _psms,
    dest_dir: Path,
    file_prefix: str,
    do_rollup: bool,
    max_workers: int,
    score: np.ndarray[float],
):
    # Read from the input psms (PsmDataset) and write into smaller
    # sorted files, by

    # a) Create a reader that only reads columns given in
    #    psms.metadata_columns in chunks of size CONFIDENCE_CHUNK_SIZE
    reader = TabularDataReader.from_path(_psms.filename)
    file_iterator = reader.get_chunked_data_iterator(
        CONFIDENCE_CHUNK_SIZE, _psms.metadata_columns
    )
    outfile_ext = _psms.filename.suffix

    # b) Split the scores in chunks of the same size
    scores_slices = create_chunks(score, chunk_size=CONFIDENCE_CHUNK_SIZE)

    # c) Write those chunks in parallel, where the columns are given
    #    by psms.metadata plus the "scores" column
    #    (NB: after the last change the columns are now indeed in the
    #     order given by metadata_columns and not by file header order)
    Parallel(n_jobs=max_workers, require="sharedmem")(
        delayed(_save_sorted_metadata_chunks)(
            chunk_metadata,
            score_chunk,
            _psms,
            do_rollup,
            dest_dir / f"{file_prefix}scores_metadata_{i}{outfile_ext}",
        )
        for chunk_metadata, score_chunk, i in zip(
            file_iterator, scores_slices, range(len(scores_slices))
        )
    )

    scores_metadata_paths = list(
        dest_dir.glob(f"{file_prefix}scores_metadata_*")
    )
    sorted_file_iterator = merge_sort(
        scores_metadata_paths, score_column="score"
    )

    # Return the sorted iterator and clean up afterwards, regardless of whether
    # an exception was thrown in the `with` block
    try:
        yield sorted_file_iterator
    finally:
        for sc_path in scores_metadata_paths:
            try:
                sc_path.unlink()
            except Exception as e:
                LOGGER.warning(
                    "Caught exception while deleting temp files: %s", e
                )


@typechecked
def _save_sorted_metadata_chunks(
    chunk_metadata: pd.DataFrame,
    score_chunk: np.ndarray[float],
    psms,
    deduplication: bool,
    chunk_write_path: Path,
):
    chunk_metadata = convert_targets_column(
        data=chunk_metadata,
        target_column=psms.target_column,
    )

    chunk_metadata = chunk_metadata.assign(score=score_chunk)
    chunk_metadata.sort_values(by="score", ascending=False, inplace=True)

    if deduplication:
        chunk_metadata = chunk_metadata.drop_duplicates(psms.spectrum_columns)
    chunk_writer = TabularDataWriter.from_suffix(
        chunk_write_path,
        columns=psms.metadata_columns + ["score"],
    )
    chunk_writer.write(chunk_metadata)


@typechecked
def get_unique_peptides_from_psms(
    iterable, peptide_col_name, out_peptides: Path, write_columns: list, sep
):
    f_peptide = open(out_peptides, "a")
    seen_peptide = set()
    for line_list in iterable:
        line_hash_peptide = line_list[peptide_col_name]
        if line_hash_peptide not in seen_peptide:
            seen_peptide.add(line_hash_peptide)
            f_peptide.write(
                f"{sep.join([line_list[key] for key in write_columns])}\n"
            )

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
