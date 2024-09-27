"""One of the primary purposes of mokapot is to assign confidence estimates to
PSMs. This task is accomplished by ranking PSMs according to a score and using
an appropriate confidence estimation procedure for the type of data. mokapot
can provide confidence estimates based any score, regardless of whether it was
the result of a learned :py:func:`~mokapot.model.Model` instance or provided
independently.

The following classes store the confidence estimates for a dataset based on the
provided score. They provide utilities to access, save, and plot these
estimates for the various relevant levels (i.e. PSMs, peptides, and proteins).
The :py:func:`Confidence` class is appropriate for most data-dependent
acquisition proteomics datasets.

We recommend using the :py:func:`~mokapot.brew()` function or the
:py:meth:`~mokapot.PsmDataset.assign_confidence()` method to obtain these
confidence estimates, rather than initializing the classes below directly.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence, Iterator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typeguard import typechecked

from mokapot.column_defs import get_standard_column_name
from mokapot.constants import CONFIDENCE_CHUNK_SIZE
from mokapot.dataset import OnDiskPsmDataset
from mokapot.peps import (
    peps_from_scores,
    TDHistData,
    peps_func_from_hist_nnls,
    PepsConvergenceError,
)
from mokapot.picked_protein import picked_protein
from mokapot.qvalues import qvalues_from_scores, qvalues_func_from_hist
from mokapot.statistics import OnlineStatistics, HistData
from mokapot.tabular_data import (
    BufferType,
    ColumnMappedReader,
    ComputedTabularDataReader,
    ConfidenceSqliteWriter,
    join_readers,
    MergedTabularDataReader,
)
from mokapot.tabular_data import TabularDataReader, TabularDataWriter
from mokapot.tabular_data.target_decoy_writer import TargetDecoyWriter
from mokapot.utils import (
    convert_targets_column,
)

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
@typechecked
class Confidence(object):
    """Estimate the statistical confidence for a collection of PSMs."""

    def __init__(
        self,
        dataset: OnDiskPsmDataset,
        levels: list[str],
        level_paths: dict[str, Path],
        out_writers: dict[str, Sequence[TabularDataWriter]],
        eval_fdr: float = 0.01,
        write_decoys: bool = False,
        do_rollup: bool = True,
        proteins=None,
        peps_error: bool = False,
        rng=0,
        peps_algorithm: str = "qvality",
        qvalue_algorithm: str = "tdc",
        stream_confidence: bool = False,
        score_stats=None,
    ):
        """Initialize a Confidence object.

        Assign confidence estimates to a set of PSMs

        Estimate q-values and posterior error probabilities (PEPs) for PSMs and
        peptides when ranked by the provided scores.

        Parameters
        ----------
        dataset : OnDiskPsmDataset
            An OnDiskPsmDataset.
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
        eval_fdr : float
            The FDR threshold at which to report performance. This parameter
            has no affect on the analysis itself, only logging messages.
        write_decoys : bool
            Save decoys confidence estimates as well?
        """

        self._score_column = "score"
        self._target_column = dataset.target_column
        self._protein_column = "proteinIds"
        self._metadata_column = dataset.metadata_columns
        self._peptide_column = "peptide"

        self._eval_fdr = eval_fdr
        self.do_rollup = do_rollup

        if proteins:
            self.write_protein_level_data(level_paths, proteins, rng)

        self._assign_confidence(
            levels=levels,
            level_path_map=level_paths,
            out_writers_map=out_writers,
            write_decoys=write_decoys,
            peps_error=peps_error,
            peps_algorithm=peps_algorithm,
            qvalue_algorithm=qvalue_algorithm,
            stream_confidence=stream_confidence,
            score_stats=score_stats,
            eval_fdr=eval_fdr,
        )

    def _assign_confidence(
        self,
        levels: list[str],
        level_path_map: dict[str, Path],
        out_writers_map: dict[str, Sequence[TabularDataWriter]],
        write_decoys: bool = False,
        peps_error: bool = False,
        peps_algorithm: str = "qvality",
        qvalue_algorithm: str = "tdc",
        stream_confidence: bool = False,
        score_stats=None,
        eval_fdr: float = 0.01,
    ):
        """
        Assign confidence to PSMs and peptides.

        Parameters
        ----------
        level_path_map : List(Path)
            Files with unique psms and unique peptides.
        levels : List(str)
            the levels at which confidence estimation was performed
        out_paths : List(Path)
            The output files where the results will be written
        write_decoys : bool, optional
            Save decoys confidence estimates as well?
        """
        if stream_confidence:
            if score_stats is None:
                raise ValueError(
                    "score stats must be provided for streamed confidence"
                )

        for level in levels:
            level_path = level_path_map[level]
            out_writers = out_writers_map[level]

            reader = TabularDataReader.from_path(level_path)
            reader = ComputedTabularDataReader(
                reader,
                "is_decoy",
                np.dtype("bool"),
                lambda df: ~df[self._target_column].values,
            )

            writer = TargetDecoyWriter(
                out_writers, write_decoys, decoy_column="is_decoy"
            )

            compute_and_write_confidence(
                reader,
                writer,
                qvalue_algorithm,
                peps_algorithm,
                stream_confidence,
                score_stats,
                peps_error,
                level,
                eval_fdr,
            )
            # todo: discuss: This should probably not be done here, but rather
            #  in the calling code, that intializes the writers
            for writer in out_writers:
                writer.finalize()

            level_path.unlink(missing_ok=True)

    def write_protein_level_data(self, level_paths, proteins, rng):
        psms = TabularDataReader.from_path(level_paths["psms"]).read()
        proteins = picked_protein(
            psms,
            self._target_column,
            self._peptide_column,
            self._score_column,
            proteins,
            rng,
        )
        proteins = proteins.sort_values(
            by=self._score_column, ascending=False
        ).reset_index(drop=True)
        protein_writer = TabularDataWriter.from_suffix(
            file_name=level_paths["proteins"],
            columns=proteins.columns.tolist(),
            column_types=proteins.dtypes.tolist(),
        )
        protein_writer.write(proteins)
        LOGGER.info("\t- Found %i unique protein groups.", len(proteins))


# Functions -------------------------------------------------------------------
@typechecked
def assign_confidence(
    datasets: list[OnDiskPsmDataset],
    scores_list: list[np.ndarray[float]],
    max_workers: int = 1,
    eval_fdr=0.01,
    dest_dir: Path | None = None,
    file_root: str = "",
    prefixes: list[str | None] | None = None,
    write_decoys: bool = False,
    deduplication=True,
    do_rollup=True,
    proteins=None,
    append_to_output_file=False,
    rng=0,
    peps_error=False,
    peps_algorithm="qvality",
    qvalue_algorithm="tdc",
    sqlite_path=None,
    stream_confidence=False,
):
    """Assign confidence to PSMs peptides, and optionally, proteins.

    Parameters
    ----------
    max_workers
    datasets : list[OnDiskPsmDataset]
        A collection of PSMs.
    scores_list : list[numpy.ndarray]
        The scores by which to rank the PSMs.
    rng : int or np.random.Generator, optional
        A seed or generator used for cross-validation split creation and to
        break ties, or ``None`` to use the default random number generator
        state.
    eval_fdr : float
        The FDR threshold at which to report and evaluate performance. If
        `scores` is not :code:`None`, this parameter has no affect on the
        analysis itself, but does affect logging messages and the FDR
        threshold applied for some output formats, such as FlashLFQ.
    dest_dir : Path or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    prefixes : [str]
        The prefixes added to all output file names.
    write_decoys : bool, optional
        Save decoys confidence estimates as well?
    deduplication: bool
        Are we performing deduplication on the psm level?
    do_rollup: bool
        do we apply rollup on peptides, modified peptides etc.?
    proteins: Proteins, optional
        collection of proteins
    append_to_output_file: bool
        do we append results to file ?
    sqlite_path: Path to the sqlite database to write mokapot results

    Returns
    -------
    None
    """
    is_sqlite = sqlite_path is not None

    if dest_dir is None:
        dest_dir = Path()

    # just take the first one for info (and make sure the other are the same)
    curr_dataset = datasets[0]
    file_ext = curr_dataset.reader.get_default_extension()
    for dataset in datasets[1:]:
        assert dataset.columns == curr_dataset.columns

    # Level data for psm level
    level = "psms"
    levels = [level]
    level_data_path = {level: dest_dir / f"{file_root}{level}{file_ext}"}
    level_hash_columns = {level: curr_dataset.spectrum_columns}

    # Level data for higher rollup levels
    extra_output_columns = []
    if do_rollup:
        level_columns = curr_dataset.level_columns

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
        level_hash_columns[level] = curr_dataset.protein_column

    output_column_names = [
        "PSMId",
        "peptide",
        *extra_output_columns,
        "score",
        "q-value",
        "posterior_error_prob",
        "proteinIds",
    ]

    output_column_names_proteins = [
        "mokapot protein group",
        "best peptide",
        "stripped sequence",
        "score",
        "q-value",
        "posterior_error_prob",
    ]

    @typechecked
    def create_output_writer(path: Path, level: str, initialize: bool):
        if level == "proteins":
            output_columns = output_column_names_proteins
        else:
            output_columns = output_column_names

        # Create the writers
        if is_sqlite:
            writer = ConfidenceSqliteWriter(
                sqlite_path,
                columns=output_columns,
                column_types=[],
                level=level,
                qvalue_column="q-value",
                pep_column="posterior_error_prob",
            )
        else:
            writer = TabularDataWriter.from_suffix(path, output_columns, [])

        if initialize:
            writer.initialize()
        return writer

    for dataset, score, prefix in zip(datasets, scores_list, prefixes):
        # todo: nice to have: move this column renaming stuff into the
        #   column defs module, and further, have standardized columns
        #   directly from the pin reader (applying the renaming itself)
        level_column_names = [
            "PSMId",
            dataset.target_column,
            "peptide",
            *extra_output_columns,
            "proteinIds",
            "score",
        ]
        level_input_column_names = [
            dataset.specId_column,
            dataset.target_column,
            dataset.peptide_column,
            *extra_output_columns,
            dataset.protein_column,
            "score",
        ]

        level_input_output_column_mapping = {
            in_col: out_col
            for in_col, out_col in zip(
                level_input_column_names, level_column_names
            )
        }

        file_prefix = file_root
        if prefix:
            file_prefix = f"{file_prefix}{prefix}."

        output_writers = {}
        for level in levels_or_proteins:
            output_writers[level] = []

            outfile_targets = (
                dest_dir / f"{file_prefix}targets.{level}{file_ext}"
            )

            output_writers[level].append(
                create_output_writer(
                    outfile_targets, level, not append_to_output_file
                )
            )

            if write_decoys and not is_sqlite:
                outfile_decoys = (
                    dest_dir / f"{file_prefix}decoys.{level}{file_ext}"
                )
                output_writers[level].append(
                    create_output_writer(
                        outfile_decoys, level, not append_to_output_file
                    )
                )

        score_reader = TabularDataReader.from_array(score, "score")
        with create_sorted_file_reader(
            dataset,
            score_reader,
            dest_dir,
            file_prefix,
            level_hash_columns["psms"] if deduplication else None,
            max_workers,
            level_input_output_column_mapping,
        ) as sorted_file_reader:
            LOGGER.info("Assigning confidence...")
            LOGGER.info("Performing target-decoy competition...")
            LOGGER.info(
                "Keeping the best match per %s columns...",
                "+".join(dataset.spectrum_columns),
            )

            # The columns we get from the sorted file iterator
            sorted_file_iterator = sorted_file_reader.get_row_iterator(
                row_type=BufferType.Dicts
            )
            type_map = sorted_file_reader.get_schema(as_dict=True)
            level_column_types = [
                type_map[name] for name in level_column_names
            ]

            level_writers = {
                level: TabularDataWriter.from_suffix(
                    level_data_path[level],
                    columns=level_column_names,
                    column_types=level_column_types,
                    buffer_size=CONFIDENCE_CHUNK_SIZE,
                    buffer_type=BufferType.Dicts,
                )
                for level in levels
            }
            for writer in level_writers.values():
                writer.initialize()

            def hash_data_row(data_row):
                return str([
                    data_row[level_input_output_column_mapping.get(col, col)]
                    for col in level_hash_columns[level]
                ])

            seen_level_entities = {level: set() for level in levels}
            score_stats = OnlineStatistics()
            psm_count = 0
            for data_row in sorted_file_iterator:
                psm_count += 1
                for level in levels:
                    if level != "psms" or deduplication:
                        psm_hash = hash_data_row(data_row)
                        if psm_hash in seen_level_entities[level]:
                            if level == "psms":
                                # If we are on the psms level, we can skip
                                # checking the other levels
                                break
                            continue
                        seen_level_entities[level].add(psm_hash)
                    out_row = {
                        col: data_row[col] for col in level_column_names
                    }
                    level_writers[level].append_data(out_row)
                    score_stats.update_single(data_row["score"])

            for level in levels:
                count = len(seen_level_entities[level])
                level_writers[level].finalize()
                if level == "psms":
                    if deduplication:
                        LOGGER.info(
                            f"\t- Found {count} PSMs from unique spectra."
                        )
                    else:
                        LOGGER.info(f"\t- Found {psm_count} PSMs.")
                    LOGGER.info(
                        f"\t- The average score was {score_stats.mean:.3f} "
                        f"with standard deviation {score_stats.sd:.3f}."
                    )
                else:
                    LOGGER.info(f"\t- Found {count} unique {level}.")

        Confidence(
            dataset=dataset,
            levels=levels_or_proteins,
            level_paths=level_data_path,
            out_writers=output_writers,
            eval_fdr=eval_fdr,
            write_decoys=write_decoys,
            do_rollup=do_rollup,
            proteins=proteins,
            rng=rng,
            peps_error=peps_error,
            peps_algorithm=peps_algorithm,
            qvalue_algorithm=qvalue_algorithm,
            stream_confidence=stream_confidence,
            score_stats=score_stats,
        )
        if not prefix:
            append_to_output_file = True


@contextmanager
@typechecked
def create_sorted_file_reader(
    dataset: OnDiskPsmDataset,
    score_reader: TabularDataReader,
    dest_dir: Path,
    file_prefix: str,
    deduplication_columns: list[str] | None,
    max_workers: int,
    input_output_column_mapping,
):
    """Read from the input psms and write into smaller sorted files by score"""

    # Create a reader that only reads columns given in psms.metadata_columns
    # in chunks of size CONFIDENCE_CHUNK_SIZE and joins the scores to it
    reader = join_readers([
        ColumnMappedReader(dataset.reader, input_output_column_mapping),
        score_reader,
    ])
    input_columns = dataset.metadata_columns + ["score"]
    output_columns = [
        input_output_column_mapping.get(name, name) for name in input_columns
    ]
    file_iterator = reader.get_chunked_data_iterator(
        CONFIDENCE_CHUNK_SIZE, output_columns
    )

    #  Write those chunks in parallel, where the columns are given
    #  by dataset.metadata plus the "scores" column

    outfile_ext = dataset.reader.file_name.suffix
    scores_metadata_paths = Parallel(n_jobs=max_workers, require="sharedmem")(
        delayed(_save_sorted_metadata_chunks)(
            chunk_metadata,
            dest_dir / f"{file_prefix}scores_metadata_{i}{outfile_ext}",
            output_columns,
            dataset.target_column,
            deduplication_columns,
        )
        for i, chunk_metadata in enumerate(file_iterator)
    )

    readers = [
        TabularDataReader.from_path(path) for path in scores_metadata_paths
    ]

    sorted_file_reader = MergedTabularDataReader(
        readers,
        priority_column="score",
        reader_chunk_size=CONFIDENCE_CHUNK_SIZE,
    )

    # Return the sorted iterator and clean up afterwards, regardless of whether
    # an exception was thrown in the `with` block
    try:
        yield sorted_file_reader
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
    chunk_write_path: Path,
    output_columns,
    target_column,
    deduplication_columns,
):
    chunk_metadata = convert_targets_column(
        data=chunk_metadata,
        target_column=target_column,
    )
    chunk_metadata.sort_values(by="score", ascending=False, inplace=True)

    if deduplication_columns is not None:
        # This is not strictly necessary, as we deduplicate also afterwards,
        # but speeds up the process
        chunk_metadata.drop_duplicates(deduplication_columns, inplace=True)

    chunk_writer = TabularDataWriter.from_suffix(
        chunk_write_path,
        columns=output_columns,
        column_types=[],
    )
    chunk_writer.write(chunk_metadata)
    return chunk_write_path


@typechecked
def compute_and_write_confidence(
    temp_reader: TabularDataReader,
    writer: TabularDataWriter,
    qvalue_algorithm: str,
    peps_algorithm: str,
    stream_confidence: bool,
    score_stats: OnlineStatistics,
    peps_error: bool,
    level: str,
    eval_fdr: float,
):
    qvals_column = get_standard_column_name("q-value")
    peps_column = get_standard_column_name("posterior_error_prob")

    if not stream_confidence:
        # Read all data at once, compute the peps and qvalues and write in one
        # large chunk
        data = temp_reader.read()
        scores = data["score"].to_numpy()
        targets = ~data["is_decoy"].to_numpy()
        if all(targets):
            LOGGER.warning(
                "No decoy PSMs remain for confidence estimation. "
                "Confidence estimates may be unreliable."
            )

        # Estimate q-values and assign
        LOGGER.info(
            f"Assigning q-values to {level} "
            f"(using {qvalue_algorithm} algorithm) ..."
        )
        qvals = qvalues_from_scores(scores, targets, qvalue_algorithm)
        data[qvals_column] = qvals

        # Logging update on q-values
        num_found = (qvals[targets] <= eval_fdr).sum()
        LOGGER.info(f"\t- Found {num_found} {level} with q<={eval_fdr}")

        # Calculate PEPs
        LOGGER.info(
            "Assigning PEPs to %s (using %s algorithm) ...",
            level,
            peps_algorithm,
        )
        try:
            peps = peps_from_scores(
                scores, targets, is_tdc=True, pep_algorithm=peps_algorithm
            )
        except PepsConvergenceError:
            LOGGER.info(
                f"\t- Encountered convergence problems in `{peps_algorithm}`. "
                "Falling back to qvality ...",
            )
            peps = peps_from_scores(
                scores, targets, is_tdc=True, pep_algorithm="qvality"
            )

        if peps_error and all(peps == 1):
            raise ValueError("PEP values are all equal to 1.")
        data[peps_column] = peps

        writer.append_data(data)

    else:  # Here comes the streaming part
        LOGGER.info("Computing statistics for q-value and PEP assignment...")
        bin_edges = HistData.get_bin_edges(score_stats, clip=(50, 500))
        score_target_iterator = create_score_target_iterator(
            temp_reader.get_chunked_data_iterator(
                chunk_size=CONFIDENCE_CHUNK_SIZE, columns=["score", "is_decoy"]
            )
        )
        hist_data = TDHistData.from_score_target_iterator(
            bin_edges, score_target_iterator
        )
        if hist_data.decoys.counts.sum() == 0:
            LOGGER.warning(
                "No decoy PSMs remain for confidence estimation. "
                "Confidence estimates may be unreliable."
            )

        LOGGER.info("Estimating q-value and PEP assignment functions...")
        qvalues_func = qvalues_func_from_hist(hist_data, is_tdc=True)
        peps_func = peps_func_from_hist_nnls(hist_data, is_tdc=True)

        LOGGER.info("Streaming q-value and PEP assignments...")
        for df_chunk in temp_reader.get_chunked_data_iterator(
            chunk_size=CONFIDENCE_CHUNK_SIZE
        ):
            scores = df_chunk["score"].values
            df_chunk[qvals_column] = qvalues_func(scores)
            df_chunk[peps_column] = peps_func(scores)

            writer.append_data(df_chunk)


@typechecked
def create_score_target_iterator(chunked_iterator: Iterator):
    for df_chunk in chunked_iterator:
        scores = df_chunk["score"].values
        targets = ~df_chunk["is_decoy"].values
        yield scores, targets
