"""One of the primary purposes of mokapot is to assign confidence estimates to
PSMs. This task is accomplished by ranking PSMs according to a score and using
an appropriate confidence estimation procedure for the type of data. mokapot
can provide confidence estimates based any score, regardless of whether it was
the result of a learned :py:func:`~mokapot.model.Model` instance or provided
independently.

The following classes store the confidence estimates for a dataset based on the
provided score. They provide utilities to access, save, and plot these
estimates for the various relevant levels (i.e. PSMs, peptides, and proteins).
The :py:class:`Confidence` class is the primary interface for handling
confidence estimates in data-dependent acquisition proteomics datasets.

The module provides:
- Confidence estimation at PSM, peptide, and protein levels
- Streaming processing for large datasets
- Visualization utilities for confidence estimates
- Export capabilities in various formats

We recommend using the :py:func:`~mokapot.brew()` followed by the
`assign_confidence` function to obtain these confidence estimates,
rather than initializing the classes below directly.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typeguard import typechecked

from mokapot.column_defs import Q_VALUE_COL_NAME, get_standard_column_name
from mokapot.constants import CONFIDENCE_CHUNK_SIZE
from mokapot.dataset import PsmDataset
from mokapot.peps import (
    PepsConvergenceError,
    TDHistData,
    peps_from_scores,
    peps_func_from_hist_nnls,
)
from mokapot.picked_protein import picked_protein
from mokapot.proteins import Proteins
from mokapot.qvalues import qvalues_from_scores, qvalues_func_from_hist
from mokapot.statistics import HistData, OnlineStatistics
from mokapot.tabular_data import (
    BufferType,
    ColumnMappedReader,
    ComputedTabularDataReader,
    ConfidenceSqliteWriter,
    MergedTabularDataReader,
    TabularDataReader,
    TabularDataWriter,
    join_readers,
)
from mokapot.tabular_data.target_decoy_writer import TargetDecoyWriter
from mokapot.utils import (
    make_bool_trarget,
    strictzip,
)
from mokapot.writers.flashlfq import to_flashlfq

LOGGER = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except ImportError:
    LOGGER.warning(
        "Matplotlib is not installed. Confidence plots will not be available."
    )
    plt = None


# Classes ---------------------------------------------------------------------
@typechecked
class Confidence:
    """Calculate, Store and provide access to confidence estimates.

    This class stores confidence estimates (q-values and PEPs) computed for
    different levels (PSMs, peptides, proteins) and provides methods
    to access, save and visualize these estimates.

    Parameters
    ----------
    dataset : PsmDataset
        The dataset containing PSMs and metadata
    levels : list[str]
        Levels at which confidence estimation was performed
        (e.g. ['psms','peptides'])
    level_paths : dict[str, Path]
        Paths to intermediate files for each confidence level
    out_writers : dict[str, Sequence[TabularDataWriter]]
        Writers for outputting results at each level
    eval_fdr : float, optional
        FDR threshold for evaluation metrics, by default 0.01
    write_decoys : bool, optional
        Whether to write decoy results, by default False
    do_rollup : bool, optional
        Whether to perform protein inference, by default True
    proteins : pd.DataFrame, optional
        Protein annotations if protein inference is used
    peps_error : bool, optional
        Whether to raise error on PEP calculation failure, by default False
    rng : int or np.random.Generator, optional
        Random number generator for reproducibility
    peps_algorithm : str, optional
        Algorithm for PEP calculation, by default "qvality"
    qvalue_algorithm : str, optional
        Algorithm for q-value calculation, by default "tdc"
    stream_confidence : bool, optional
        Whether to stream confidence calculations, by default False
    score_stats : OnlineStatistics, optional
        Pre-computed score statistics if streaming
    """

    def __init__(
        self,
        dataset: PsmDataset,
        levels: list[str],
        level_paths: dict[str, Path],
        out_writers: dict[str, Sequence[TabularDataWriter]],
        eval_fdr: float = 0.01,
        write_decoys: bool = False,
        do_rollup: bool = True,
        proteins: Proteins | None = None,
        peps_error: bool = False,
        rng: int | np.random.Generator = 0,
        peps_algorithm: str = "qvality",
        qvalue_algorithm: str = "tdc",
        stream_confidence: bool = False,
        score_stats: OnlineStatistics | None = None,
    ):
        # As far as I can tell, the dataset is only used to export to flashlfq
        self.dataset = dataset
        self._score_column = "score"
        self._target_column = dataset.target_column
        self._peptide_column = "peptide"

        self.eval_fdr = eval_fdr
        self.level_paths = level_paths
        self.out_writers = out_writers
        self.write_decoys = write_decoys
        self.levels = levels
        self.do_rollup = do_rollup
        self._proteins = proteins
        self.peps_error = peps_error
        self.rng = rng
        self.score_stats = score_stats

        if proteins:
            self._write_protein_level_data(level_paths, proteins, rng)

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

    def __repr__(self) -> str:
        ds_sec = self.dataset.__repr__()
        ds_sec = "".join([
            "\t" + x.strip("\n") + "\n" for x in ds_sec.split("\n")
        ])
        rep = "Confidence object\n"
        rep += f"Dataset: \n{ds_sec}"
        rep += f"Levels: {self.levels}\n"
        rep += f"Level paths: {self.level_paths}\n"
        rep += f"Out writers: {self.out_writers}\n"
        rep += f"Eval FDR: {self.eval_fdr}\n"
        rep += f"Write decoys: {self.write_decoys}\n"
        rep += f"Do rollup: {self.do_rollup}\n"
        rep += f"Peps error: {self.peps_error}\n"
        rep += f"Rng: {self.rng}\n"
        rep += f"Score stats: {self.score_stats}\n"
        return rep

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
        score_stats: OnlineStatistics | None = None,
        eval_fdr: float = 0.01,
    ):
        """Assign confidence estimates to PSMs and peptides.

        This method processes the dataset to assign confidence estimates
        (q-values and PEPs) at different levels (PSMs, peptides) using the
        specified algorithms. It can optionally stream the confidence
        calculations for large datasets.

        Parameters
        ----------
        levels : list[str]
            The levels at which to compute confidence estimates
            (e.g., ['psms', 'peptides']).
        level_path_map : dict[str, Path]
            Mapping of confidence levels to their corresponding file paths
            for intermediate data.
        out_writers_map : dict[str, Sequence[TabularDataWriter]]
            Mapping of confidence levels to their output writers for results.
        write_decoys : bool, optional
            Whether to include decoy results in the output, by default False.
        peps_error : bool, optional
            Whether to raise an error if PEP calculation fails
            by default False.
        peps_algorithm : str, optional
            Algorithm to use for posterior error probability calculation.
            Currently supports "qvality" (default).
        qvalue_algorithm : str, optional
            Algorithm to use for q-value calculation.
            Currently supports "tdc" (default).
        stream_confidence : bool, optional
            Whether to stream confidence calculations for large datasets
            by default False.
        score_stats : OnlineStatistics | None, optional
            Pre-computed score statistics if streaming is enabled
            by default None.
        eval_fdr : float, optional
            FDR threshold for evaluation metrics, by default 0.01.
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

    def _write_protein_level_data(
        self,
        level_paths: dict[str, Path],
        proteins: Proteins,
        rng: int | np.random.Generator,
    ):
        psms = TabularDataReader.from_path(level_paths["psms"]).read()
        proteins_df = picked_protein(
            peptides=psms,
            target_column=self._target_column,
            peptide_column=self._peptide_column,
            score_column=self._score_column,
            proteins=proteins,
            rng=rng,
        )
        proteins_df = proteins_df.sort_values(
            by=self._score_column, ascending=False
        ).reset_index(drop=True)
        protein_writer = TabularDataWriter.from_suffix(
            file_name=level_paths["proteins"],
            columns=proteins_df.columns.tolist(),
            column_types=proteins_df.dtypes.tolist(),
        )
        protein_writer.write(proteins_df)
        LOGGER.info("\t- Found %i unique protein groups.", len(proteins_df))

    def read(self, level: str) -> pd.DataFrame:
        """Read the results for a given level."""
        if level not in self.levels:
            raise ValueError(
                f"Level {level} not found. Available levels are: {self.levels}"
            )
        tmp = [x.read() for x in self.out_writers[level]]
        return pd.concat(tmp)

    @property
    def peptides(self) -> pd.DataFrame:
        return self.read("peptides")

    @property
    def psms(self) -> pd.DataFrame:
        return self.read("psms")

    @property
    def proteins(self) -> pd.DataFrame:
        return self.read("proteins")

    def to_flashlfq(self, out_file="mokapot.flashlfq.txt"):
        """Save confidenct peptides for quantification with FlashLFQ."""
        return to_flashlfq(self, out_file)

    def plot_qvalues(
        self, level: str, threshold: float = 0.1, ax=None, **kwargs
    ):
        """Plot the q-values for a given level.

        Parameters
        ----------
        level : str
            The level to plot.
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
            A `matplotlib.axes.Axes` with the cumulative
            number of accepted target PSMs or peptides.
        """

        all_read = [x.read() for x in self.out_writers[level]]
        qvals = pd.concat(all_read)[Q_VALUE_COL_NAME]
        return plot_qvalues(qvals, threshold=threshold, ax=ax, **kwargs)


# Functions -------------------------------------------------------------------
@typechecked
def assign_confidence(
    datasets: list[PsmDataset],
    scores_list: list[np.ndarray[float]] | None = None,
    max_workers: int = 1,
    eval_fdr: float = 0.01,
    dest_dir: Path | None = None,
    file_root: str = "",
    prefixes: list[str | None] | None = None,
    write_decoys: bool = False,
    deduplication: bool = True,
    do_rollup: bool = True,
    proteins: Proteins | None = None,
    protein_column: str | None = None,
    append_to_output_file: bool = False,
    rng: int | np.random.Generator = 0,
    peps_error: bool = False,
    peps_algorithm="qvality",
    qvalue_algorithm="tdc",
    sqlite_path: Path | None = None,
    stream_confidence: bool = False,
):
    """Assign confidence to PSMs, peptides, and optionally proteins.

    Parameters
    ----------
    datasets : list[PsmDataset]
        A collection of PSMs.
    scores_list : list[numpy.ndarray[float]] | None, optional
        The scores by which to rank the PSMs. Usually derived from
        `mokapot.brew`, by default None.
    max_workers : int, optional
        Number of parallel workers to use for processing, by default 1.
    eval_fdr : float, optional
        The FDR threshold at which to report and evaluate performance,
        by default 0.01.
    dest_dir : Path | None, optional
        The directory in which to save the files. None will use the
        current working directory, by default None.
    file_root : str, optional
        Base name prefix for output files, by default "".
    prefixes : list[str | None] | None, optional
        The prefixes added to all output file names.
        If None, a single concatenated file will be created, by default None.
    write_decoys : bool, optional
        Save decoy confidence estimates as well?, by default False.
    deduplication : bool, optional
        Whether to perform deduplication on the PSM level, by default True.
    do_rollup : bool, optional
        Whether to apply rollup on peptides, modified peptides etc.
        by default True.
    proteins : Proteins | None, optional
        Collection of proteins for protein inference, by default None.
    append_to_output_file : bool, optional
        Whether to append results to existing file, by default False.
    rng : int | np.random.Generator, optional
        Random number generator or seed for reproducibility, by default 0.
    peps_error : bool, optional
        Whether to raise error on PEP calculation failure, by default False.
    peps_algorithm : {'qvality', 'qvality_bin', 'kde_nnls', 'hist_nnls'}
        Algorithm for posterior error probability calculation
        by default "qvality".
    qvalue_algorithm : {'tdc', 'hist'}, optional
        Algorithm for q-value calculation, by default "tdc".
    sqlite_path : Path | None, optional
        Path to the SQLite database to write mokapot results, by default None.
    stream_confidence : bool, optional
        Whether to stream confidence calculations for large datasets
        by default False.

    Returns
    -------
    list[Confidence]
        A list of Confidence objects containing the confidence estimates for
        each dataset.
    """

    # Note: I think we can do better on this API ... it feels wrong that
    # half of the arguments dont matter anymore if an sqlite path is passed.
    # At least we should check for mutually exclusive sets of arguments.
    # JSPP 2024-12-05

    # _is_sqlite = sqlite_path is not None

    if dest_dir is None:
        dest_dir = Path()

    # just take the first one for info (and make sure the other are the same)
    curr_dataset = datasets[0]
    for di, dataset in enumerate(datasets[1:]):
        if dataset.columns != curr_dataset.columns:
            raise ValueError(
                "Datasets must have the same columns. "
                f"Dataset 1 has columns {curr_dataset.columns} "
                f"and dataset {di + 2} has columns {dataset.columns}"
            )

    if protein_column is None:
        protein_column = curr_dataset.column_groups.optional_columns.protein

    level_manager = LevelManager.from_dataset(
        dataset=curr_dataset,
        do_rollup=do_rollup,
        dest_dir=dest_dir,
        file_root=file_root,
        protein_column=protein_column,
        disable_proteins=proteins is None,
    )

    output_writers_factory = OutputWriterFactory(
        id_column=curr_dataset.column_groups.optional_columns.id,
        spectra_columns=curr_dataset.spectrum_columns,
        extra_output_columns=level_manager.extra_output_columns,
        sqlite_path=sqlite_path,
        append_to_output_file=append_to_output_file,
        write_decoys=write_decoys,
    )

    if prefixes is None:
        prefixes = [None] * len(datasets)

    level_input_output_column_mapping = level_manager.build_output_col_mapping(
        curr_dataset
    )

    scores_use = scores_list
    if scores_use is None:
        LOGGER.info("No scores passed, attempting to find them.")
        if any(dataset.scores is None for dataset in datasets):
            LOGGER.info("No scores found, attempting to find best feature.")
            feature = datasets[0].find_best_feature(eval_fdr).feature
            LOGGER.info("Best feature found: %s", feature)
            scores_use = [
                dataset.read_data(columns=[feature.name])[
                    feature.name
                ].to_numpy()
                for dataset in datasets
            ]
            # TODO: warn that no scores are present and will fall back
        else:
            LOGGER.info("Scores found in psms, using them.")
            scores_use = [dataset.scores for dataset in datasets]

    out = []

    for dataset, score, prefix in strictzip(datasets, scores_use, prefixes):
        # todo: nice to have: move this column renaming stuff into the
        #   column defs module, and further, have standardized columns
        #   directly from the pin reader (applying the renaming itself)

        output_writers, file_prefix = output_writers_factory.build_writers(
            level_manager,
            prefix=prefix,
        )

        score_reader = TabularDataReader.from_array(score, "score")
        with create_sorted_file_reader(
            dataset=dataset,
            score_reader=score_reader,
            dest_dir=dest_dir,
            file_prefix=file_prefix,
            deduplication_columns=(
                level_manager.level_hash_columns["psms"]
                if deduplication
                else None
            ),
            max_workers=max_workers,
            input_output_column_mapping=level_input_output_column_mapping,
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
            level_writers = LevelWriterCollection.from_manager(
                level_manager=level_manager,
                type_map=type_map,
                level_input_output_column_mapping=level_input_output_column_mapping,
                deduplication=deduplication,
            )

            level_writers.sink_iterator(sorted_file_iterator)
            level_writers.finalize()

        con = Confidence(
            dataset=dataset,
            levels=level_manager.levels_or_proteins,
            level_paths=level_manager.level_data_paths,
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
            score_stats=level_writers.score_stats,
        )
        out.append(con)
        if not prefix:
            # Having None as a prefix means that all outputs will be
            # written to a single file, thus after the first iteration
            # we stop initializing the writers (bc that generates over-writing
            # the files instead of appending to them).
            output_writers_factory.append_to_output_file = True

    return out


class LevelWriterCollection:
    def __init__(
        self,
        levels: list[str],
        level_data_paths: dict[str, Path],
        schema_dict: dict[str, np.dtype],
        level_input_output_column_mapping: dict[str, str],
        level_hash_columns: dict[str, list[str]],
        deduplication: bool,
    ):
        # Do I need to pass the levels? cant I use the keys of the data paths?
        self.levels = levels
        self.deduplication = deduplication
        self.level_input_output_column_mapping = (
            level_input_output_column_mapping
        )
        self.level_hash_columns = level_hash_columns
        level_column_types = [
            schema_dict[name]
            for name in level_input_output_column_mapping.values()
        ]

        self.level_writers = {
            level: TabularDataWriter.from_suffix(
                level_data_paths[level],
                columns=list(level_input_output_column_mapping.values()),
                column_types=level_column_types,
                buffer_size=CONFIDENCE_CHUNK_SIZE,
                buffer_type=BufferType.Dicts,
            )
            for level in levels
        }
        self.seen_level_entities = {level: set() for level in levels}
        for level, writer in self.level_writers.items():
            LOGGER.info(f"Initializing writer for level {level}")
            LOGGER.debug(f"\t {writer}")
            writer.initialize()

        self.score_stats = OnlineStatistics()
        self.per_level_counts = defaultdict(int)

    def __repr__(self):
        pretty_dict = pformat(self.__dict__)
        return f"{self.__class__!s}({pretty_dict})"

    @staticmethod
    def from_manager(
        level_manager: LevelManager,
        type_map: dict[str, np.dtype],
        level_input_output_column_mapping: dict[str, str],
        deduplication: bool,
    ) -> LevelWriterCollection:
        level_data_paths = level_manager.level_data_paths
        levels = level_manager.levels
        hash_columns = level_manager.level_hash_columns
        return LevelWriterCollection(
            levels=levels,
            level_data_paths=level_data_paths,
            schema_dict=type_map,
            level_input_output_column_mapping=level_input_output_column_mapping,
            level_hash_columns=hash_columns,
            deduplication=deduplication,
        )

    def hash_data_row(self, data_row, level):
        # TODO: benchmark if actually hashing here would be better.
        # .     It feels inefficient to keep large numbers of large strings
        #       in memory.
        return str([
            data_row[self.level_input_output_column_mapping.get(col, col)]
            for col in self.level_hash_columns[level]
        ])

    def sink_iterator(self, sorted_file_iterator):
        for data_row in sorted_file_iterator:
            self.per_level_counts["total"] += 1
            for level in self.levels:
                if level != "psms" or self.deduplication:
                    psm_hash = self.hash_data_row(data_row, level=level)
                    if psm_hash in self.seen_level_entities[level]:
                        continue
                    self.per_level_counts[level] += 1
                    self.seen_level_entities[level].add(psm_hash)
                out_row = {
                    col: data_row[col]
                    for col in self.level_input_output_column_mapping.values()
                }
                self.level_writers[level].append_data(out_row)
                self.score_stats.update_single(data_row["score"])

    def finalize(self):
        for level in self.levels:
            count = len(self.seen_level_entities[level])
            curr_writer = self.level_writers[level]
            LOGGER.debug(f"Finalizing writer for level {level}: {curr_writer}")
            curr_writer.finalize()
            if level == "psms":
                if self.deduplication:
                    LOGGER.info(f"\t- Found {count} PSMs from unique spectra.")
                else:
                    psm_count = self.per_level_counts["psms"]
                    LOGGER.info(f"\t- Found {psm_count} PSMs.")
                LOGGER.info(
                    f"\t- The average score was {self.score_stats.mean:.3f} "
                    f"with standard deviation {self.score_stats.sd:.3f}."
                )
            else:
                LOGGER.info(f"\t- Found {count} unique {level}.")


class LevelManager:
    """Manages level-specific data and operations.

    This class is meant to be used internally by the `Confidence` class.
    But it basically bundles the output path creation logic and the column
    +level mapping logic.

    Parameters
    ----------
    level_columns : list of str
        The columns that can be used to aggregate PSMs.
        For example, peptides, modified peptides, precursors.
        would generate "rollups" of the PSMs at the PSM (default)
        and in addition to that, the peptide and modified peptide
        columns would generate "peptide groups" of PSMs (each).
    default_extension : str
        The default extension to use for the output files.
        The extension will be used to determine the output format
        when initializing the `LevelWriterCollection` which internally
        uses the `TabularDataWriter.from_suffix` method.
    spectrum_columns : list of str
        The columns that uniquely identify a mass spectrum.
    do_rollup : bool
        Do we apply rollup on peptides, modified peptides etc.?
    use_proteins : bool
        Whether to roll up protein-level confidence estimates.
    dest_dir : Path
        The directory in which to save the files.
    file_root : str
        The prefix added to all output file names.
        The final file names will be:
        `dest_dir / file_root+level+default_extension`
    protein_column : str, optional
        The column that specifies which protein(s) the detected peptide might
        have originated from. This column is not used to compute protein-level
        confidence estimates, it is just propagated to the output.
    """

    def __init__(
        self,
        *,
        level_columns: list[str] | tuple[str, ...],
        default_extension: str,
        spectrum_columns: list[str] | tuple[str, ...],
        do_rollup: bool,
        dest_dir: Path,
        file_root: str,
        protein_column: str | None,
        use_proteins: bool,
        id_column: str | None,
    ):
        self.level_columns = level_columns
        self.default_extension = default_extension
        self.spectrum_columns = spectrum_columns
        self.id_column = id_column
        self.use_proteins = (protein_column is not None) and use_proteins
        self.dest_dir = dest_dir
        self.file_root = file_root
        self.do_rollup = do_rollup
        self.protein_column = protein_column

        self.levels = self._initialize_levels()
        self.level_data_paths = self._setup_level_paths()
        self.level_hash_columns = self._setup_hash_columns()
        self._setup_protein_levels()
        self._setup_extra_output_columns()

    @staticmethod
    def from_dataset(
        *,
        dataset: PsmDataset,
        do_rollup: bool,
        dest_dir: Path,
        file_root: str,
        protein_column: str | None,
        disable_proteins: bool = False,
    ):
        level_columns = dataset.confidence_level_columns
        default_extension = dataset.get_default_extension()
        spectrum_columns = dataset.spectrum_columns
        id_col = dataset.column_groups.optional_columns.id
        return LevelManager(
            level_columns=level_columns,
            default_extension=default_extension,
            spectrum_columns=spectrum_columns,
            do_rollup=do_rollup,
            protein_column=protein_column,
            dest_dir=dest_dir,
            file_root=file_root,
            use_proteins=not disable_proteins,
            id_column=id_col,
        )

    def __repr__(self) -> str:
        formatted_dict = pformat(self.__dict__)
        return f"{self.__class__!s}({formatted_dict})"

    def _initialize_levels(self) -> list[str]:
        """Initialize processing levels based on configuration."""
        levels = ["psms"]
        if self.do_rollup:
            level_columns = self.level_columns
            levels.extend(col.lower() + "s" for col in level_columns)

        return levels

    def _setup_level_paths(
        self,
    ) -> dict[str, Path]:
        """Setup paths for each processing level."""
        level_data_paths = {}
        file_ext = self.default_extension
        for level in self.levels:
            level_data_paths[level] = (
                self.dest_dir / f"{self.file_root}{level}{file_ext}"
            )

        return level_data_paths

    def _setup_hash_columns(self) -> dict[str, list[str]]:
        """Setup hash columns for each level."""
        # Note: we do not use the id col as part of the hash columns, since by
        #       definition it will never be duplicated.
        # base_cols = [self.id_column] if self.id_column is not None else []
        # base_cols += self.spectrum_columns

        base_cols = list(self.spectrum_columns)

        level_hash_columns = {"psms": base_cols}
        for level in self.levels[1:]:
            if level != "proteins":
                level_hash_columns[level] = [level.rstrip("s").capitalize()]

        return level_hash_columns

    def _setup_protein_levels(self) -> None:
        levels_or_proteins = self.levels
        file_ext = self.default_extension
        if self.use_proteins:
            levels_or_proteins = [*levels_or_proteins, "proteins"]
            self.level_data_paths["proteins"] = (
                self.dest_dir / f"{self.file_root}proteins{file_ext}"
            )
            self.level_hash_columns["proteins"] = self.protein_column

        self.levels_or_proteins = levels_or_proteins

    def _setup_extra_output_columns(self) -> None:
        extra_output_columns = []
        if self.do_rollup:
            level_columns = self.level_columns

            for level_column in level_columns:
                level = level_column.lower() + "s"  # e.g. Peptide to peptides
                if level not in self.levels:
                    self.levels.append(level)

                self.level_data_paths[level] = (
                    self.dest_dir
                    / f"{self.file_root}{level}{self.default_extension}"
                )

                # I am not sure why but over-writing some of the levels here is
                # important, I think it has to do with with how the rollup
                # levels are handled (columns are renamed).
                self.level_hash_columns[level] = [level_column]
                if level not in ["psms", "peptides", "proteins"]:
                    extra_output_columns.append(level_column)

        self.extra_output_columns = extra_output_columns

    def build_output_col_mapping(self, dataset: PsmDataset) -> dict:
        # Note: we could in theory add here the checks for the output
        # column names, so constraints like it being sql-safe or
        # something like that could be enforced in the future.

        if self.protein_column is not None:
            if self.protein_column not in dataset.get_column_names():
                raise ValueError(
                    f"Protein column {self.protein_column}"
                    " not found in dataset"
                )

        id_col = dataset.column_groups.optional_columns.id
        id_col = [id_col] if id_col is not None else []

        level_column_names = [
            *id_col,
            *dataset.spectrum_columns,
            dataset.target_column,
            "peptide",
            *self.extra_output_columns,
            "proteinIds",
            "score",
        ]
        level_input_column_names = [
            *id_col,
            *dataset.spectrum_columns,
            dataset.target_column,
            dataset.peptide_column,
            *self.extra_output_columns,
            self.protein_column,
            "score",
        ]

        level_input_output_column_mapping = {
            in_col: out_col
            for in_col, out_col in strictzip(
                level_input_column_names,
                level_column_names,
            )
            if in_col is not None
        }

        return level_input_output_column_mapping


class OutputWriterFactory:
    """Factory class for creating output writers based on configuration."""

    def __init__(
        self,
        *,
        id_column: str | None,
        spectra_columns: list[str] | tuple[str, ...],
        extra_output_columns: list[str],
        append_to_output_file: bool,
        write_decoys: bool,
        sqlite_path: Path | None,
    ):
        # Q: are we deleting the sqlite ops?
        self.is_sqlite = sqlite_path is not None
        self.sqlite_path = sqlite_path
        self.write_decoys = write_decoys
        self.extra_output_columns = extra_output_columns
        self.append_to_output_file = append_to_output_file
        id_col = [id_column] if id_column is not None else []
        self.output_column_names = [
            *id_col,
            *spectra_columns,
            "peptide",
            *extra_output_columns,
            # Q: should we prefix these with "mokapot"?
            "score",
            Q_VALUE_COL_NAME,
            "posterior_error_prob",
            "proteinIds",
        ]

        self.output_column_names_proteins = [
            "mokapot protein group",
            "best peptide",
            "stripped sequence",
            "score",
            Q_VALUE_COL_NAME,
            "posterior_error_prob",
        ]

    def __repr__(self) -> str:
        formatted_dict = pformat(self.__dict__)
        return f"{self.__class__!s}({formatted_dict})"

    def _create_writer(
        self,
        *,
        path: Path | None,
        level: str,
        initialize: bool,
    ) -> TabularDataWriter | ConfidenceSqliteWriter:
        """Create appropriate writer based on output type and level.

        Meant to be used internally. (see `build_writers`)
        """
        # JSPP 2024-12-20 The same pattern happens here; where passing sqlite
        # Just ignores all other arguments ...
        output_columns = (
            self.output_column_names_proteins
            if level == "proteins"
            else self.output_column_names
        )

        if self.is_sqlite:
            # Note this assertion is not meant to be used
            # for control flow, it checks that the code is sound
            # so far and I dont think it should be a recoverable
            # error.
            assert isinstance(self.sqlite_path, Path), (
                "sqlite_path must be a Path object, "
                f"got {type(self.sqlite_path)}"
            )
            if path is not None:
                LOGGER.warning(
                    "The path argument is ignored when using sqlite output."
                )
            return ConfidenceSqliteWriter(
                self.sqlite_path,
                columns=output_columns,
                column_types=[],
                level=level,
                qvalue_column=Q_VALUE_COL_NAME,
                pep_column="posterior_error_prob",
            )

        writer = TabularDataWriter.from_suffix(path, output_columns, [])
        if initialize:
            writer.initialize()
        return writer

    def build_writers(
        self, level_manager: LevelManager, prefix: str | None = None
    ) -> tuple[
        dict[str, list[TabularDataWriter] | list[ConfidenceSqliteWriter]], str
    ]:
        """Build output writers for each level.

        Parameters
        ----------
        level_manager : LevelManager
            The level manager.
        prefix : str, optional
            The prefix to use for the output files, by default None.
            It will be used to create the file names whose pattern is
            "{self.dest_dir}/{file_root}{prefix}.{level}{file_ext}".

        Returns
        -------
        tuple[
            dict[
                str,
                list[TabularDataWriter] | list[ConfidenceSqliteWriter]
            ],
            str
        ]
            A tuple containing the output writers and the file prefix.

        """
        output_writers = {}

        file_prefix = level_manager.file_root
        if prefix:
            file_prefix = f"{file_prefix}{prefix}."

        for level in level_manager.levels_or_proteins:
            output_writers[level] = []

            name = [
                str(file_prefix),
                "targets.",
                str(level),
                str(level_manager.default_extension),
            ]

            outfile_targets = level_manager.dest_dir / "".join(name)

            output_writers[level].append(
                self._create_writer(
                    path=outfile_targets,
                    level=level,
                    initialize=not self.append_to_output_file,
                )
            )

            if self.write_decoys and not self.is_sqlite:
                decoy_name = [
                    str(file_prefix),
                    "decoys.",
                    str(level),
                    str(level_manager.default_extension),
                ]
                outfile_decoys = level_manager.dest_dir / "".join(decoy_name)
                output_writers[level].append(
                    self._create_writer(
                        path=outfile_decoys,
                        level=level,
                        initialize=not self.append_to_output_file,
                    )
                )

        return output_writers, file_prefix


@contextmanager
@typechecked
def create_sorted_file_reader(
    dataset: PsmDataset,
    score_reader: TabularDataReader,
    dest_dir: Path,
    file_prefix: str,
    deduplication_columns: list[str] | tuple[str, ...] | None,
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

    # Q: Why is it reading all non-feature columns instead of just reading the
    #    spectrum columns?
    input_columns = list(dataset.metadata_columns) + ["score"]
    output_columns = [
        input_output_column_mapping.get(name, name) for name in input_columns
    ]
    file_iterator = reader.get_chunked_data_iterator(
        chunk_size=CONFIDENCE_CHUNK_SIZE,
        columns=output_columns,
    )

    #  Write those chunks in parallel, where the columns are given
    #  by dataset.metadata plus the "scores" column

    # Q: why does it have to save the temp chunks in the same format?
    # OLD: outfile_ext = dataset.reader.file_name.suffix
    outfile_ext = dataset.get_default_extension()
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
        readers=readers,
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
    tmp = make_bool_trarget(chunk_metadata.loc[:, target_column])
    # Setting the temporaty column and deleting the original
    # column solves a deprecation warning that mentions "assiging
    # column with incompatible dtype"
    del chunk_metadata[target_column]
    chunk_metadata.loc[:, target_column] = tmp
    chunk_metadata.sort_values(by="score", ascending=False, inplace=True)

    if deduplication_columns is not None:
        # This is not strictly necessary, as we deduplicate also afterwards,
        # but speeds up the process
        try:
            chunk_metadata.drop_duplicates(deduplication_columns, inplace=True)
        except KeyError as e:
            msg = "Duplication error trying to use the following columns: "
            msg += str(deduplication_columns)
            msg += f". Found: {chunk_metadata.columns} "
            msg += ". Please check the input data."
            raise KeyError(msg) from e

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
    score_stats: OnlineStatistics | None,
    peps_error: bool,
    level: str,
    eval_fdr: float,
):
    # Note: the score stats are only used for the confidence estimation
    #       if the streaming mode is used.
    # TODO: split this function into two, one for streaming and one for
    #       non-streaming.
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


def plot_qvalues(qvalues: np.ndarray[float], threshold=0.1, ax=None, **kwargs):
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
        if plt is None:
            raise RuntimeError(
                "Matplotlib is not installed. Confidence plots will not be "
                "available."
            )
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
