import logging
from pathlib import Path

import numpy as np
from typeguard import typechecked

from mokapot.cli_helper import make_timer
from mokapot.column_defs import STANDARD_COLUMN_NAME_MAP
from mokapot.confidence import compute_and_write_confidence
from mokapot.statistics import OnlineStatistics
from mokapot.tabular_data import (
    auto_finalize,
    BufferType,
    ComputedTabularDataReader,
    MergedTabularDataReader,
    remove_columns,
    TabularDataReader,
    TabularDataWriter,
)
from mokapot.tabular_data.target_decoy_writer import TargetDecoyWriter


@typechecked
def compute_rollup_levels(
    base_level: str, parent_levels: dict[str, str] | None = None
) -> list[str]:
    if parent_levels is None:
        parent_levels = DEFAULT_PARENT_LEVELS
    levels = [base_level]
    changed = True
    while changed:
        changed = False
        for child, parent in parent_levels.items():
            if (parent in levels) and (child not in levels):
                levels.append(child)
                changed = True
    return levels


def get_target_decoy_reader(path: Path, is_decoy: bool):
    return ComputedTabularDataReader(
        reader=TabularDataReader.from_path(path, column_map=STANDARD_COLUMN_NAME_MAP),
        column="is_decoy",
        dtype=np.dtype("bool"),
        func=lambda df: np.full(len(df), is_decoy),
    )


DEFAULT_PARENT_LEVELS = {
    "precursor": "psm",
    "modified_peptide": "precursor",
    "peptide": "modified_peptide",
    "peptide_group": "precursor",  # due to "unknown nature" of peptide groups
}


@typechecked
def do_rollup(config):
    # todo: refactor: this function is far too long. Should be split. Probably
    #  at least one function to configure the input readers, one to write the
    #  intermediate/temp files, and one that computes the statistics (q-values
    #  and peps and writes the output files)
    base_level: str = config.level
    src_dir: Path = config.src_dir
    dest_dir: Path = config.dest_dir
    file_root: str = config.file_root + "."

    # Determine input files
    if len(list(src_dir.glob(f"*.{base_level}s.parquet"))) > 0:
        if len(list(src_dir.glob(f"*.{base_level}s.csv"))) > 0:
            raise RuntimeError(
                "Only input files of either type CSV or type Parquet should "
                f"exist in '{src_dir}', but both types were found."
            )
        suffix = ".parquet"
    else:
        suffix = ".csv"

    target_files: list[Path] = sorted(src_dir.glob(f"*.targets.{base_level}s{suffix}"))
    decoy_files: list[Path] = sorted(src_dir.glob(f"*.decoys.{base_level}s{suffix}"))
    target_files = [
        file for file in target_files if not file.name.startswith(file_root)
    ]
    decoy_files = [file for file in decoy_files if not file.name.startswith(file_root)]
    in_files: list[Path] = sorted(target_files + decoy_files)
    logging.info(f"Reading files: {[str(file) for file in in_files]}")
    if len(in_files) == 0:
        raise ValueError("No input files found.")

    # Configure readers (read targets/decoys and adjoin is_decoy column)
    target_readers = [get_target_decoy_reader(path, False) for path in target_files]
    decoy_readers = [get_target_decoy_reader(path, True) for path in decoy_files]
    reader = MergedTabularDataReader(
        target_readers + decoy_readers,
        priority_column="score",
        reader_chunk_size=10000,
    )

    # Determine out levels
    levels = compute_rollup_levels(base_level, DEFAULT_PARENT_LEVELS)
    levels_not_found = [
        level for level in levels if level not in reader.get_column_names()
    ]
    levels = [level for level in levels if level in reader.get_column_names()]
    logging.info(f"Rolling up to levels: {levels}")
    if len(levels_not_found) > 0:
        logging.info(f"  (Rollup levels not found in input: {levels_not_found})")

    # Determine temporary files
    temp_files = {
        level: dest_dir / f"{file_root}temp.{level}s{suffix}" for level in levels
    }
    logging.debug(
        "Using temp files: "
        f"{ {level: str(file) for level, file in temp_files.items()} }"
    )

    # Determine columns for output files and intermediate files
    in_column_names = reader.get_column_names()
    in_column_types = reader.get_column_types()

    temp_column_names, temp_column_types = remove_columns(
        in_column_names, in_column_types, ["q_value", "posterior_error_prob"]
    )

    # Configure temp writers
    merge_row_type = BufferType.Dicts

    temp_buffer_size = 1000

    temp_writers = {
        level: TabularDataWriter.from_suffix(
            temp_files[level],
            columns=temp_column_names,
            column_types=temp_column_types,
            buffer_size=temp_buffer_size,
            buffer_type=merge_row_type,
        )
        for level in levels
    }

    # todo: discuss: We need an option to write parquet or sql for example
    #  (also, the  output file type could depend on the input file type)

    # Write temporary files which contain only the best scoring entity of a
    # given level
    logging.debug("Writing temp files: %s", [str(file) for file in temp_files.values()])

    timer = make_timer()
    score_stats = OnlineStatistics()
    with auto_finalize(temp_writers.values()):
        count = 0
        seen_entities: dict[str, set] = {level: set() for level in levels}
        for data_row in reader.get_row_iterator(
            temp_column_names, row_type=merge_row_type
        ):
            count += 1
            if count % 10000 == 0:
                logging.debug(f"  Processed {count} lines ({timer():.2f} seconds)")

            for level in levels:
                seen = seen_entities[level]
                id_col = level
                if merge_row_type == BufferType.DataFrame:
                    id = data_row.loc[0, id_col]
                else:
                    id = data_row[id_col]
                if id not in seen:
                    seen.add(id)
                    temp_writers[level].append_data(data_row)

            score_stats.update_single(data_row["score"])

        logging.info(f"Read {count} PSMs")
        logging.debug(f"Score statistics: {score_stats.describe()}")
        for level in levels:
            seen = seen_entities[level]
            logging.info(f"Rollup level {level}: found {len(seen)} unique entities")

    # Determine output files
    out_files_map = {
        level: [
            dest_dir / f"{file_root}targets.{level}s{suffix}",
            dest_dir / f"{file_root}decoys.{level}s{suffix}",
        ]
        for level in levels
    }

    # Configure temp readers and output writers
    buffer_size = 1000
    output_columns, output_types = remove_columns(
        in_column_names, in_column_types, ["is_decoy"]
    )
    output_options = dict(
        columns=output_columns,
        column_types=output_types,
        buffer_size=buffer_size,
    )

    def create_writer(path: Path):
        return TabularDataWriter.from_suffix(path, **output_options)

    for level in levels:
        output_writers = list(map(create_writer, out_files_map[level]))
        writer = TargetDecoyWriter(
            output_writers, write_decoys=True, decoy_column="is_decoy"
        )
        with auto_finalize(output_writers):
            temp_reader = temp_writers[level].get_associated_reader()
            compute_and_write_confidence(
                temp_reader,
                writer,
                config.qvalue_algorithm,
                config.peps_algorithm,
                config.stream_confidence,
                score_stats,
                peps_error=True,
                level=level,
                eval_fdr=0.01,
            )
