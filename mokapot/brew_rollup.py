"""
This is the command line interface for mokapot
"""

import datetime
import logging
import sys
import time
import argparse
from argparse import _ArgumentGroup  as ArgumentGroup
from pathlib import Path

import numpy as np
import pandas as pd
from typeguard import typechecked, TypeCheckError

from mokapot.streaming import merge_readers, MergedTabularDataReader, \
    ComputedTabularDataReader
from mokapot import __version__, qvalues
from mokapot.tabular_data import TabularDataWriter, TabularDataReader, \
    auto_finalize
from mokapot.peps import peps_from_scores


def parse_arguments(main_args):
    # Get command line arguments
    """The parser"""
    # Todo: we should update this
    desc = (
        f"mokapot version {__version__}.\n"
        "Written by William E. Fondrie (wfondrie@talus.bio) while in the \n"
        "Department of Genome Sciences at the University of Washington.\n\n"
        "Official code website: https://github.com/wfondrie/mokapot\n\n"
        "More documentation and examples: https://mokapot.readthedocs.io"
    )

    parser = argparse.ArgumentParser(
        description=desc  # , formatter_class=MokapotHelpFormatter
    )

    main_options = parser.add_argument_group("Main options")
    add_main_options(main_options)

    output_options = parser.add_argument_group("Output options")
    add_output_options(output_options)

    confidence_options = parser.add_argument_group("Confidence options")
    add_confidence_options(confidence_options)

    misc_options = parser.add_argument_group("Miscellaneous options")
    add_misc_options(misc_options)

    args = parser.parse_args(args=main_args)
    return args


def add_main_options(parser: ArgumentGroup) -> None:
    parser.add_argument(
        "--level",
        choices=['psm', 'precursor', 'modifiedpeptide', 'peptide', 'peptidegroup'],
        required=True,
        help=(
            "Load previously saved models and skip model training."
            "Note that the number of models must match the value of --folds."
        ),
    )



def add_output_options(parser: ArgumentGroup) -> None:
    parser.add_argument(
        "--keep_decoys",
        default=False,
        action="store_true",
        help=("Keep the decoys in the output .txt files"),
    )
    parser.add_argument(
        "-d",
        "--dest_dir",
        type=Path,
        help=(
            "The directory in which to write the result files. Defaults to "
            "the current working directory"
        ),
    )
    parser.add_argument(
        "-r",
        "--file_root",
        default="rollup",
        type=str,
        help="The prefix added to all file names.",
    )


def add_confidence_options(parser: ArgumentGroup) -> None:
    parser.add_argument(
        "--peps_algorithm",
        default="qvality",
        choices=["qvality", "qvality_bin", "kde_nnls", "hist_nnls"],
        help=(
            "Specify the algorithm for pep computation. 'qvality_bin' works "
            "only if the qvality binary is on the search path"
        ),
    )
    parser.add_argument(
        "--peps_error",
        default=False,
        action="store_true",
        help=("raise error when all PEPs values are equal to 1."),
    )
    parser.add_argument(
        "--qvalue_algorithm",
        default="tdc",
        choices=["tdc", "from_peps", "from_counts"],
        help=(
            "Specify the algorithm for qvalue computation. `tdc is` the default mokapot algorithm."
        ),
    )


def add_misc_options(parser: ArgumentGroup) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help=("An integer to use as the random seed."),
    )
    parser.add_argument(
        "-w",
        "--max_workers",
        default=1,
        type=int,
        help=(
            "The number of processes to use for model training. Note that "
            "using more than one worker will result in garbled logging "
            "messages."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        default=2,
        type=int,
        choices=[0, 1, 2, 3],
        help=(
            "Specify the verbosity of the current "
            "process. Each level prints the following "
            "messages, including all those at a lower "
            "verbosity: 0-errors, 1-warnings, 2-messages"
            ", 3-debug info."
        ),
    )
    parser.add_argument(
        "--suppress_warnings",
        default=False,
        action="store_true",
        help=(
            "Suppress warning messages when running mokapot. "
            "Should only be used when running mokapot in production."
        ),
    )


def setup_logging(config):
    # Setup logging
    verbosity_dict = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    logging.basicConfig(
        format=("[{levelname}] {message}"),
        style="{",
        level=verbosity_dict[config.verbosity],
    )
    logging.captureWarnings(True)


def output_start_message(config):
    # todo: need to update that too
    start_time = time.time()
    logging.info("mokapot version %s", str(__version__))
    logging.info("Written by William E. Fondrie (wfondrie@uw.edu) in the")
    logging.info("Department of Genome Sciences at the University of Washington.")
    logging.info("Command issued:")
    logging.info("%s", " ".join(sys.argv))
    logging.info("")
    logging.info("Starting Analysis")
    logging.info("=================")
    return start_time

def output_end_message(config, start_time):
    total_time = round(time.time() - start_time)
    total_time = str(datetime.timedelta(seconds=total_time))

    logging.info("")
    logging.info("=== DONE! ===")
    logging.info("mokapot analysis completed in %s", total_time)

DEFAULT_PARENT_LEVELS = {
    "precursor": "psm",
    "modified_peptide": "precursor",
    "peptide": "modified_peptide",
    "peptide_group": "precursor", # due to "unknown nature" of peptide groups
}

@typechecked
def compute_rollup_levels(base_level: str, parent_levels: dict[str, str] | None=None) -> list[str]:
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

STANDARD_COLUMN_NAME_MAP = {
    "SpecId": "psm_id",
    "PSMId": "psm_id",
    "Precursor": "precursor",
    "pcm": "precursor",
    "PCM": "precursor",
    "Peptide": "peptide",
    "PeptideGroup": "peptide_group",
    "peptidegroup": "peptide_group",
    "ModifiedPeptide": "modified_peptide",
    "modifiedpeptide": "modified_peptide",
    "q-value": "q_value"
}


@typechecked
def do_rollup(config):
    base_level = config.level
    dest_dir = config.dest_dir
    file_root = config.file_root + "."

    # Determine input files
    target_files: list[Path] = sorted(dest_dir.glob(f"*.targets.{base_level}s"))
    target_files = [file for file in target_files if not file.name.startswith(file_root)]
    decoy_files: list[Path] = sorted(dest_dir.glob(f"*.decoys.{base_level}s"))
    decoy_files = [file for file in decoy_files if not file.name.startswith(file_root)]
    in_files: list[Path] = sorted(target_files + decoy_files)
    logging.info(f"Reading files: {[str(file) for file in in_files]}")
    # todo: message if no input files found

    # Configure readers (read targets/decoys and adjoin is_decoy column)
    target_readers = [ComputedTabularDataReader(
        reader=TabularDataReader.from_path(path, column_map=STANDARD_COLUMN_NAME_MAP),
        column="is_decoy", dtype=np.dtype('bool'), func=lambda df: np.full(len(df), False)) for path in target_files]
    decoy_readers = [ComputedTabularDataReader(
        reader=TabularDataReader.from_path(path, column_map=STANDARD_COLUMN_NAME_MAP),
        column="is_decoy", dtype=np.dtype('bool'), func=lambda df: np.full(len(df), True)) for path in decoy_files]
    reader = MergedTabularDataReader(target_readers + decoy_readers, priority_column="score")

    # Determine out levels
    levels = compute_rollup_levels(base_level, DEFAULT_PARENT_LEVELS)
    levels_not_found = [level for level in levels if level not in reader.get_column_names()]
    levels = [level for level in levels if level in reader.get_column_names()]
    logging.info(f"Rolling up to levels: {levels}")
    logging.info(f"  (Rollup levels not found in input: {levels_not_found})")

    # Determine temporary files
    temp_files = {level: dest_dir / f"{file_root}temp.{level}s" for level in levels}
    logging.info(f"Using temp files: { {level: str(file) for level, file in temp_files.items()} }")

    # Determine output files
    out_files = {level: [dest_dir / f"{file_root}targets.{level}s",
                         dest_dir / f"{file_root}decoys.{level}s", ]
                 for level in levels}
    logging.info(f"Writing to files: { {level: list(map(str, files)) for level, files in out_files.items()} }")


    # Determine columns for output files and intermediate files
    column_names = reader.get_column_names()
    column_types = reader.get_column_types()

    temp_columns = [(column, type) for column, type in zip(column_names, column_types)
        if column not in ["q_value", "posterior_error_prob"]]
    temp_column_names, temp_column_types = map(list, zip(*temp_columns))

    # Configure temp writers
    temp_buffer_size = 1000

    temp_writers = {level: TabularDataWriter.from_suffix(temp_files[level],
                                                         columns=temp_column_names,
                                                         column_types=temp_column_types,
                                                         buffer_size = temp_buffer_size)
                    for level in levels}

    # todo: We need an option to write parquet or sql for example (also, the
    #  output file type could depend on the input file type)

    # Write temporary files which contain only the best scoring entity of a given level
    with auto_finalize(temp_writers.values()):
        count = 0
        seen_entities: dict[str, set] = {level: set() for level in levels}
        for line in reader.get_row_iterator(temp_column_names):
            count += 1
            for level in levels:
                seen = seen_entities[level]
                id_col = level
                id = line.loc[0, id_col]
                if id not in seen:
                    seen.add(id)
                    temp_writers[level].append_data(line)

        logging.info(f"Read {count} PSMs")
        for level in levels:
            seen = seen_entities[level]
            logging.info(f"Rollup level {level}: found {len(seen)} unique entities")

    # Configure temp readers and output writers
    buffer_size = 1000
    output_options = dict(columns=column_names, column_types=column_types,
                          buffer_size=buffer_size)
    create_writer = lambda path: TabularDataWriter.from_suffix(path, **output_options)


    for level in levels:
        reader = temp_writers[level].get_associated_reader()
        output_writers = list(map(create_writer, out_files[level]))

        # data = reader.read(columns=["is_decoy", "score"])
        data = reader.read()

        scores = data["score"].values
        targets = ~data["is_decoy"].values

        qvals = qvalues.qvalues_from_scores(scores, targets, config.qvalue_algorithm)
        peps = peps_from_scores(scores, targets, config.peps_algorithm)

        data["q_value"] = qvals
        data["posterior_error_prob"] = peps

        # todo: need to remove is_decoy column again
        output_writers[0].write(data.loc[targets, column_names])
        output_writers[1].write(data.loc[~targets, column_names])


def main(main_args=None):
    """The CLI entry point"""

    config = parse_arguments(main_args)

    setup_logging(config)

    start_time = output_start_message(config)

    np.random.seed(config.seed)

    if config.dest_dir is not None:
        config.dest_dir.mkdir(exist_ok=True)

    do_rollup(config)

    output_end_message(config, start_time)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"[Error] {e}")
        sys.exit(250)  # input failure
