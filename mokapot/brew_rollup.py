"""
This is the command line interface for mokapot
"""

import argparse
import logging
import sys
from argparse import _ArgumentGroup as ArgumentGroup
from pathlib import Path

import numpy as np

from mokapot import __version__
from mokapot.algorithms import configure_algorithms
from mokapot.cli_helper import output_end_message, output_start_message, setup_logging
from mokapot.rollup import do_rollup


def parse_arguments(main_args):
    # Get command line arguments
    """The parser"""
    # todo: we should update this copyright notice asap
    desc = (
        f"mokapot version {__version__}.\n"
        "Originally written by William E. Fondrie (wfondrie@talus.bio) while in the \n"
        "Department of Genome Sciences at the University of Washington.\n\n"
        "Extended by Samia Ben Fredj, Elmar Zander, Vishal Sukumar and \n"
        "Siegfried Gessulat while at MSAID. \n\n"
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
    add_logging_options(misc_options)

    args = parser.parse_args(args=main_args)
    return args


def add_main_options(parser: ArgumentGroup) -> None:
    parser.add_argument(
        "--level",
        choices=[
            "psm",
            "precursor",
            "modifiedpeptide",
            "peptide",
            "peptidegroup",
        ],
        required=True,
        help=(
            "Load previously saved models and skip model training."
            "Note that the number of models must match the value of --folds."
        ),
    )

    parser.add_argument(
        "-s",
        "--src_dir",
        type=Path,
        default=Path("."),
        help=("The directory in which to look for the files to rollup."),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help=("An integer to use as the random seed."),
    )


def add_output_options(parser: ArgumentGroup) -> None:
    parser.add_argument(
        "-d",
        "--dest_dir",
        type=Path,
        default=Path("."),
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
        "--tdc",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Specifies whether input comes from target decoy competition "
            "(default) or from separate search."
        ),
    )

    parser.add_argument(
        "--pi0_algorithm",
        default="default",
        choices=[
            "default",
            "ratio",
            "slope",
            "storey_smoother",
            "storey_fixed",
            "storey_bootstrap",
        ],
        help=("Specify the algorithm for pi0 estimation. "),
    )

    parser.add_argument(
        "--pi0_eval_lambda",
        default=0.5,
        help=(
            "Specify the lambda in Storey's pi0 estimation for evaluation "
            "(works currently only with storey_* pi0 algorithms."
        ),
    )

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
        "--qvalue_algorithm",
        default="default",
        choices=["default", "tdc", "from_counts", "storey"],
        help=(
            "Specify the algorithm for qvalue computation. If the `tdc` option"
            "is set to true (which is the default0 `default` evals to `tdc`, "
            "the original mokapot algorithm, which works only with tdc. "
            "Otherwise, it defaults to `storey`."
        ),
    )
    parser.add_argument(
        "--stream_confidence",
        default=False,
        action="store_true",
        help=("Specify whether confidence assignment shall be streamed."),
    )


def add_logging_options(parser: ArgumentGroup) -> None:
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
        "--log_time",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=("Specifies whether (and thread) should also be logged."),
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


def main(main_args=None):
    """The CLI entry point"""

    config = parse_arguments(main_args)
    prog_name = "brew_rollup"

    setup_logging(config)

    configure_algorithms(config)

    start_time = output_start_message(prog_name, config)

    np.random.seed(config.seed)

    if config.dest_dir is not None:
        config.dest_dir.mkdir(exist_ok=True)

    do_rollup(config)

    output_end_message(prog_name, config, start_time)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"[Error] {e}")
        sys.exit(250)  # input failure
