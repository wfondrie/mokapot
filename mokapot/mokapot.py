"""
This is the command line interface for mokapot
"""
import os
import sys
import logging

from .config import Config
from .parsers import read_pin
from .brew import brew


def main():
    """The CLI entry point"""
    # Get command line arguments
    config = Config()

    # If no args are present, show help and exit.
    if len(sys.argv) == 1:
        config.parser.print_help(sys.stderr)
        sys.exit()

    # Setup logging
    verbosity_dict = {0: logging.ERROR,
                      1: logging.WARNING,
                      2: logging.INFO,
                      3: logging.DEBUG}

    logging.basicConfig(format=("[{levelname}] {message}"),
                        style="{", level=verbosity_dict[config.verbosity])

    dataset = read_pin(config.pin_files)
    psms = brew(dataset,
                train_fdr=config.train_fdr,
                test_fdr=config.test_fdr,
                max_iter=config.max_iter,
                direction=config.direction,
                folds=config.folds)

    psms.to_txt(dest_dir=config.dest_dir, file_root=config.file_root)
