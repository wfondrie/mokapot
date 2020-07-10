"""
This is the command line interface for mokapot
"""
import sys
import logging

import numpy as np

from mokapot import __version__
from .config import Config
from .parsers import read_pin
from .brew import brew


def main():
    """The CLI entry point"""
    # Get command line arguments
    config = Config()

    # Setup logging
    verbosity_dict = {0: logging.ERROR,
                      1: logging.WARNING,
                      2: logging.INFO,
                      3: logging.DEBUG}

    logging.basicConfig(format=("[{levelname}] {message}"),
                        style="{", level=verbosity_dict[config.verbosity])

    logging.info("mokapot version %s", str(__version__))
    logging.info("Written by William E. Fondrie (wfondrie@uw.edu) in the")
    logging.info("Department of Genome Sciences at the University of "
                 "Washington.")
    logging.info("Command issued:")
    logging.info("%s", " ".join(sys.argv))
    logging.info("")
    logging.info("Starting Analysis")
    logging.info("=================")
    np.random.seed(config.seed)
    dataset = read_pin(config.pin_files)
    psms = brew(dataset,
                train_fdr=config.train_fdr,
                test_fdr=config.test_fdr,
                max_iter=config.max_iter,
                direction=config.direction,
                folds=config.folds)

    psms.to_txt(dest_dir=config.dest_dir, file_root=config.file_root)
