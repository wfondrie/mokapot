"""
This is the command line interface for mokapot
"""
import os
import sys
import time
import logging
import datetime

import numpy as np

from mokapot import __version__
from .config import Config
from .parsers import read_pin
from .brew import brew
from .model import PercolatorModel
from .proteins import FastaProteins


def main():
    """The CLI entry point"""
    start = time.time()

    # Get command line arguments
    config = Config()

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

    logging.info("mokapot version %s", str(__version__))
    logging.info("Written by William E. Fondrie (wfondrie@uw.edu) in the")
    logging.info(
        "Department of Genome Sciences at the University of " "Washington."
    )
    logging.info("Command issued:")
    logging.info("%s", " ".join(sys.argv))
    logging.info("")
    logging.info("Starting Analysis")
    logging.info("=================")

    np.random.seed(config.seed)

    # Parse Datasets
    if config.aggregate or len(config.pin_files) == 1:
        datasets = read_pin(config.pin_files)
    else:
        datasets = [read_pin(f) for f in config.pin_files]
        prefixes = [
            os.path.splitext(os.path.basename(f))[0] for f in config.pin_files
        ]

    # Parse FASTA, if required:
    if config.proteins is not None:
        logging.info("Protein-level confidence estimates enabled.")
        proteins = FastaProteins(
            config.proteins,
            enzyme=config.enzyme,
            missed_cleavages=config.missed_cleavages,
            clip_nterm_methionine=config.clip_nterm_methionine,
            min_length=config.min_length,
            max_length=config.max_length,
            semi=config.semi,
            decoy_prefix=config.decoy_prefix,
        )

        if config.aggregate or len(config.pin_files) == 1:
            datasets.add_proteins(proteins)
        else:
            for dataset in datasets:
                dataset.add_proteins(proteins)

    # Define a model:
    model = PercolatorModel(
        train_fdr=config.train_fdr,
        max_iter=config.max_iter,
        direction=config.direction,
        override=config.override,
        subset_max_train=config.subset_max_train,
    )

    # Fit the models:
    psms, models = brew(
        datasets, model=model, test_fdr=config.test_fdr, folds=config.folds
    )

    if config.dest_dir is not None:
        os.makedirs(config.dest_dir, exist_ok=True)

    if config.save_models:
        logging.info("Saving models...")
        for i, trained_model in enumerate(models):
            out_file = f"mokapot.model_fold-{i+1}.pkl"

            if config.file_root is not None:
                out_file = ".".join([config.file_root, out_file])

            if config.dest_dir is not None:
                out_file = os.path.join(config.dest_dir, out_file)

            trained_model.save(out_file)

    # Determine how to write the results:
    logging.info("Writing results...")
    if config.aggregate or len(config.pin_files) == 1:
        psms.to_txt(dest_dir=config.dest_dir, file_root=config.file_root)
    else:
        for dat, prefix in zip(psms, prefixes):
            if config.file_root is not None:
                prefix = ".".join([config.file_root, prefix])

            dat.to_txt(dest_dir=config.dest_dir, file_root=prefix)

    total_time = round(time.time() - start)
    total_time = str(datetime.timedelta(seconds=total_time))

    logging.info("")
    logging.info("=== DONE! ===")
    logging.info("mokapot analysis completed in %s", total_time)


if __name__ == "__main__":
    main()
