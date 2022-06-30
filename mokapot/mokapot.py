"""
This is the command line interface for mokapot
"""
import sys
import time
import logging
import datetime
from functools import partial
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config
from .parsers.pin import read_pin
from .parsers.pepxml import read_pepxml
from .parsers.fasta import read_fasta
from .brew import brew
from .model import PercolatorModel, load_model


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
        "Department of Genome Sciences at the University of Washington."
    )
    logging.info("Command issued:")
    logging.info("%s", " ".join(sys.argv))
    logging.info("")
    logging.info("Starting Analysis")
    logging.info("=================")

    np.random.seed(config.seed)

    # Parse Datasets
    parse = get_parser(config)
    if config.aggregate or len(config.psm_files) == 1:
        datasets = parse(config.psm_files)
    else:
        datasets = [parse(f) for f in config.psm_files]
        prefixes = [Path(f).stem for f in config.psm_files]

    # Parse FASTA, if required:
    if config.proteins is not None:
        logging.info("Protein-level confidence estimates enabled.")
        proteins = read_fasta(
            config.proteins,
            enzyme=config.enzyme,
            missed_cleavages=config.missed_cleavages,
            clip_nterm_methionine=config.clip_nterm_methionine,
            min_length=config.min_length,
            max_length=config.max_length,
            semi=config.semi,
            decoy_prefix=config.decoy_prefix,
        )

        if config.aggregate or len(config.psm_files) == 1:
            datasets.add_proteins(proteins)
        else:
            for dataset in datasets:
                dataset.add_proteins(proteins)

    # Define a model:
    if config.load_models:
        model = [load_model(model_file) for model_file in config.load_models]
    else:
        model = PercolatorModel(
            train_fdr=config.train_fdr,
            max_iter=config.max_iter,
            direction=config.direction,
            override=config.override,
            subset_max_train=config.subset_max_train,
        )

    # Fit the models:
    psms, models = brew(
        datasets,
        model=model,
        test_fdr=config.test_fdr,
        folds=config.folds,
        max_workers=config.max_workers,
    )

    if config.dest_dir is not None:
        Path(config.dest_dir).mkdir(exist_ok=True)

    if config.save_models:
        logging.info("Saving models...")
        for i, trained_model in enumerate(models):
            out_file = f"mokapot.model_fold-{i+1}.pkl"

            if config.file_root is not None:
                out_file = ".".join([config.file_root, out_file])

            if config.dest_dir is not None:
                out_file = Path(config.dest_dir, out_file)

            trained_model.save(str(out_file))

    # Determine how to write the results:
    logging.info("Writing results...")
    if config.aggregate or len(config.psm_files) == 1:
        psms.to_txt(
            dest_dir=config.dest_dir,
            file_root=config.file_root,
            decoys=config.keep_decoys,
        )
    else:
        for dat, prefix in zip(psms, prefixes):
            if config.file_root is not None:
                prefix = ".".join([config.file_root, prefix])

            dat.to_txt(
                dest_dir=config.dest_dir,
                file_root=prefix,
                decoys=config.keep_decoys,
            )

    total_time = round(time.time() - start)
    total_time = str(datetime.timedelta(seconds=total_time))

    logging.info("")
    logging.info("=== DONE! ===")
    logging.info("mokapot analysis completed in %s", total_time)


def get_parser(config):
    """Figure out which parser to use.

    Note that this just looks at file extensions, but in the future it might be
    good to check the contents of the file. I'm just not sure how to do this
    in an efficient way, particularly for gzipped files.

    Parameters
    ----------
    config : argparse object
         The configuration details.

    Returns
    -------
    callable
         Returns the correct parser for the files.

    """
    pepxml_ext = {".pep.xml", ".pepxml", ".xml"}
    num_pepxml = 0
    for psm_file in config.psm_files:
        ext = Path(psm_file).suffixes
        if len(ext) > 2:
            ext = "".join(ext[-2:])
        else:
            ext = "".join(ext)

        if ext.lower() in pepxml_ext:
            num_pepxml += 1

    if num_pepxml == len(config.psm_files):
        return partial(
            read_pepxml,
            open_modification_bin_size=config.open_modification_bin_size,
            decoy_prefix=config.decoy_prefix,
        )

    return read_pin


if __name__ == "__main__":
    main()
