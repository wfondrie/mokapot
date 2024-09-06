"""
This is the command line interface for mokapot
"""

import datetime
import logging
import sys
import time
import warnings
import shutil
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config
from .parsers.pin import read_pin
from .parsers.pin_to_tsv import is_valid_tsv, pin_to_valid_tsv
from .parsers.fasta import read_fasta
from .brew import brew
from .model import PercolatorModel, load_model
from .confidence import assign_confidence


def main(main_args=None):
    """The CLI entry point"""
    start = time.time()

    # Get command line arguments
    parser = Config().parser
    config = Config(parser, main_args=main_args)

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

    # Suppress warning if asked for
    if config.suppress_warnings:
        warnings.filterwarnings("ignore")

    logging.info("mokapot version %s", str(__version__))
    logging.info("Written by William E. Fondrie (wfondrie@uw.edu) in the")
    logging.info(
        "Department of Genome Sciences at the University of Washington."
    )
    logging.info("Command issued:")
    logging.info("%s", " ".join(sys.argv))
    logging.info("")

    logging.info("Verify PIN format")
    logging.info("=================")
    if config.verify_pin:
        for path_pin in config.psm_files:
            with open(path_pin, "r") as f_pin:
                valid_tsv = is_valid_tsv(f_pin)
            if not valid_tsv:
                logging.info(f"{path_pin} invalid tsv, converting")
                path_tsv = f"{path_pin}.tsv"
                with open(path_pin, "r") as f_pin:
                    with open(path_tsv, "a") as f_tsv:
                        pin_to_valid_tsv(f_in=f_pin, f_out=f_tsv)
                shutil.move(path_tsv, path_pin)

    logging.info("Starting Analysis")
    logging.info("=================")

    np.random.seed(config.seed)

    # Parse
    datasets = read_pin(config.psm_files, max_workers=config.max_workers)
    if config.aggregate or len(config.psm_files) == 1:
        prefixes = ["" for f in config.psm_files]
    else:
        prefixes = [f.stem for f in config.psm_files]

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
    else:
        proteins = None

    # Define a model:
    model = None
    if config.load_models:
        model = [load_model(model_file) for model_file in config.load_models]

    if model is None:
        logging.debug("Loading Percolator model.")
        model = PercolatorModel(
            train_fdr=config.train_fdr,
            max_iter=config.max_iter,
            direction=config.direction,
            override=config.override,
            rng=config.seed,
        )

    # Fit the models:
    psms, models, scores, desc = brew(
        datasets,
        model=model,
        test_fdr=config.test_fdr,
        folds=config.folds,
        max_workers=config.max_workers,
        subset_max_train=config.subset_max_train,
        ensemble=config.ensemble,
        rng=config.seed,
    )
    logging.info("")

    if config.dest_dir is not None:
        config.dest_dir.mkdir(exist_ok=True)

    if config.file_root is not None:
        file_root = f"{config.file_root}."
    else:
        file_root = ""

    assign_confidence(
        psms=psms,
        max_workers=config.max_workers,
        scores=scores,
        descs=desc,
        eval_fdr=config.test_fdr,
        dest_dir=config.dest_dir,
        file_root=file_root,
        prefixes=prefixes,
        decoys=config.keep_decoys,
        do_rollup=not config.skip_rollup,
        proteins=proteins,
        peps_error=config.peps_error,
        peps_algorithm=config.peps_algorithm,
        qvalue_algorithm=config.qvalue_algorithm,
        sqlite_path=config.sqlite_db_path,
    )

    if config.save_models:
        logging.info("Saving models...")
        for i, trained_model in enumerate(models):
            out_file = Path(f"mokapot.model_fold-{i + 1}.pkl")

            if config.file_root is not None:
                out_file = Path(config.file_root + "." + out_file.name)

            if config.dest_dir is not None:
                out_file = config.dest_dir / out_file

            trained_model.save(out_file)

    total_time = round(time.time() - start)
    total_time = str(datetime.timedelta(seconds=total_time))

    logging.info("")
    logging.info("=== DONE! ===")
    logging.info("mokapot analysis completed in %s", total_time)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except RuntimeError as _e:
        logging.error(f"[Error] {traceback.format_exc()}")
        sys.exit(250)  # input failure
    except ValueError as _e:
        logging.error(f"[Error] {traceback.format_exc()}")
        sys.exit(250)  # input failure
    except Exception as _e:
        logging.error(f"[Error] {traceback.format_exc()}")
        sys.exit(252)
