"""
This is the command line interface for mokapot
"""

import datetime
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from . import __version__
from .algorithms import configure_algorithms
from .brew import brew
from .cli_helper import setup_logging
from .confidence import assign_confidence
from .config import create_config_parser
from .model import load_model, PercolatorModel
from .parsers.fasta import read_fasta
from .parsers.pin import read_pin
from .typecheck import register_numpy_typechecker


def main(main_args=None):
    """The CLI entry point"""
    start = time.time()
    register_numpy_typechecker()

    # Get command line arguments
    parser = create_config_parser()
    config = parser.parse_args(args=main_args)

    # Setup logging
    setup_logging(config)

    # Suppress warning if asked for
    if config.suppress_warnings:
        warnings.filterwarnings("ignore")

    # Write header
    logging.info("mokapot version %s", str(__version__))
    logging.info("Written by William E. Fondrie (wfondrie@uw.edu) in the")
    logging.info("Department of Genome Sciences at the University of Washington.")

    # Configure confidence algorithms
    configure_algorithms(config)

    # Check config parameter validity
    if config.stream_confidence and config.peps_algorithm != "hist_nnls":
        raise ValueError(
            f"Streaming and PEPs algorithm `{config.peps_algorithm}` not "
            "compatible. Use `--peps_algorithm=hist_nnls` instead.`"
        )

    # Start analysis
    logging.info("Command issued:")
    logging.info("%s", " ".join(sys.argv))
    logging.info("")
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
        logging.debug(f"Loading models ({config.load_models})")
        model = [load_model(model_file) for model_file in config.load_models]
    else:
        logging.debug("Instantiating Percolator model.")
        model = PercolatorModel(
            train_fdr=config.train_fdr,
            max_iter=config.max_iter,
            direction=config.direction,
            override=config.override,
            rng=config.seed,
        )

    # Fit the models:
    models, scores = brew(
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
        datasets=datasets,
        max_workers=config.max_workers,
        scores_list=scores,
        eval_fdr=config.test_fdr,
        dest_dir=config.dest_dir,
        file_root=file_root,
        prefixes=prefixes,
        write_decoys=config.keep_decoys,
        deduplication=not config.skip_deduplication,
        do_rollup=not config.skip_rollup,
        proteins=proteins,
        peps_error=config.peps_error,
        peps_algorithm=config.peps_algorithm,
        sqlite_path=config.sqlite_db_path,
        stream_confidence=config.stream_confidence,
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
