"""
This is the command line interface for mokapot
"""

import datetime
import logging
import sys
import time
import warnings
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
from .confidence import assign_confidence
from .plugins import get_plugins


def main(main_args=None):
    """The CLI entry point"""
    start = time.time()
    plugins = get_plugins()

    # Get command line arguments
    parser = Config().parser
    for plugin_name, plugin in plugins.items():
        parsergroup = parser.add_argument_group(plugin_name)
        plugin.add_arguments(parsergroup)

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
    logging.debug("Loaded plugins: %s", plugins.keys())

    np.random.seed(config.seed)

    # Parse
    enabled_plugins = {p: plugins[p]() for p in config.plugin}

    datasets = read_pin(config.psm_files, max_workers=config.max_workers)
    if config.aggregate or len(config.psm_files) == 1:
        for plugin in enabled_plugins.values():
            datasets = plugin.process_data(datasets, config)
        prefixes = ["" for f in config.psm_files]
    else:
        for plugin in enabled_plugins.values():
            datasets = [plugin.process_data(ds, config) for ds in datasets]
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
    elif enabled_plugins:
        plugin_models = {}
        for plugin_name, plugin in enabled_plugins.items():
            model = plugin.get_model(config)
            if model is not None:
                logging.debug(f"Loaded model for {plugin_name}")
                plugin_models[plugin_name] = model

        if not plugin_models:
            logging.info(
                "No models were defined by plugins. Using default model."
            )
            model = None
        else:
            first_mod_name = list(plugin_models.keys())[0]
            if len(plugin_models) > 1:
                logging.warning(
                    "More than one model was defined by plugins."
                    " Using the first one. (%s)",
                    first_mod_name,
                )
            model = list(plugin_models.values())[0]

    if model is None:
        logging.debug(f"Loading Percolator model.")
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
    try:
        main()
    except RuntimeError as e:
        logging.error(f"[Error] {e}")
        sys.exit(250)  # input failure
    except ValueError as e:
        logging.error(f"[Error] {e}")
        sys.exit(250)  # input failure
    except Exception as e:
        logging.error(f"[Error] {e}")
        sys.exit(252)
