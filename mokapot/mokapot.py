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
from .parsers.pin import read_pin, read_data_for_rescale
from .parsers.pepxml import read_pepxml
from .parsers.fasta import read_fasta
from .brew import brew
from .model import PercolatorModel, load_model
from .confidence import assign_confidence
from .plugins import get_plugins


def main():
    """The CLI entry point"""
    start = time.time()
    plugins = get_plugins()

    # Get command line arguments
    parser = Config().parser
    for plugin_name, plugin in plugins.items():
        parsergroup = parser.add_argument_group(plugin_name)
        plugin.add_arguments(parsergroup)

    config = Config(parser)

    # Setup logging
    verbosity_dict = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    if verbosity_dict[config.verbosity] != logging.DEBUG:
        warnings.filterwarnings("ignore")
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
    logging.debug("Loaded plugins: %s", plugins.keys())

    np.random.seed(config.seed)

    # Parse Datasets
    parse = get_parser(config)
    enabled_plugins = {p: plugins[p]() for p in config.plugin}

    datasets = parse(config.psm_files)
    if config.aggregate or len(config.psm_files) == 1:
        for plugin in enabled_plugins.values():
            datasets = plugin.process_data(datasets, config)
        prefixes = [None for f in config.psm_files]
    else:
        for plugin in enabled_plugins.values():
            datasets = [plugin.process_data(ds, config) for ds in datasets]
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
    else:
        proteins = None

    # Define a model:
    model = None
    if config.load_models:
        data_to_rescale = None
        if config.rescale:
            data_to_rescale = read_data_for_rescale(
                psms=datasets,
                subset_max_rescale=config.subset_max_rescale,
            )
        model = [
            load_model(model_file, data_to_rescale)
            for model_file in config.load_models
        ]
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
        Path(config.dest_dir).mkdir(exist_ok=True)
    deduplication = not config.skip_deduplication
    if config.file_root is not None:
        config.dest_dir = f"{Path(config.dest_dir, config.file_root)}."
    assign_confidence(
        psms=psms,
        scores=scores,
        eval_fdr=config.test_fdr,
        descs=desc,
        dest_dir=config.dest_dir,
        decoys=config.keep_decoys,
        deduplication=deduplication,
        proteins=proteins,
        prefixes=prefixes,
        rng=config.seed,
    )

    if config.save_models:
        logging.info("Saving models...")
        for i, trained_model in enumerate(models):
            out_file = f"mokapot.model_fold-{i+1}.pkl"

            if config.file_root is not None:
                out_file = ".".join([config.file_root, out_file])

            if config.dest_dir is not None:
                out_file = Path(config.dest_dir, out_file)

            trained_model.save(str(out_file))

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
    try:
        main()
    except RuntimeError as e:
        logging.error(f"[Error] {e}")
        sys.exit(250)  # input failure
    except ValueError as e:
        logging.error(f"[Error] {e}")
        sys.exit(250)  # input failure
