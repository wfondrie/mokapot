"""The command line interface for mokapot."""
import datetime
import logging
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__, to_parquet, to_txt, utils
from .brew import brew
from .config import Config
from .model import PercolatorModel, load_model
from .parsers.fasta import read_fasta
from .parsers.pin import read_pin
from .plugins import get_plugins


def main() -> None:  # noqa: C901
    """The CLI entry point."""
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

    logging.basicConfig(
        format=("[{levelname}] {message}"),
        style="{",
        level=verbosity_dict[config.verbosity],
    )

    logging.info("mokapot version %s", str(__version__))
    logging.info("Command issued:")
    logging.info("%s", " ".join(sys.argv))
    logging.info("")
    logging.info("Starting Analysis")
    logging.info("=================")
    logging.debug("Loaded plugins: %s", plugins.keys())

    rng = np.random.default_rng(config.seed)
    enabled_plugins = {p: plugins[p]() for p in config.plugin}

    # Parse Datasets
    if config.aggregate or len(config.psm_files) == 1:
        prefixes = [""]
        datasets = utils.listify(
            read_pin(
                config.psm_files,
                rng=rng,
                subset=config.subset_max_train,
            )
        )
    else:
        datasets = [
            read_pin(f, rng=rng, subset=config.subset_max_train)
            for f in config.psm_files
        ]
        prefixes = [Path(f).stem for f in config.psm_files]

    for plugin in enabled_plugins.values():
        datasets = [plugin.process_data(ds, config) for ds in datasets]

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
            rng=rng,
        )
        for dataset in datasets:
            dataset.proteins = proteins

    # Define a model:
    model = None
    if config.load_models:
        model = [load_model(model_file) for model_file in config.load_models]
    elif enabled_plugins:
        plugin_models = {}
        for plugin_name, plugin in enabled_plugins.items():
            model = plugin.get_model(config)
            if model is not None:
                logging.debug("Loaded model for %s", plugin_name)
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
        logging.debug("Using Percolator model.")
        model = PercolatorModel(
            train_fdr=config.train_fdr,
            max_iter=config.max_iter,
            override=config.override,
            rng=rng,
        )

    # Fit the models:
    psms, models = brew(
        datasets,
        model=model,
        folds=config.folds,
        max_workers=config.max_workers,
        rng=rng,
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
    writer = to_parquet if config.parquet else to_txt
    for dat, prefix in zip(utils.listify(psms), prefixes):
        if config.file_root is not None:
            if prefix:
                prefix = ".".join([config.file_root, prefix])
            else:
                prefix = config.file_root

        writer(
            dat,
            dest_dir=config.dest_dir,
            stem=prefix,
            decoys=config.keep_decoys,
        )

    total_time = round(time.time() - start)
    total_time = str(datetime.timedelta(seconds=total_time))

    logging.info("")
    logging.info("=== DONE! ===")
    logging.info("mokapot analysis completed in %s", total_time)


if __name__ == "__main__":
    main()
