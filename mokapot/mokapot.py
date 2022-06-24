"""This is the command line interface for mokapot"""
import sys
import time
import logging
import datetime
from functools import partial
from pathlib import Path

# import click
import numpy as np
import rich_click as click

from . import __version__
from .config import Config
from .parsers.pin import read_pin
from .parsers.pepxml import read_pepxml
from .parsers.fasta import read_fasta
from .brew import brew
from .model import PercolatorModel


@click.command(
    epilog="""
    Official code website: https://github.com/wfondrie/mokapot
    More documentation and examples: https://mokapot.readthedocs.io
    """,
    # context_setting=dict(help_option_names=['-h', '--help']),
)
@click.argument(
    "psm_files", type=click.Path(exists=True), nargs=-1, required=True
)
@click.option(
    "-d",
    "--dest_dir",
    type=click.Path(exists=True),
    help="""
    The directory in which to write the result files. Defaults to the current
    working directory.
    """,
)
@click.option(
    "-w",
    "--max_workers",
    default=1,
    type=int,
    help="""
    The number of processes to use for model training. Note that using more
    than one worker will disable some logging to the console.
    """,
)
@click.option(
    "-r",
    "--file_root",
    type=str,
    help="An optional prefix added to all output file names.",
)
@click.option(
    "--proteins",
    type=click.Path(exists=True),
    help="""
    The FASTA file used for the database search. Use this option to enable
    protein-level confidence estimates using the 'picked-protein' approach.
    Note that the FASTA file must contain both target and decoy sequence.
    Additionally, verify that the '--enzyme', '--missed_cleavages',
    '--min_length', '--max_length', '--semi', '--clip_nterm_methionine', and
    '--decoy_prefix' options match your search engine parameters.
    """,
)
@click.option(
    "--decoy_prefix",
    type=str,
    default="decoy_",
    help="""
    The prefix used to indicate a decoy protein in the FASTA file. For mokapot
    to provide accurate confidence estimates for proteins, decoy proteins
    should have the same description as the target proteins they were generated
    from, but with this string preprended.
    """,
)
@click.option(
    "--enzyme",
    default="[KR]",
    type=str,
    help="""
    A regular expression defining the enzyme specificity. The cleavage site is
    interpreted as the end of the match. The default is trypsin, without
    proline suppression.
    """,
)
@click.option(
    "--missed_cleavages",
    default=2,
    type=int,
    help="The allowed number of missed cleavages.",
)
@click.option(
    "--clip_nterm_methionine",
    is_flag=True,
    default=False,
    help="Remove methionine residues that occur at the protein N-terminus",
)
@click.option(
    "--min_length",
    default=6,
    type=int,
    help="The minimum peptide length that was considered.",
)
@click.option(
    "--max_length",
    default=50,
    help="The maximum peptide length that was considered.",
)
@click.option(
    "--semi",
    is_flag=True,
    default=False,
    help="""
    A semi-enzymatic digest was used to assigne PSMs. If this flag is used,
    the protein database will likely contain too many shared peptides to
    yield helpful protin-level confidence estimates. We do not recommend using
    this option.
    """,
)
@click.option(
    "--train_fdr",
    default=0.01,
    type=float,
    help="""
    The maximum false discovery rate at which to consider a target PSM as a
    positive example during model training.
    """,
)
@click.option(
    "--test_fdr",
    default=0.01,
    type=float,
    help="""
    The false-discovery rate threshold at which to evaluate the learned models.
    """,
)
@click.option(
    "--max_iter",
    default=10,
    type=int,
    help="The number of iterations to use for training.",
)
@click.option(
    "--seed",
    type=int,
    default=1,
    help="""
    An integer to use as a the random seed. This is used to separate PSMs into
    cross-validation folds and break ties between PSMs for the same spectrum.
    """,
)
@click.option(
    "--direction",
    type=str,
    help="""
    The name of the feature to use as the initial direction for ranking PSMs.
    The default automatically selects the feature that finds the most PSMs
    below the `train_fdr`.
    """,
)
@click.option(
    "--aggregate",
    is_flag=True,
    default=False,
    help="""
    Aggregate PSMs from multiple PSM files to be analyzed together. Otherwise,
    a joint model will be trained, but confidence estimates will be calculated
    separately for each PSM file. This flag only has an effect when multiple
    PSM files are provided. We recommend using this flag for fractionated
    samples and building spectral libraries.
    """,
)
@click.option(
    "--subset_max_train",
    type=int,
    help="""
    The maximum number of PSMs to use while training each cross-validation
    fold. This is useful for very large datasets and will be ignored if
    greater than the total number of PSMs.
    """,
)
@click.option(
    "--override",
    is_flag=True,
    default=False,
    help="""
    Use the learned model, even if it performs worse than the best feature.
    """,
)
@click.option(
    "--save_models",
    is_flag=True,
    default=False,
    help="""
    Save the models learned by mokapot as pickled Python objects.
    """,
)
@click.option(
    "--keep_decoys",
    is_flag=True,
    default=False,
    help="""
    Keep the decoys in the output .txt files. Note that, because mokapot used
    the decoys during model training within specific cross-validation folds,
    they are inappropriate to use for further FDR analysis.
    """,
)
@click.option(
    "--folds",
    default=3,
    type=int,
    help="""
    The number of cross-validation folds to use. PSMs originating from the same
    mass spectrum are always in the same fold.
    """,
)
@click.option(
    "--open_modification_bin_size",
    type=float,
    help="""
    This parameter only affects reading PSMs from PepXML files.
    If specified, modification masses are binned according to this value. The
    binned mass difference is appended to the end of the peptide and will be
    used when grouping peptides for peptide-level confidence estimation. Use
    this option for open modification search results. We recommend 0.01 as a
    good starting point.
    """,
)
@click.option(
    "-v",
    "--verbosity",
    default="2",
    type=click.Choice(["0", "1", "2", "3"]),
    help="""
    Specify the verbosity of the current process. Each level prints the
    following messages, including all those at a lower verbosity:
    0-errors, 1-warning, 2-messages, 3-debug info.
    """,
)
def mokapot(
    psm_files,
    dest_dir,
    max_workers,
    file_root,
    proteins,
    decoy_prefix,
    enzyme,
    missed_cleavages,
    clip_nterm_methionine,
    min_length,
    max_length,
    semi,
    train_fdr,
    test_fdr,
    max_iter,
    seed,
    direction,
    aggregate,
    subset_max_train,
    override,
    save_models,
    keep_decoys,
    folds,
    open_modification_bin_size,
    verbosity,
):
    """Assign confidence estimates to proteomics experiments.

    Mokapot uses a semi-supervised learning approach to enhance peptide
    detection from bottom-up proteomics experiments. It takes features
    describing putative peptide-spectrum matches (PSMs) from database search
    engines as input, re-scores them, and yields statistical
    measures—confidence estimates, such as q-values and posterior error
    probabilities—indicating their quality.

    PSM_FILES is one or more collections of PSMs in the Percolator
    tab-delimited or PepXML format.
    """
    start = time.time()

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
        level=verbosity_dict[int(verbosity)],
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

    np.random.seed(seed)

    # Parse Datasets
    parse = get_parser(psm_files, decoy_prefix, open_modification_bin_size)
    if aggregate or len(psm_files) == 1:
        datasets = parse(psm_files)
    else:
        datasets = [parse(f) for f in psm_files]
        prefixes = [Path(f).stem for f in psm_files]

    # Parse FASTA, if required:
    if proteins is not None:
        logging.info("Protein-level confidence estimates enabled.")
        proteins = read_fasta(
            proteins,
            enzyme=enzyme,
            missed_cleavages=missed_cleavages,
            clip_nterm_methionine=clip_nterm_methionine,
            min_length=min_length,
            max_length=max_length,
            semi=semi,
            decoy_prefix=decoy_prefix,
        )

        if aggregate or len(psm_files) == 1:
            datasets.add_proteins(proteins)
        else:
            for dataset in datasets:
                dataset.add_proteins(proteins)

    # Define a model:
    model = PercolatorModel(
        train_fdr=train_fdr,
        max_iter=max_iter,
        direction=direction,
        override=override,
        subset_max_train=subset_max_train,
    )

    # Fit the models:
    psms, models = brew(
        datasets,
        model=model,
        test_fdr=test_fdr,
        folds=folds,
        max_workers=max_workers,
    )

    if dest_dir is not None:
        Path(dest_dir).mkdir(exist_ok=True)

    if save_models:
        logging.info("Saving models...")
        for i, trained_model in enumerate(models):
            out_file = f"mokapot.model_fold-{i+1}.pkl"

            if file_root is not None:
                out_file = ".".join([file_root, out_file])

            if dest_dir is not None:
                out_file = Path(dest_dir, out_file)

            trained_model.save(str(out_file))

    # Determine how to write the results:
    logging.info("Writing results...")
    if aggregate or len(psm_files) == 1:
        psms.to_txt(
            dest_dir=dest_dir,
            file_root=file_root,
            decoys=keep_decoys,
        )
    else:
        for dat, prefix in zip(psms, prefixes):
            if file_root is not None:
                prefix = ".".join([file_root, prefix])

            dat.to_txt(
                dest_dir=dest_dir,
                file_root=prefix,
                decoys=keep_decoys,
            )

    total_time = round(time.time() - start)
    total_time = str(datetime.timedelta(seconds=total_time))

    logging.info("")
    logging.info("=== DONE! ===")
    logging.info("mokapot analysis completed in %s", total_time)


def get_parser(config, psm_files, decoy_prefix, open_modification_bin_size):
    """Figure out which parser to use.

    Note that this just looks at file extensions, but in the future it might be
    good to check the contents of the file. I'm just not sure how to do this
    in an efficient way, particularly for gzipped files.

    Parameters
    ----------
    psm_files : list of str
        The PSM files to parse.
    decoy_prefix : str
        The decoy prefix that is used.
    open_modification_bin_size : float
        The open modification bin size.

    Returns
    -------
    callable
         Returns the correct parser for the files.

    """
    pepxml_ext = {".pep.xml", ".pepxml", ".xml"}
    num_pepxml = 0
    for psm_file in psm_files:
        ext = Path(psm_file).suffixes
        if len(ext) > 2:
            ext = "".join(ext[-2:])
        else:
            ext = "".join(ext)

        if ext.lower() in pepxml_ext:
            num_pepxml += 1

    if num_pepxml == len(psm_files):
        return partial(
            read_pepxml,
            open_modification_bin_size=open_modification_bin_size,
            decoy_prefix=decoy_prefix,
        )

    return read_pin


if __name__ == "__main__":
    mokapot()
