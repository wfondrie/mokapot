"""
Contains all of the configuration details for running mokapot
from the command line.
"""
import argparse
import textwrap

from mokapot import __version__


class MokapotHelpFormatter(argparse.HelpFormatter):
    """Format help text to keep newlines and whitespace"""

    def _fill_text(self, text, width, indent):
        text_list = text.splitlines(keepends=True)
        return "\n".join(_process_line(l, width, indent) for l in text_list)


class Config:
    """
    The mokapot configuration options.

    Options can be specified as command-line arguments.
    """

    def __init__(self, parser=None) -> None:
        """Initialize configuration values."""
        self._namespace = None
        if parser is None:
            self.parser = _parser()
        else:
            self.parser = parser

    @property
    def args(self):
        """Collect args lazily."""
        if self._namespace is None:
            self._namespace = vars(self.parser.parse_args())

        return self._namespace

    def __getattr__(self, option):
        return self.args[option]


def _parser():
    """The parser"""
    desc = (
        f"mokapot version {__version__}.\n"
        "Written by William E. Fondrie (wfondrie@talus.bio) while in the \n"
        "Department of Genome Sciences at the University of Washington.\n\n"
        "Official code website: https://github.com/wfondrie/mokapot\n\n"
        "More documentation and examples: https://mokapot.readthedocs.io"
    )

    parser = argparse.ArgumentParser(
        description=desc, formatter_class=MokapotHelpFormatter
    )

    parser.add_argument(
        "psm_files",
        type=str,
        nargs="+",
        help=(
            "A collection of PSMs in the Percolator tab-delimited or PepXML "
            "format."
        ),
    )

    parser.add_argument(
        "-d",
        "--dest_dir",
        type=str,
        help=(
            "The directory in which to write the result files. Defaults to "
            "the current working directory"
        ),
    )

    parser.add_argument(
        "-w",
        "--max_workers",
        default=1,
        type=int,
        help=(
            "The number of processes to use for model training. Note that "
            "using more than one worker will result in garbled logging "
            "messages."
        ),
    )

    parser.add_argument(
        "-r",
        "--file_root",
        type=str,
        help="The prefix added to all file names.",
    )

    parser.add_argument(
        "--proteins",
        type=str,
        help=(
            "The FASTA file used for the database search. Using this "
            "option enable protein-level confidence estimates using "
            "the 'picked-protein' approach. Note that the FASTA file "
            "must contain both target and decoy sequences. "
            "Additionally, verify that the '--enzyme', "
            "'--missed_cleavages, '--min_length', '--max_length', "
            "'--semi', '--clip_nterm_methionine', and '--decoy_prefix' "
            "parameters match your search engine conditions."
        ),
    )

    parser.add_argument(
        "--decoy_prefix",
        type=str,
        default="decoy_",
        help=(
            "The prefix used to indicate a decoy protein in the "
            "FASTA file. For mokapot to provide accurate confidence "
            "estimates, decoy proteins should have same description "
            "as the target proteins they were generated from, but "
            "this string prepended."
        ),
    )

    parser.add_argument(
        "--enzyme",
        type=str,
        default="[KR]",
        help=(
            "A regular expression defining the enzyme specificity. "
            "The cleavage site is interpreted as the end of the match. "
            "The default is trypsin, without proline suppression: [KR]"
        ),
    )

    parser.add_argument(
        "--missed_cleavages",
        type=int,
        default=2,
        help="The allowed number of missed cleavages",
    )

    parser.add_argument(
        "--clip_nterm_methionine",
        default=False,
        action="store_true",
        help="Remove methionine residues that occur at the protein N-terminus.",
    )

    parser.add_argument(
        "--min_length",
        type=int,
        default=6,
        help="The minimum peptide length to consider.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="The maximum peptide length to consider.",
    )

    parser.add_argument(
        "--semi",
        default=False,
        action="store_true",
        help=(
            "Was a semi-enzymatic digest used to assign PSMs? If"
            " so, the protein database will likely contain "
            "shared peptides and yield unhelpful protein-level confidence "
            "estimates. We do not recommend using this option."
        ),
    )

    parser.add_argument(
        "--train_fdr",
        default=0.01,
        type=float,
        help=(
            "The maximum false discovery rate at which to "
            "consider a target PSM as a positive example "
            "during model training."
        ),
    )

    parser.add_argument(
        "--test_fdr",
        default=0.01,
        type=float,
        help=(
            "The false-discovery rate threshold at which to "
            "evaluate the learned models."
        ),
    )

    parser.add_argument(
        "--max_iter",
        default=10,
        type=int,
        help=("The number of iterations to use for training."),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help=("An integer to use as the random seed."),
    )

    parser.add_argument(
        "--direction",
        type=str,
        help=(
            "The name of the feature to use as the initial "
            "direction for ranking PSMs. The default "
            "automatically selects the feature that finds "
            "the most PSMs below the `train_fdr`."
        ),
    )

    parser.add_argument(
        "--aggregate",
        default=False,
        action="store_true",
        help=(
            "If used, PSMs from multiple PIN files will be "
            "aggregated and analyzed together. Otherwise, "
            "a joint model will be trained, but confidence "
            "estimates will be calculated separately for "
            "each PIN file. This flag only has an effect "
            "when multiple PIN files are provided."
        ),
    )

    parser.add_argument(
        "--subset_max_train",
        type=int,
        default=None,
        help=(
            "Maximum number of PSMs to use during the training "
            "of each of the cross validation folds in the model. "
            "This is useful for very large datasets and will be "
            "ignored if less PSMS are available."
        ),
    )

    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
        help=(
            "Use the learned model even if it performs worse "
            "than the best feature."
        ),
    )

    parser.add_argument(
        "--save_models",
        default=False,
        action="store_true",
        help=("Save the models learned by mokapot as pickled Python objects."),
    )

    parser.add_argument(
        "--load_models",
        type=str,
        nargs="+",
        help=(
            "Load previously saved models and skip model training."
            "Note that the number of models must match the value of --folds."
        ),
    )

    parser.add_argument(
        "--plugin",
        type=str,
        action="append",
        default=[],
        help=("The names of the plugins to use."),
    )

    parser.add_argument(
        "--keep_decoys",
        default=False,
        action="store_true",
        help=("Keep the decoys in the output .txt files"),
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help=(
            "The number of cross-validation folds to use. "
            "PSMs originating from the same mass spectrum "
            "are always in the same fold."
        ),
    )

    parser.add_argument(
        "--open_modification_bin_size",
        type=float,
        help=(
            "This parameter only affect reading PSMs from PepXML files. "
            "If specified, modification masses are binned according to the "
            "value. The binned mass difference is appended to the end of the "
            "peptide and will be used when grouping peptides for peptide-level"
            " confidence estimation. Using this option for open modification "
            "search results. We recommend 0.01 as a good starting point."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        default=2,
        type=int,
        choices=[0, 1, 2, 3],
        help=(
            "Specify the verbosity of the current "
            "process. Each level prints the following "
            "messages, including all those at a lower "
            "verbosity: 0-errors, 1-warnings, 2-messages"
            ", 3-debug info."
        ),
    )

    return parser


def _process_line(line, width, indent):
    line = textwrap.fill(
        line,
        width,
        initial_indent=indent,
        subsequent_indent=indent,
        replace_whitespace=False,
    )
    return line.strip()
