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


class Config():
    """
    The xenith configuration options.

    Options can be specified as command-line arguments.
    """
    def __init__(self) -> None:
        """Initialize configuration values."""
        self.parser = _parser()
        self._namespace = vars(self.parser.parse_args())

    def __getattr__(self, option):
        return self._namespace[option]


def _parser():
    """The parser"""
    desc = ("mokapot: Fast and flexible semi-supervised learning for "
            "peptide detection.\n" +
            ("="*80 + "\n") +
            f"Version {__version__}\n"
            "Official code website: https://github.com/wfondrie/mokapot\n\n"
            "More documentation and examples: https://mokapot.readthedocs.io")

    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=MokapotHelpFormatter)

    parser.add_argument("pin_files", type=str, nargs="+",
                        help=("A collection of PSMs in the Percolator tab-"
                              "delimited format."))

    parser.add_argument("-d", "--dest_dir", type=str,
                        help=("The directory in which to write the result "
                              "files. Defaults to the current working "
                              "directory"))

    parser.add_argument("-r", "--file_root", type=str,
                        help="The prefix added to all file names.")

    parser.add_argument("--train_fdr", default=0.01, type=float,
                        help=("The maximum false discovery rate at which to "
                              "consider a target PSM as a positive example "
                              "during model training."))

    parser.add_argument("--test_fdr", default=0.01, type=float,
                        help=("The false-discovery rate threshold at which to "
                              "evaluate the learned models."))

    parser.add_argument("--max_iter", default=10, type=int,
                        help=("The number of iterations to use for training."))

    parser.add_argument("--seed", type=int, default=1,
                        help=("An integer to use as the random seed."))

    parser.add_argument("--direction", type=str,
                        help=("The name of the feature to use as the initial "
                              "direction for ranking PSMs. The default "
                              "automatically selects the feature that finds "
                              "the most PSMs below the `train_fdr`."))

    parser.add_argument("--aggregate", default=False, action="store_true",
                        help=("If used, PSMs from multiple PIN files will be "
                              "aggregated and analyzed together. Otherwise, "
                              "a joint model will be trained, but confidence "
                              "estimates will be calculated separately for "
                              "each PIN file. This flag only has an effect "
                              "when multiple PIN files are provided."))

    parser.add_argument("--folds", type=int, default=3,
                        help=("The number of cross-validation folds to use. "
                              "PSMs originating from the same mass spectrum "
                              "are always in the same fold."))

    parser.add_argument("-v", "--verbosity",
                        default=2,
                        type=int,
                        choices=[0, 1, 2, 3],
                        help=("Specify the verbosity of the current "
                              "process. Each level prints the following "
                              "messages, including all those at a lower "
                              "verbosity: 0-errors, 1-warnings, 2-messages"
                              ", 3-debug info."))

    return parser


def _process_line(line, width, indent):
    line = textwrap.fill(line, width, initial_indent=indent,
                         subsequent_indent=indent,
                         replace_whitespace=False)
    return line.strip()
