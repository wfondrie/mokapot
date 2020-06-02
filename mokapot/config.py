"""
Contains all of the configuration details for running mokapot
from the command line.
"""
import argparse
import textwrap


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
    desc = ("mokapot")
    docs = "blah"
    parser = argparse.ArgumentParser(description=desc, epilog=docs,
                                     formatter_class=MokapotHelpFormatter)

    parser.add_argument("-v", "--verbosity",
                        default=2,
                        type=int,
                        choices=[0, 1, 2, 3],
                        help=("Specify the verbosity of the current "
                              "process. Each level prints the following "
                              "messages, including all those at a lower "
                              "verbosity: 0-errors, 1-warnings, 2-messages"
                              ", 3-debug info."))

    parser.add_argument("pin_files", type=str, nargs="+",
                        help=("A collection of PSMs in Percolator input "
                              "format"))

    parser.add_argument("-o", "--output_dir", default=".", type=str,
                        help=("The directory in which to write the result "
                              "files"))

    parser.add_argument("-r", "--fileroot", type=str,
                        default="mokapot",
                        help="The prefix added to all file names.")

    parser.add_argument("-s", "--seed",
                        type=int,
                        default=1,
                        help=("An integer to use as the random seed."))

    return parser


def _process_line(line, width, indent):
    line = textwrap.fill(line, width, initial_indent=indent,
                         subsequent_indent=indent,
                         replace_whitespace=False)
    return line.strip()
