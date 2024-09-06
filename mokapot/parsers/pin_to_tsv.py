"""Parse PIN files."""

from pathlib import Path
from typing import TextIO
import argparse


# PIN file specification from
# https://github.com/percolator/percolator/wiki/Interface#tab-delimited-file-format
def parse_pin_header_columns(
    header: str,
    sep_column: str = "\t",
) -> (int, int):
    """Parse the PIN file header.

    Parse the header of a PIN file to get the number of columns and the
    index of the Proteins column.

    Parameters
    ----------
    header : str
        The header line from the PIN file.
    sep_column : str, optional
        Column separator.

    Returns
    -------
    n_col : int
        The total number of columns in the PIN file.
    idx_protein_col : int
        The index of the 'Proteins' column.

    """
    columns = header.strip().split(sep_column)
    assert "Proteins" in columns
    n_col = len(columns)
    idx_protein_col = columns.index("Proteins")
    return n_col, idx_protein_col


def convert_line_pin_to_tsv(
    line: str,
    idx_protein_col: int,
    n_col: int,
    sep_column: str = "\t",
    sep_protein: str = ":",
):
    """
    Convert a single line from a PIN file format to a TSV format.

    Parameters
    ----------
    line : str
        A single line from the PIN file.
    idx_protein_col : int
        The index of the first protein column.
    n_col : int
        The total number of columns in the PIN file (excluding additional
        protein columns).
    sep_column : str, optional
        The separator used between columns (default is "\t").
    sep_protein : str, optional
        The separator to use between multiple proteins (default is ":").

    Returns
    -------
    str
        The converted line in TSV format.
    """
    elements = line.split(sep=sep_column)  # this contains columns and proteins
    n_proteins = len(elements) - n_col
    idx_prot_start = idx_protein_col
    idx_prot_end = idx_protein_col + n_proteins + 1
    proteins: str = sep_protein.join(elements[idx_prot_start:idx_prot_end])
    columns: list = (
        elements[:idx_prot_start] + [proteins] + elements[idx_prot_end:]
    )
    tsv_line: str = sep_column.join(columns)
    return tsv_line


def is_valid_tsv(
    f_in: TextIO,
    sep_column: str = "\t",
) -> bool:
    """Verify that a file is a valid TSV.

    This function verifies that:
    1. All rows have the same number of columns as the header row.
    2. The file does not contain a "DefaultDirection" line as the second line.

    Parameters
    ----------
    f_in : TextIO
        Input file object to read from. This should be an opened file or
        file-like bject that supports iteration.
    sep_column : str, optional
        Column separator

    Returns
    -------
    bool
        True if the file is a valid TSV according to the specified criteria,
        False otherwise.

    """
    n_col_header = len(next(f_in).split(sep_column))
    line_2 = next(f_in)

    # check for optional DefaultDirection line
    if line_2.startswith("DefaultDirection"):
        return False
    n_col = len(line_2.split(sep_column))
    if n_col != n_col_header:
        return False

    # check if sep_column is really only used for columns
    for line in f_in:
        n_col = len(line.split(sep_column))
        if n_col != n_col_header:
            return False
    return True


def pin_to_valid_tsv(
    f_in: TextIO, f_out: TextIO, sep_column: str = "\t", sep_protein: str = ":"
) -> None:
    """Convert a PIN file to a valid TSV file.

    This assumes that the input file is in PIN format and that the first line
    is a header. It preserves the header in the output file and ignores the
    second line if it starts with "DefaultDirection".

    Parameters
    ----------
    f_in : TextIO
        Input file object to read from.
    f_out : TextIO
        Output file object to write to.
    sep_column : str, optional
        Column separator (default is PIN_SEP).
    sep_protein : str, optional
        Protein separator (default is ":").

    Returns
    -------
    None
    """
    header: str = next(f_in).strip()
    f_out.write(header + "\n")
    n_col, idx_protein_col = parse_pin_header_columns(header, sep_column)

    # Optionally, the second line of a PIN file might declare DefaultDirection
    # This is ignored with this conversion
    # https://github.com/percolator/percolator/wiki/Interface#pintsv-tab-delimited-file-format
    second_line = next(f_in).strip()

    if not second_line.startswith("DefaultDirection"):
        tsv_line: str = convert_line_pin_to_tsv(
            second_line,
            n_col=n_col,
            idx_protein_col=idx_protein_col,
            sep_column=sep_column,
            sep_protein=sep_protein,
        )
        f_out.write(tsv_line + "\n")

    for line in f_in:
        line = line.strip()
        tsv_line: str = convert_line_pin_to_tsv(
            line,
            n_col=n_col,
            idx_protein_col=idx_protein_col,
            sep_column=sep_column,
            sep_protein=sep_protein,
        )
        f_out.write(tsv_line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PIN file to valid TSV"
    )
    parser.add_argument("path_in", type=Path, help="Input PIN file path")
    parser.add_argument("path_out", type=Path, help="Output TSV file path")
    parser.add_argument(
        "--sep_column",
        type=str,
        default="\t",
        help="Column separator (default: '\\t')",
    )
    parser.add_argument(
        "--sep_protein",
        type=str,
        default=":",
        help="Protein separator (default: ':')",
    )
    args = parser.parse_args()
    with open(args.path_in, "r") as f_in:
        with open(args.path_out, "a") as f_out:
            pin_to_valid_tsv(
                f_in=f_in,
                f_out=f_out,
                sep_column=args.sep_column,
                sep_protein=args.sep_protein,
            )


if __name__ == "__main__":
    main()
