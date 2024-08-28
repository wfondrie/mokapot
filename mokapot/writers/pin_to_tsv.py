from pathlib import Path
from io import StringIO
from typing import TextIO
from unittest.mock import Mock

import argparse

# PIN file specification from
# https://github.com/percolator/percolator/wiki/Interface#tab-delimited-file-format
"""
PSMId <tab> Label <tab> ScanNr <tab> feature1name <tab> ... <tab> featureNname <tab> Peptide <tab> Proteins
DefaultDirection <tab> - <tab> - <tab> feature1weight <tab> ... <tab> featureNweight [optional]
"""

EXAMPLE_PIN = """SpecId\tLabel\tScanNr\tExpMass\tPeptide\tProteins
target_0_16619_2_-1\t1\t16619\t750.4149\tK.SEFLVR.E\tsp|Q96QR8|PURB_HUMAN\tsp|Q00577|PURA_HUMAN
target_0_2025_2_-1\t1\t2025\t751.4212\tR.HTALGPR.S\tsp|Q9Y4H4|GPSM3_HUMAN"""
EXAMPLE_HEADER, EXAMPLE_LINE_1, EXAMPLE_LINE_2 = EXAMPLE_PIN.split('\n')

PIN_SEP = '\t'


def parse_pin_header_columns(
        header: str,
        sep_column: str = PIN_SEP,

) -> (int, int):
    """
    Parse the header of a PIN file to get the number of columns and the index of the
    Proteins column.

    Parameters
    ----------
    header : str
        The header line from the PIN file.
    sep_column : str, optional
        Column separator (default is PIN_SEP).

    Returns
    -------
    n_col : int
        The total number of columns in the PIN file.
    idx_protein_col : int
        The index of the 'Proteins' column.

    Examples
    --------
    >>> n_col, idx_protein_col = parse_pin_header_columns(EXAMPLE_HEADER)
    >>> n_col, idx_protein_col
    (6, 5)
    """
    columns = header.split(sep_column)
    assert "Proteins" in columns
    n_col = len(columns)
    idx_protein_col = columns.index("Proteins")
    return n_col, idx_protein_col


def convert_line_pin_to_tsv(
        line: str,
        idx_protein_col: int,
        n_col: int,
        sep_column: str = "\t",
        sep_protein: str = ":"
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
        The total number of columns in the PIN file (excluding additional protein columns).
    sep_column : str, optional
        The separator used between columns (default is "\t").
    sep_protein : str, optional
        The separator to use between multiple proteins (default is ":").

    Returns
    -------
    str
        The converted line in TSV format.

    Examples
    --------
    >>> header = EXAMPLE_HEADER
    >>> n_col, idx_protein_col = parse_pin_header_columns(header)
    >>> tsv_line = convert_line_pin_to_tsv(EXAMPLE_LINE_1, n_col=n_col, idx_protein_col=idx_protein_col)
    >>> tsv_line.expandtabs(4)  # needed for docstring to work
    'target_0_16619_2_-1 1   16619   750.4149    K.SEFLVR.E  sp|Q96QR8|PURB_HUMAN:sp|Q00577|PURA_HUMAN'
    >>> tsv_line = convert_line_pin_to_tsv(EXAMPLE_LINE_2, n_col=n_col, idx_protein_col=idx_protein_col)
    >>> tsv_line.expandtabs(4)  # needed for docstring to work
    'target_0_2025_2_-1  1   2025    751.4212    R.HTALGPR.S sp|Q9Y4H4|GPSM3_HUMAN'
    """
    elements = line.split(sep=sep_column)  # this contains columns and proteins
    n_proteins = len(elements) - n_col
    idx_prot_start = idx_protein_col
    idx_prot_end = idx_protein_col + n_proteins + 1
    proteins: str = sep_protein.join(elements[idx_prot_start:idx_prot_end])
    columns: list = elements[:idx_prot_start] + [proteins] + elements[idx_prot_end:]
    tsv_line: str = sep_column.join(columns)
    return tsv_line


def is_valid_tsv(
        f_in: TextIO,
        sep_column: str = PIN_SEP
) -> bool:
    """
    This function verifies that:
    1. All rows have the same number of columns as the header row.
    2. The file does not contain a "DefaultDirection" line as the second line.

    Parameters
    ----------
    f_in : TextIO
        Input file object to read from. This should be an opened file or file-like
        object that supports iteration.
    sep_column : str, optional
        Column separator (default is PIN_SEP, which is assumed to be a tab character).

    Returns
    -------
    bool
        True if the file is a valid TSV according to the specified criteria,
        False otherwise.

    Examples
    --------
    >>> input = StringIO(EXAMPLE_PIN)
    >>> is_valid_tsv(input)
    False
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
        f_in: TextIO,
        f_out: TextIO,
        sep_column: str = PIN_SEP,
        sep_protein: str = ":"
) -> None:
    """
    Convert a PIN file to a valid TSV file.

    This assumes that the input file is in PIN format and that the first line
    is a header. It preserves the header in the output file and ignores the second line
    if it starts with "DefaultDirection".

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

    Examples
    --------
    >>> mock_input = StringIO(EXAMPLE_PIN)
    >>> mock_output = Mock()
    >>> mock_output.write = Mock()
    >>> pin_to_valid_tsv(mock_input, mock_output)
    >>> mock_output.write.call_count
    3
    >>> mock_output.write.assert_any_call(EXAMPLE_HEADER + "\\n")
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
            sep_protein=sep_protein
        )
        f_out.write(tsv_line + "\n")

    for line in f_in:
        line = line.strip()
        tsv_line: str = convert_line_pin_to_tsv(
            line,
            n_col=n_col,
            idx_protein_col=idx_protein_col,
            sep_column=sep_column,
            sep_protein=sep_protein
        )
        f_out.write(tsv_line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert PIN file to valid TSV")
    parser.add_argument("path_in", type=Path, help="Input PIN file path")
    parser.add_argument("path_out", type=Path, help="Output TSV file path")
    parser.add_argument("--sep_column", type=str, default="\t",
                        help="Column separator (default: '\\t')")
    parser.add_argument("--sep_protein", type=str, default=":",
                        help="Protein separator (default: ':')")
    args = parser.parse_args()
    with open(args.path_in, 'r') as f_in:
        with open(args.path_out, 'a') as f_out:
            pin_to_valid_tsv(
                f_in=f_in,
                f_out=f_out,
                sep_column=args.sep_column,
                sep_protein=args.sep_protein
            )


if __name__ == "__main__":
    main()
