from io import StringIO
from pathlib import Path

import pandas as pd


def is_traditional_pin(path: Path) -> bool:
    """Check if the PIN file is a traditional PIN file.

    The traditional PIN file uses tabs both as field delimiters
    and as the separator for multiple proteins in the last column.

    So it can be identified if:
    1. The last column name is `protein(s)?`.
    2. The header is tab delimited.
    3. The rest of the file is tab delimited.
    4. The number of delimiters in the other rows is >= number of columns.

    Parameters
    ----------
    path : Path
        The path to the PIN file.

    Returns
    -------
    bool
        True if the PIN file is a traditional PIN file
        (with ragged protein ends); False otherwise (readable as a tsv).

    Raises
    ------
    ValueError
        If the PIN file is not a PIN file. (or is corrupted)
    """
    with open(path) as f:
        nread = 0
        header = f.readline().strip()
        nread += 1
        header = header.split("\t")
        if len(header) == 1:
            raise ValueError(f"File '{path}' file is not a PIN file.")

        if not header[-1].lower().startswith("protein"):
            raise ValueError(
                f"File '{path}' is not a PIN file "
                f"(last column '{header[-1]}' is not 'protein')."
                " Which is expected from the traditional PIN file format."
            )

        num_fields = len(header)
        for line in f:
            nread += 1
            line = line.strip()
            if line.startswith("#") or line.startswith("DefaultDirection"):
                continue

            local_num_fields = len(line.split("\t"))
            if local_num_fields < num_fields:
                raise ValueError(
                    f"File '{path}' is not a PIN file: "
                    "The number of fields is less than number of columns"
                    f" on line {nread}, expected {num_fields} but "
                    f"got {local_num_fields}"
                )

            if local_num_fields > num_fields:
                return True

        return False


def read_traditional_pin(path) -> pd.DataFrame:
    """Reads the file in memory and bundles the proteins.

    The PIN file is assumed to be a traditional PIN file.
    The PIN file is read in memory and the proteins are bundled into a single
    column.

    Parameters
    ----------
    path : Path
        The path to the PIN file.

    Returns
    -------
    pd.DataFrame
        The PIN file as a pandas DataFrame.
    """
    header_names = None
    num_cols = None

    out_lines = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue

            if header_names is None:
                header_names = line.split("\t")
                num_cols = len(header_names)
                continue

            line_data = line.split("\t")
            line_data = line_data[: num_cols - 1] + [
                ":".join(line_data[num_cols - 1 :])
            ]
            if len(line_data) != num_cols:
                raise RuntimeError(
                    "Error parsing PIN file. "
                    f" Line: {line}"
                    f" Expected: {num_cols} columns"
                )
            out_lines.append("\t".join(line_data))

    with StringIO("\n".join(out_lines)) as f:
        df = pd.read_csv(f, sep="\t", header=None)
        df.columns = header_names
        return df
