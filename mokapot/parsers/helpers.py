from typing import Optional, List

from typeguard import typechecked


@typechecked
def find_columns(col: str, columns: List[str]) -> List[str]:
    """
    Parameters
    ----------
    col : str
        The column name to search for.

    columns : List[str]
        The list of columns to search within.

    Returns
    -------
    List[str]
        The list of columns that match the given column name, ignoring case
        sensitivity.
    """
    return [c for c in columns if c.lower() == col.lower()]


def find_column_new(col, columns, required=True, unique=True, ignore_case=False):
    if ignore_case:
        str_compare = lambda str1, str2: str1.lower() == str2.lower()
    else:
        str_compare = lambda str1, str2: str1 == str2
    found_columns = [c for c in columns if str_compare(c, col)]

    if required and not found_columns:
        raise ValueError(f"The column '{col}' was not found.")

    if unique and len(found_columns) > 1:
        raise ValueError(f"The column '{col}' should be unique. Found {found_columns}.")

    return found_columns[0] if unique else found_columns


@typechecked
def find_column(col: Optional[str], columns: List[str], default: str) -> Optional[str]:
    """
    Parameters
    ----------
    col : Optional[str]
        The column to check. If None, the default column will be searched in
        `columns`.

    columns : List[str]
        The list of available columns to check against.

    default : str
        The default column to search for if `col` is None.

    Returns
    -------
    Optional[str]:
        The validated column. If `col` is None, it returns the first matching
        column in `columns` with case-insensitive comparison to `default`. If
        `col` is not None, it returns `col` after ensuring it is present in
        `columns`.

    Raises
    ------
    ValueError
        If `col` is not None and it is not found in `columns`.
    """
    if col is None:
        try:
            return [c for c in columns if c.lower() == default.lower()][0]
        except IndexError:
            return None

    if col not in columns:
        raise ValueError(f"The '{col}' column was not found.")

    return col
