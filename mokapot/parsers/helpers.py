from typing import Optional, List


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
