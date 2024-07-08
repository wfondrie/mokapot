from __future__ import annotations
from typeguard import typechecked


@typechecked
def find_column(
    col: str, columns: list[str], required=True, unique=True, ignore_case=False
) -> str | list[str] | None:
    """
    Parameters
    ----------
    col : str
        The column name to search for.

    columns : list[str]
        The list of column names to search within.

    required : bool, optional
        Specifies whether the column is required. If set to True (default),
        an error will be raised if the column is not found.

    unique : bool, optional
        Specifies whether the column should be unique.
        If set to True (default), an error will be raised if multiple
        columns with the same name are found.

    ignore_case : bool, optional
        Specifies whether case should be ignored when comparing column names.
        If set to True, the comparison will be case-insensitive.

    Returns
    -------
    str or list[str] or None
        Returns the matched column name(s) based on the search criteria.

    Raises
    ------
    ValueError
        If the column is required and not found, or if multiple columns
            are found but unique is set to True.
    """
    if ignore_case:

        def str_compare(str1, str2):
            return str1.lower() == str2.lower()
    else:

        def str_compare(str1, str2):
            return str1 == str2

    found_columns = [c for c in columns if str_compare(c, col)]

    if required and len(found_columns) == 0:
        raise ValueError(f"The column '{col}' was not found.")

    if unique:
        if len(found_columns) > 1:
            raise ValueError(
                f"The column '{col}' should be unique. Found {found_columns}."
            )
        return found_columns[0] if len(found_columns) > 0 else None
    else:
        return found_columns


@typechecked
def find_columns(col: str, columns: list[str]) -> list[str]:
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
    return find_column(
        col, columns, required=False, unique=False, ignore_case=True
    )


@typechecked
def find_required_column(col: str, columns: list[str]) -> str:
    """
    Parameters
    ----------
    col : str
        The column to search for (case-insensitive).

    columns : list[str]
        The list of columns to search within.

    Returns
    -------
    str
        The required column found with correct case.

    Raises
    ------
    ValueError
        If the column was not found or not unique.
    """
    return find_column(
        col, columns, required=True, unique=True, ignore_case=True
    )


@typechecked
def find_optional_column(
    col: str | None, columns: list[str], default: str
) -> str | None:
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
    return find_column(
        col or default,
        columns,
        required=col is not None,
        unique=True,
        ignore_case=col is None,
    )
