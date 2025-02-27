"""
Utility functions
"""

import gzip
import itertools
from pathlib import Path
from typing import Any, Dict, Iterator, NewType, Union

import numpy as np
import pandas as pd
from typeguard import typechecked


@typechecked
def open_file(file_name: Path):
    if file_name.suffix == ".gz":
        return gzip.open(file_name)
    else:
        return open(file_name)


def groupby_max(df, by_cols, max_col, rng):
    """Quickly get the indices for the maximum value of col"""
    by_cols = tuplize(by_cols)
    idx = (
        df.sample(frac=1, random_state=rng)
        .sort_values(list(by_cols) + [max_col], axis=0)
        .drop_duplicates(list(by_cols), keep="last")
        .index
    )
    return idx


def flatten(split):
    """Get the indices from split"""
    return list(itertools.chain.from_iterable(split))


def safe_divide(numerator, denominator, ones=False):
    """Divide ignoring div by zero warnings"""
    if isinstance(numerator, pd.Series):
        numerator = numerator.values

    if isinstance(denominator, pd.Series):
        denominator = denominator.values

    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    if ones:
        out = np.ones_like(numerator)
    else:
        out = np.zeros_like(numerator)

    return np.divide(numerator, denominator, out=out, where=(denominator != 0))


def tuplize(obj) -> tuple:
    """Convert obj to a tuple, without splitting strings"""
    try:
        _ = iter(obj)
    except TypeError:
        obj = (obj,)
    else:
        if isinstance(obj, str):
            obj = (obj,)

    return tuple(obj)


@typechecked
def create_chunks(
    data: Union[list, np.array], chunk_size: int
) -> list[Union[list, np.array]]:
    """
    Splits the given data into chunks of the specified size.

    Parameters
    ----------
    data : Union[list, np.array]
        The input data to be split into chunks.

    chunk_size : int
        The size of each individual chunk.

    Returns
    -------
    list[Union[list, np.array]]
        A list containing sublists, where each sublist is a chunk of the input
        data.

    """
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


# Using Dict over dict is needed while we support Python 3.8
DataRow = NewType("DataRow", Dict[str, Any])


@typechecked
def get_next_row(
    row_iterator_dict: dict[int, Iterator[DataRow]],
    current_row_dict: dict[int, DataRow],
    score_column: str,
) -> DataRow:
    max_key = max_row = None
    max_score = None
    for key, row in current_row_dict.items():
        score = float(row[score_column])
        if max_score is None or max_score < score:
            max_score = score
            max_key = key
            max_row = row

    try:
        current_row_dict[max_key] = next(row_iterator_dict[max_key])
    except StopIteration:
        del current_row_dict[max_key]
        del row_iterator_dict[max_key]

    return max_row


@typechecked
def make_bool_trarget(target_column: pd.Series):
    """Convert target column to boolean if possible.

    1. If its a 0-1 col convert to bool
    2. If its a -1,1 values > 0 becomes True, rest False
    """
    if target_column.dtype == bool:
        return target_column

    if target_column.dtype == object:
        try:
            target_column = target_column.astype(int)
        except ValueError as e:
            raise ValueError(
                "The target column is not convertible to integers."
            ) from e

    if target_column.dtype == int or target_column.dtype == float:
        # Check if all values are 0 or 1
        uniq_vals = target_column.unique().tolist()
        uniq_vals.sort()
        if uniq_vals == [0, 1]:
            # If so, we can just cast to bool
            return target_column.astype(bool)
        elif uniq_vals == [-1, 1]:
            return target_column > 0
        else:
            raise ValueError(
                f"Target column "
                "has values that are not boolean or 0/1 or -1/1."
                f"Please check and fix. (found {uniq_vals})"
            )

    # If not raise an error ... most likely the cast will be
    # Something the user does not want.
    raise ValueError(
        "Target column "
        "has values that are not boolean, 0/1 or -1/1,"
        f" please check and fix. ({target_column})"
        f" of dtype {target_column.dtype}"
    )


def convert_targets_column(
    data: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    """Converts target column to boolean.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the target column to be converted (will be
        modified in-place).
    target_column : str
        The name of the target column in the DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the target column converted to boolean.

    Raises
    ------
    ValueError
        If the target column contains values other than -1, 0, or 1.
    """
    data[target_column] = make_bool_trarget(data[target_column])
    return data


@typechecked
def map_columns_to_indices(
    search: list | tuple | dict, columns: list[str]
) -> list | tuple | dict:
    """
    Map columns to indices in recursive fashion preserving order and structure.

    Parameters
    ----------
    search : list | tuple
        The list or tuple of search items to map to indices. It can contain
        strings or nested lists/tuples of search items.

    columns : list[str]
        The list of columns in which to search for the items. This must be a
        list of strings.

    Returns
    -------
    list | tuple
        The result of the mapping, with the same structure as the `search`
        parameter but with indices instead of the search items. If the `search`
        parameter is a list, the result will be a list as well. If the `search`
        parameter is a tuple, the result will be a tuple. The order of the
        items in the result will be preserved.

    Raises
    ------
    ValueError
        If the search list/tuple contains a string that is not contained in
        `columns`
    """
    assert all(item is not None for item in search)
    if isinstance(search, dict):
        result = {
            k: (
                columns.index(s)
                if isinstance(s, str)
                else map_columns_to_indices(s, columns)
            )
            for k, s in search.items()
        }
    else:
        result = type(search)(
            (
                columns.index(s)
                if isinstance(s, str)
                else map_columns_to_indices(s, columns)
            )
            for s in search
        )
    return result


def strictzip(*iterables):
    """Strict zip.

    Backport of zip(strict=True) for pre-3.10 python.

    Derived from: https://peps.python.org/pep-0618/
    """
    if not iterables:
        return
    iterators = tuple(iter(iterable) for iterable in iterables)
    try:
        while True:
            items = []
            for iterator in iterators:
                items.append(next(iterator))
            yield tuple(items)
    except StopIteration:
        pass

    if items:
        i = len(items)
        plural = " " if i == 1 else "s 1-"
        msg = f"zip() argument {i + 1} is shorter than argument{plural}{i}"
        raise ValueError(msg)
    sentinel = object()
    for i, iterator in enumerate(iterators[1:], 1):
        if next(iterator, sentinel) is not sentinel:
            plural = " " if i == 1 else "s 1-"
            msg = f"zip() argument {i + 1} is longer than argument{plural}{i}"
            raise ValueError(msg)
