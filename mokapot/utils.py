"""
Utility functions
"""

import itertools
import gzip
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from .constants import MERGE_SORT_CHUNK_SIZE
import pyarrow.parquet as pq
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


def get_next_row(file_handles, current_rows, col_index, sep="\t"):
    max_key = max_row = None
    max_score = None
    for key, row in current_rows.items():
        score = float(row[col_index])
        if max_score is None or max_score < score:
            max_score = score
            max_key = key
            max_row = row

    try:
        current_rows[max_key] = next(file_handles[max_key]).split(sep)

    except StopIteration:
        file_handles[max_key].close()
        del current_rows[max_key]
        del file_handles[max_key]
        return [max_row, max_key]

    return [max_row, max_key]


def get_row_from_batch(
    file_handles, current_batches, current_indices, max_indices, col_score
):
    max_key = None
    max_score = None
    for key, batch in current_batches.items():
        score = batch[current_indices[key]][col_score]
        if max_score is None or max_score < score:
            max_score = score
            max_key = key
    max_row = list(
        map(str, current_batches[max_key][current_indices[max_key]].values())
    )
    max_row[-1] += "\n"
    current_indices[max_key] += 1
    if current_indices[max_key] == max_indices[max_key]:
        try:
            current_batches[max_key] = next(file_handles[max_key]).to_pylist()
            current_indices[max_key] = 0
            max_indices[max_key] = len(current_batches[max_key])
        except StopIteration:
            del file_handles[max_key]
            del current_batches[max_key]
    return [max_row, max_key]


def merge_sort(paths, col_score, target_column=None, sep="\t"):

    if paths[0].suffix == ".parquet":
        file_handles = {
            i: pq.ParquetFile(path).iter_batches(MERGE_SORT_CHUNK_SIZE)
            for i, path in enumerate(paths)
        }
        current_batches = {}
        current_indices = [0] * len(paths)
        max_indices = [0] * len(paths)
        for key, file in file_handles.items():
            current_batches[key] = next(file).to_pylist()
            max_indices[key] = len(current_batches[key])
        while file_handles != {}:
            [row, key] = get_row_from_batch(
                file_handles,
                current_batches,
                current_indices,
                max_indices,
                col_score,
            )
            if row is not None:
                if target_column:
                    row.insert(1, str(key))
                yield row
    else:
        file_handles = {i: open(path, "r") for i, path in enumerate(paths)}
        current_rows = {}
        for key, f in file_handles.items():
            next(f)
            first_row = next(f)
            current_rows[key] = first_row.split(sep)

        with open(paths[0], "r") as f:
            header = next(f)
        col_index = header.rstrip().split(sep).index(col_score)
        while file_handles != {}:
            [row, key] = get_next_row(file_handles, current_rows, col_index)
            if row is not None:
                if target_column:
                    row.insert(1, str(key))
                yield row


@typechecked
def convert_targets_column(
    data: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    """Converts target column values to boolean
    (True if value is 1, False otherwise).

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
    if data[target_column].dtype == bool:
        return data

    labels = data[target_column].astype(int)
    if any(labels < -1) or any(labels > 1):
        raise ValueError(
            f"Invalid target column '{target_column}' "
            "contains values not in {-1, 0, 1}"
        )

    data[target_column] = labels == 1
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
        parameter is a tuple, the result will be a tuple. The order of the items
        in the result will be preserved.

    Raises
    ------
    ValueError
        If the search list/tuple contains a string that is not contained in
        `columns`
    """
    assert all(item is not None for item in search)
    if isinstance(search, dict):
        return {
            k: (
                columns.index(s)
                if isinstance(s, str)
                else map_columns_to_indices(s, columns)
            )
            for k, s in search.items()
        }
    else:
        return type(search)(
            (
                columns.index(s)
                if isinstance(s, str)
                else map_columns_to_indices(s, columns)
            )
            for s in search
        )
