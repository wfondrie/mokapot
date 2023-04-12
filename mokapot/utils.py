"""
Utility functions
"""
import itertools
import gzip

import numpy as np
import pandas as pd


def open_file(file_name):
    if str(file_name).endswith(".gz"):
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


def tuplize(obj):
    """Convert obj to a tuple, without splitting strings"""
    try:
        _ = iter(obj)
    except TypeError:
        obj = (obj,)
    else:
        if isinstance(obj, str):
            obj = (obj,)

    return tuple(obj)


def create_chunks(data, chunk_size):
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def get_unique_psms_and_peptides(iterable, out_psms, out_peptides, sep):
    seen_psm = set()
    seen_peptide = set()
    f_psm = open(out_psms, "a")
    f_peptide = open(out_peptides, "a")
    for line_list in iterable:
        line_hash_psm = tuple([int(line_list[2]), float(line_list[3])])
        line_hash_peptide = line_list[-3]
        if line_hash_psm not in seen_psm:
            seen_psm.add(line_hash_psm)
            f_psm.write(
                f"{sep.join([line_list[0], line_list[1], line_list[-3], line_list[-2], line_list[-1]])}"
            )
            if line_hash_peptide not in seen_peptide:
                seen_peptide.add(line_hash_peptide)
                f_peptide.write(
                    f"{sep.join([line_list[0], line_list[1], line_list[-3], line_list[-2], line_list[-1]])}"
                )
    f_psm.close()
    f_peptide.close()
    return [len(seen_psm), len(seen_peptide)]


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
    except:
        file_handles[max_key].close()
        del current_rows[max_key]
        del file_handles[max_key]
        return max_row

    return max_row


def merge_sort(paths, col_score, sep="\t"):
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
        row = get_next_row(file_handles, current_rows, col_index)
        if row is not None:
            yield row


def convert_targets_column(data, target_column):
    data[target_column] = data[target_column].astype(int)
    if any(data[target_column] == -1):
        data[target_column] = ((data[target_column] + 1) / 2).astype(bool)
    return data
