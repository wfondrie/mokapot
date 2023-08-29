"""
Utility functions
"""
import itertools

import numpy as np
import polars as pl


def flatten(split):
    """Get the indices from split"""
    return list(itertools.chain.from_iterable(split))


def safe_divide(numerator, denominator, ones=False):
    """Divide ignoring div by zero warnings"""
    if isinstance(numerator, pl.Series):
        numerator = numerator.values

    if isinstance(denominator, pl.Series):
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


def listify(obj):
    """Convert obj to a list, without splitting strings"""
    try:
        _ = iter(obj)
    except TypeError:
        obj = [obj]
    else:
        if isinstance(obj, str):
            obj = [obj]

    return list(obj)


def make_lazy(data: pl.DataFrame | pl.LazyFrame | dict) -> pl.LazyFrame:
    """Coerce data into a LazyFrame.

    Parameters
    ----------
    data : DataFrame or dict
        A polars or pandas DataFrame or a dictionary that can be coerced into
        one containing the data.
    """
    try:
        return data.lazy().clone()
    except AttributeError as exc:
        last_exc = exc

    try:
        return pl.from_pandas(data).lazy().clone()
    except ValueError as exc:
        last_exc = exc

    try:
        return pl.DataFrame(data).lazy().clone()
    except ValueError as exc:
        last_exc = exc

    raise ValueError("Incompatible type for 'data'") from last_exc
