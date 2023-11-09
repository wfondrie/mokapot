"""Utility functions."""
import functools
import time
from typing import Any, Callable

import polars as pl


def listify(obj: Any) -> list[Any]:
    """Convert obj to a list, without splitting strings or dataframes.

    Parameters
    ----------
    obj : anything
        The object to turn into a list.

    Returns
    -------
    list
        The list representation of th object.
    """
    try:
        _ = iter(obj)
    except TypeError:
        obj = [obj]
    else:
        # Don't listify strings or DataFrames.
        if isinstance(obj, str) or hasattr(obj, "columns"):
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
    except TypeError as exc:
        last_exc = exc

    try:
        return pl.DataFrame(data).lazy().clone()
    except ValueError as exc:
        last_exc = exc

    raise ValueError("Incompatible type for 'data'") from last_exc


def timethis(label: str = "") -> Callable:
    """A decorator for timing."""

    def decorator(func: Callable) -> Callable:
        """The decorator."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: dict) -> Any:
            """The returned wrapper."""
            start = time.time()
            out = func(*args, **kwargs)
            print(label, time.time() - start)  # noqa: T201
            return out

        return wrapper

    return decorator
