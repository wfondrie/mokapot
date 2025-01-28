from inspect import isclass
from types import UnionType
from typing import Any, get_args

import numpy as np
from typeguard import (
    checker_lookup_functions,
    TypeCheckerCallable,
    TypeCheckError,
    TypeCheckMemo,
)


def check_dtype(act_dtype, req_type):
    if isinstance(req_type, UnionType):
        return any([check_dtype(act_dtype, t) for t in get_args(req_type)])
    elif isinstance(req_type, type(Any)):
        return True
    else:
        return np.issubdtype(act_dtype, req_type)


def check_numpy_ndarray(
    value: Any, origin_type: Any, args: tuple[Any, ...], memo: TypeCheckMemo
) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeCheckError(f"is no numpy.ndarray[{args}].")

    for arg in args:
        base_msg = f"is an ndarray[{value.dtype}, {value.shape}]"
        if isinstance(arg, type | UnionType | type(Any)):
            if not check_dtype(value.dtype, arg):
                raise TypeCheckError(
                    f"{base_msg} and not numpy.ndarray[{args}] (type mismatch)"
                )
        elif isinstance(arg, int | tuple):
            if isinstance(arg, int):
                arg = (arg,)
            if len(arg) != len(value.shape):
                raise TypeCheckError(
                    f"{base_msg} and not numpy.ndarray[{args}] (dimension mismatch)"
                )
            okay = [s == t for s, t in zip(arg, value.shape) if s != -1]
            if not all(okay):
                raise TypeCheckError(
                    f"{base_msg} and not numpy.ndarray[{args}] (shape mismatch)"
                )
        else:
            raise TypeCheckError(
                f"{base_msg}. Unkonwn argument type: {type(arg)} {arg}"
            )


def numpy_ndarray_checker_lookup(
    origin_type: Any, args: tuple[Any, ...], extras: tuple[Any, ...]
) -> TypeCheckerCallable | None:
    if isclass(origin_type) and issubclass(origin_type, np.ndarray):
        return check_numpy_ndarray

    return None


def register_numpy_typechecker():
    if numpy_ndarray_checker_lookup not in checker_lookup_functions:
        checker_lookup_functions.append(numpy_ndarray_checker_lookup)


def unregister_numpy_typechecker():
    try:
        checker_lookup_functions.remove(numpy_ndarray_checker_lookup)
    except ValueError:
        pass
