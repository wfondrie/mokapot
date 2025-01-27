from typing import Any

import numpy as np
from pytest import raises
from typeguard import typechecked, TypeCheckError


def test_typecheck():
    # Without arguments
    @typechecked
    def f1(x: np.ndarray):
        return x

    f1(np.array([1, 2, 3]))
    f1(np.array(["1", "2", "3"]))
    raises(TypeCheckError, f1, [1, 2, 3])

    # With basic dtype argument
    @typechecked
    def f2(x: np.ndarray[float]):
        return x

    f2(np.array([1.0, 2.0, 3.0]))
    f2(np.zeros(shape=(3, 5), dtype=float))
    raises(TypeCheckError, f2, np.ones(shape=(2, 3, 4), dtype=int))
    raises(TypeCheckError, f2, [1, 2, 3])
    raises(TypeCheckError, f2, np.array([1, 2, 3]))
    raises(TypeCheckError, f2, np.array(["1", "2", "3"]))

    @typechecked
    def f3(x: np.ndarray[int]):
        return x

    f3(np.array([1, 2, 3]))
    raises(TypeCheckError, f3, np.array([1.0, 2.0, 3.0]))

    # Union dtype
    @typechecked
    def f4(x: np.ndarray[int | float]):
        return x

    f4(np.array([1.0, 2.0, 3.0]))
    f4(np.array([1, 2, 3]))
    raises(TypeCheckError, f4, np.array(["1", "2", "3"]))

    @typechecked
    def f5(x: np.ndarray[int] | np.ndarray[float]):
        return x

    f5(np.array([1.0, 2.0, 3.0]))
    f5(np.array([1, 2, 3]))
    raises(TypeCheckError, f5, np.array(["1", "2", "3"]))

    # Check with shape
    @typechecked
    def f6(x: np.ndarray[(2, 3), float]):
        return x

    f6(np.zeros(shape=(2, 3), dtype=float))
    raises(TypeCheckError, f6, np.array([1.0, 2.0, 3.0]))
    raises(TypeCheckError, f6, np.zeros(shape=(2, 3), dtype=int))
    raises(TypeCheckError, f6, np.zeros(shape=(2, 3, 1), dtype=float))
    raises(TypeCheckError, f6, np.zeros(shape=(2, 4), dtype=float))
    raises(TypeCheckError, f6, np.zeros(shape=(3, 3), dtype=float))

    @typechecked
    def f7(x: np.ndarray[int, (-1,)]):
        return x

    f7(np.ones(shape=10, dtype=int))
    raises(TypeCheckError, f7, np.ones(shape=10, dtype=float))
    raises(TypeCheckError, f7, np.ones(shape=(2, 5), dtype=float))

    @typechecked
    def f8(x: np.ndarray[Any, (3, -1, 5)]):
        return x

    f8(np.ones(shape=(3, 1, 5)))
    f8(np.ones(shape=(3, 2, 5)))
    raises(TypeCheckError, f8, np.ones(shape=(3, 1, 5, 1)))
    raises(TypeCheckError, f8, np.ones(shape=(3, 1, 4)))
    raises(TypeCheckError, f8, np.ones(shape=(2, 1, 4)))
