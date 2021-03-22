"""
These tests verify that our q-value calculations are correct.
"""
import pytest
import numpy as np

from mokapot.qvalues import tdc


@pytest.fixture
def desc_scores():
    """Create a series of descending scores and their q-values"""
    scores = np.array([10, 10, 9, 8, 7, 7, 6, 5, 4, 3, 2, 2, 1, 1, 1, 1])
    target = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    qvals = np.array(
        [
            1 / 4,
            1 / 4,
            1 / 4,
            1 / 4,
            2 / 6,
            2 / 6,
            2 / 6,
            3 / 7,
            3 / 7,
            4 / 7,
            5 / 8,
            5 / 8,
            1,
            1,
            1,
            1,
        ]
    )
    return scores, target, qvals


@pytest.fixture
def asc_scores():
    """Create a series of ascending scores and their q-values"""
    scores = np.array([1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 10])
    target = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    qvals = np.array(
        [
            1 / 4,
            1 / 4,
            1 / 4,
            1 / 4,
            2 / 6,
            2 / 6,
            2 / 6,
            3 / 7,
            3 / 7,
            4 / 7,
            5 / 8,
            5 / 8,
            1,
            1,
            1,
            1,
        ]
    )
    return scores, target, qvals


def test_tdc_descending(desc_scores):
    """Test that q-values are correct for descending scores"""
    scores, target, true_qvals = desc_scores
    dtypes = [np.float64, np.uint8, np.int8, np.float32]
    for dtype in dtypes:
        qvals = tdc(scores.astype(dtype), target, desc=True)
        np.testing.assert_array_equal(qvals, true_qvals)

        qvals = tdc(scores, target.astype(dtype), desc=True)
        np.testing.assert_array_equal(qvals, true_qvals)


def test_tdc_ascending(asc_scores):
    """Test that q-values are correct for ascending scores"""
    scores, target, true_qvals = asc_scores
    dtypes = [np.float64, np.uint8, np.int8, np.float32]
    for dtype in dtypes:
        qvals = tdc(scores.astype(dtype), target, desc=False)
        np.testing.assert_array_equal(qvals, true_qvals)

        qvals = tdc(scores, target.astype(dtype), desc=False)
        np.testing.assert_array_equal(qvals, true_qvals)


def test_tdc_non_bool():
    """If targets is not boolean, should get a value errir"""
    scores = np.array([1, 2, 3, 4, 5])
    targets = np.array(["1", "0", "1", "0", "blarg"])
    with pytest.raises(ValueError):
        tdc(scores, targets)


def test_tdc_diff_len():
    """If the arrays are different lengths, should get a ValueError"""
    scores = np.array([1, 2, 3, 4, 5])
    targets = np.array([True] * 3 + [False] * 3)
    with pytest.raises(ValueError):
        tdc(scores, targets)
