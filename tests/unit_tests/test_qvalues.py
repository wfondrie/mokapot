"""
These tests verify that our q-value calculations are correct.
"""

import pytest
import numpy as np
from scipy import stats

from mokapot.qvalues import tdc, qvalues_from_peps, qvalues_from_counts


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


@pytest.mark.parametrize("dtype", [np.float64, np.uint8, np.int8, np.float32])
def test_tdc_descending(desc_scores, dtype):
    """Test that q-values are correct for descending scores"""
    scores, target, true_qvals = desc_scores
    qvals = tdc(scores.astype(dtype), target, desc=True)
    np.testing.assert_allclose(qvals, true_qvals, atol=1e-7)

    qvals = tdc(scores, target.astype(dtype), desc=True)
    np.testing.assert_allclose(qvals, true_qvals, atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float64, np.uint8, np.int8, np.float32])
def test_tdc_ascending(asc_scores, dtype):
    """Test that q-values are correct for ascending scores"""
    scores, target, true_qvals = asc_scores

    # Q - June2024 JSP: Since the function is type-checked, what is the purpose
    # of testing different types? Shouldnt they all fail?

    qvals = tdc(scores.astype(dtype), target, desc=False)
    # Testing equality in floats is not a good idea ...
    # So assert_allclose is used over assert_array_equal
    # Since for our purposes 0.333333333333 == 1/3
    np.testing.assert_allclose(qvals, true_qvals, atol=1e-7)

    qvals = tdc(scores, target.astype(dtype), desc=False)
    np.testing.assert_allclose(qvals, true_qvals, atol=1e-7)


def test_tdc_non_bool():
    """If targets is not boolean, should get a value error"""
    scores = np.array([1, 2, 3, 4, 5])
    targets = np.array(["1", "0", "1", "0", "blarg"])
    with pytest.raises(ValueError):
        # Q-June 2024: JSP Since this function is runtime-type-checked,
        # what is the purpose of this test?
        # should it be raise if the passed type is not
        # bool or if it is not convertible to bool?
        tdc(scores, targets)


def test_tdc_diff_len():
    """If the arrays are different lengths, should get a ValueError"""
    scores = np.array([1, 2, 3, 4, 5])
    targets = np.array([True] * 3 + [False] * 3)
    with pytest.raises(ValueError):
        tdc(scores, targets)


@pytest.fixture
def rand_scores():
    np.random.seed(1240)  # this produced an error with failing iterations
    N = 5000
    pi0 = 0.7
    R0 = stats.norm(loc=-4, scale=2)
    R1 = stats.norm(loc=3, scale=2)
    NT0 = int(np.round(pi0 * N))
    NT1 = N - NT0
    target_scores = np.concatenate(
        (
            np.maximum(R1.rvs(NT1), R0.rvs(NT1)),
            R0.rvs(NT0),
        )
    )
    decoy_scores = R0.rvs(N)
    all_scores = np.concatenate((target_scores, decoy_scores))
    is_target = np.concatenate(
        (
            np.full(len(target_scores), True),
            np.full(len(decoy_scores), False),
        )
    )

    sortIdx = np.argsort(-all_scores)
    return [all_scores[sortIdx], is_target[sortIdx]]


def test_qvalues_from_peps(rand_scores):
    # Note: we should also test against some known truth
    #   (of course, up to some error margin and fixing the random seed),
    #   and also against shuffeling of the target/decoy sequences.
    scores, targets = rand_scores
    qvalues = qvalues_from_peps(scores, targets)
    assert np.all(qvalues >= 0)
    assert np.all(qvalues <= 1)
    assert np.all(np.diff(qvalues) * np.diff(scores) <= 0)


def test_qvalues_from_counts(rand_scores):
    scores, targets = rand_scores
    qvalues = qvalues_from_counts(scores, targets)
    assert np.all(qvalues >= 0)
    assert np.all(qvalues <= 1)
    assert np.all(np.diff(qvalues) * np.diff(scores) <= 0)
