"""
These tests verify that our q-value calculations are correct.
"""

import numpy as np
import pytest
from scipy import stats

from mokapot.peps import hist_data_from_scores, TDHistData
from mokapot.qvalues import (
    qvalues_from_counts,
    qvalues_from_peps,
    qvalues_from_storeys_algo,
    qvalues_func_from_hist,
    tdc,
)
from ..helpers.utils import TestOutcome


@pytest.fixture
def desc_scores():
    """Create a series of descending scores and their q-values"""
    scores = np.array([10, 10, 9, 8, 7, 7, 6, 5, 4, 3, 2, 2, 1, 1, 1, 1], dtype=float)
    target = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    qvals = np.array([
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
    ])
    return scores, target, qvals


@pytest.fixture
def asc_scores():
    """Create a series of ascending scores and their q-values"""
    scores = np.array([1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 10], dtype=float)
    target = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    qvals = np.array([
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
    ])
    return scores, target, qvals


def qvalues_are_valid(qvalues, scores=None):
    """Helper function for tests on qvalues"""
    if not np.all(qvalues >= 0):
        return TestOutcome.fail("'qvalues must be >= 0'")
    if not np.all(qvalues <= 1):
        return TestOutcome.fail("'qvalues must be <= 1'")

    if scores is not None:
        ind = np.argsort(scores)
        diff_qvals = np.diff(qvalues[ind])
        diff_scores = np.diff(scores[ind])
        if not np.all(diff_qvals * diff_scores <= 0):
            return TestOutcome.fail(
                "'qvalues are monotonically decreasing with higher scores'"
            )

        if not np.all((diff_scores != 0) | (diff_qvals == 0)):
            # Note that "!A | B" is the same as the implication "A => B"
            # When two scores are equal they must have the same qvalue, but if
            # they are different the may have the same qvalue
            return TestOutcome.fail("'equal scores must have equal qvalues'")

    return TestOutcome.success()


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
    target_scores = np.concatenate((
        np.maximum(R1.rvs(NT1), R0.rvs(NT1)),
        R0.rvs(NT0),
    ))
    decoy_scores = R0.rvs(N)
    all_scores = np.concatenate((target_scores, decoy_scores))
    is_target = np.concatenate((
        np.full(len(target_scores), True),
        np.full(len(decoy_scores), False),
    ))
    # Generate a permutation of indices and apply the permutation to both arrays
    permutation = np.random.permutation(len(all_scores))
    return [all_scores[permutation], is_target[permutation]]


def all_sorted(arrays, desc=True):
    sortIdx = np.argsort(-arrays[0] if desc else arrays[0])
    return (arr[sortIdx] for arr in arrays)


def rounded(arrays, decimals=1):
    scores, is_target = arrays
    scores = np.round(scores, decimals=decimals)
    return scores, is_target


@pytest.fixture
def rand_scores_rounded(rand_scores):
    [all_scores, is_target] = rand_scores
    sortIdx = np.argsort(-all_scores)
    return [all_scores[sortIdx], is_target[sortIdx]]


@pytest.mark.parametrize("is_tdc", [True, False])
def test_qvalues_from_peps(rand_scores, is_tdc):
    # Note: we should also test against some known truth
    #   (of course, up to some error margin and fixing the random seed),
    #   and also against shuffeling of the target/decoy sequences.
    scores, targets = rand_scores
    qvalues = qvalues_from_peps(scores, targets, is_tdc)
    assert qvalues_are_valid(qvalues, scores)


@pytest.mark.parametrize("is_tdc", [True, False])
def test_qvalues_from_counts(rand_scores, is_tdc):
    scores, targets = rand_scores
    qvalues = qvalues_from_counts(scores, targets, is_tdc)
    assert qvalues_are_valid(qvalues, scores)


def test_qvalues_from_storey(rand_scores):
    scores, targets = rand_scores
    qvalues = qvalues_from_storeys_algo(scores, targets, decoy_qvals_by_interp=True)
    assert qvalues_are_valid(qvalues, scores)

    # Test with rounded scores, so that there are scores
    scores, targets = rounded(rand_scores, decimals=0)
    qvalues = qvalues_from_storeys_algo(scores, targets, decoy_qvals_by_interp=True)
    assert qvalues_are_valid(qvalues, scores)

    # Test with separate target/decoy qvalue evaluation (qvalues not globally sorted)
    qvalues = qvalues_from_storeys_algo(scores, targets, decoy_qvals_by_interp=False)
    assert qvalues_are_valid(qvalues[targets], scores[targets])
    assert qvalues_are_valid(qvalues[~targets], scores[~targets])


def test_qvalues_from_counts_descending(desc_scores):
    """Test that q-values are correct for descending scores"""
    scores, target, true_qvals = desc_scores
    targets = target == 1
    qvals = qvalues_from_counts(scores, targets, is_tdc=True)
    np.testing.assert_allclose(qvals, true_qvals, atol=1e-7)


def test_qvalues_from_hist_desc(desc_scores):
    scores, target, true_qvals = desc_scores
    targets = target == 1
    # Use small bins covering every interval + the scores as bin edges
    bin_edges = np.linspace(0, 11, num=371)
    bin_edges = np.unique(np.sort(np.concatenate((bin_edges, scores))))

    hist_data = TDHistData.from_scores_targets(bin_edges, scores, targets)
    qvalue_func = qvalues_func_from_hist(hist_data, is_tdc=True)
    qvals = qvalue_func(scores)

    np.testing.assert_allclose(qvals, true_qvals, atol=1e-7)


def test_compare_rand_qvalues_from_hist_vs_count(rand_scores):
    # Compare the q-values computed via counts with those computed via
    # histogram on a dataset of a few thousand random scores
    scores, targets = rand_scores
    hist_data = hist_data_from_scores(scores, targets)
    qvalue_func = qvalues_func_from_hist(hist_data, is_tdc=True)
    qvals_hist = qvalue_func(scores)
    qvals_counts = qvalues_from_counts(scores, targets, is_tdc=True)

    np.testing.assert_allclose(qvals_hist, qvals_counts, atol=0.02)


def test_qvalues_discrete(rand_scores):
    scores, targets = rand_scores
    scores = np.asarray(scores > scores.mean(), dtype=float)

    qvals_tdc = tdc(scores, targets)

    qvals_counts = qvalues_from_counts(scores, targets, is_tdc=True)
    np.testing.assert_allclose(qvals_counts, qvals_tdc)

    # A tolerance of 0.1 is okay in the following tests, since the methods are
    # widely different.
    qvals_st1 = qvalues_from_storeys_algo(scores, targets, pvalue_method="conservative")
    np.testing.assert_allclose(qvals_st1, qvals_tdc, atol=0.1)

    qvals_st2 = qvalues_from_storeys_algo(scores, targets, pvalue_method="storey")
    np.testing.assert_allclose(qvals_st2, qvals_tdc, atol=0.1)
