import time

import pytest
import numpy as np
import numpy.testing as testing
from scipy import stats

import mokapot.peps as pep


@pytest.fixture
def data():
    N = 5000
    pi0 = 0.7
    R0 = stats.norm(loc=-4, scale=2)
    R1 = stats.norm(loc=3, scale=2)
    NT0 = int(np.round(pi0 * N))
    NT1 = N - NT0
    target_scores = np.concatenate((np.maximum(R1.rvs(NT1), R0.rvs(NT1)), R0.rvs(NT0)))
    decoy_scores = R0.rvs(N)
    all_scores = np.concatenate((target_scores, decoy_scores))
    is_target = np.concatenate((np.full(len(target_scores), True), np.full(len(decoy_scores), False)))

    sortIdx = np.argsort(-all_scores)
    return [all_scores[sortIdx], is_target[sortIdx]]


__tictoc_t0: float


def tic():
    global __tictoc_t0
    __tictoc_t0 = time.time()


def toc():
    global __tictoc_t0
    elapsed = time.time() - __tictoc_t0
    print("Elapsed time: ", elapsed)


def test_peps_qvality(data):
    scores, targets = data
    peps = pep.peps_from_scores_qvality(scores, targets)
    assert np.all(peps >= 0)
    assert np.all(peps <= 1)
    assert np.all(np.diff(peps) * np.diff(scores) <= 0)


def test_peps_kde_nnls(data):
    scores, targets = data
    peps = pep.peps_from_scores_kde_nnls(scores, targets)
    assert np.all(peps >= 0)
    assert np.all(peps <= 1)
    assert np.all(np.diff(peps) * np.diff(scores) <= 0)


def test_peps_hist_nnls(data):
    scores, targets = data
    peps = pep.peps_from_scores_hist_nnls(scores, targets)
    assert np.all(peps >= 0)
    assert np.all(peps <= 1)
    assert np.all(np.diff(peps) * np.diff(scores) <= 0)


def test_monotonize():
    x = np.array([2, 3, 1])
    w = np.array([2, 1, 10])
    y = pep.monotonize_nnls(x, w, False)
    testing.assert_allclose(y, np.array([7/3, 7/3, 1]))
    w = np.array([3, 1, 1])
    y = pep.monotonize_nnls(x, w, True)
    testing.assert_allclose(y, np.array([2, 2, 2]))
    w = np.array([1, 3, 1])
    y = pep.monotonize_nnls(x, w, True)
    testing.assert_allclose(y, np.array([2, 2.5, 2.5]))
