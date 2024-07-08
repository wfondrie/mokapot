import time

import pytest
import logging
from pytest import approx
import numpy as np
import numpy.testing as testing
from scipy import stats

import mokapot.peps as peps


def get_target_decoy_data():
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


__tictoc_t0: float


def tic():
    global __tictoc_t0
    __tictoc_t0 = time.time()


def toc():
    global __tictoc_t0
    elapsed = time.time() - __tictoc_t0
    logging.info(f"Elapsed time: {elapsed}")


def test_monotonize_simple():
    # ascending
    assert (peps.monotonize_simple([1, 2, 1, 3], True) == [1, 2, 2, 3]).all()
    # descending
    assert (peps.monotonize_simple([3, 2, 3, 1], False) == [3, 2, 2, 1]).all()
    # emtpy array
    assert (peps.monotonize_simple([], True) == []).all()
    assert (peps.monotonize_simple([], False) == []).all()
    # single element
    assert (peps.monotonize_simple([5], True) == [5]).all()
    assert (peps.monotonize_simple([5], False) == [5]).all()


def test_monotonize():
    x = np.array([2, 3, 1])
    w = np.array([2, 1, 10])
    y = peps.monotonize_nnls(x, w, False)
    testing.assert_allclose(y, np.array([7 / 3, 7 / 3, 1]))
    w = np.array([3, 1, 1])
    y = peps.monotonize_nnls(x, w, True)
    testing.assert_allclose(y, np.array([2, 2, 2]))
    w = np.array([1, 3, 1])
    y = peps.monotonize_nnls(x, w, True)
    testing.assert_allclose(y, np.array([2, 2.5, 2.5]))

    # Test to check monotonize on a larger array
    np.random.seed(0)
    random_array = np.random.rand(1000)
    assert np.all(np.diff(peps.monotonize(random_array, True)) >= 0)
    assert np.all(np.diff(peps.monotonize(random_array, False)) <= 0)

    # Test to check simple_averaging parameter when set to True
    assert np.allclose(
        peps.monotonize(np.array([1, 2, 5, 4, 3]), True, True),
        np.array([1, 2, 4, 4, 4]),
    )
    assert np.allclose(
        peps.monotonize(np.array([1, 2, 6, 6, 3]), True, True),
        np.array([1, 2, 4.5, 4.5, 4.5]),
    )

    # Test to check simple_averaging parameter when set to False
    assert np.allclose(
        peps.monotonize(np.array([1, 2, 5, 4, 3]), True, False),
        np.array([1, 2, 4, 4, 4]),
    )
    assert np.allclose(
        peps.monotonize(np.array([1, 2, 6, 6, 3]), True, False),
        np.array([1, 2, 5, 5, 5]),
    )


def test_fit_nnls0():
    # n = np.array([3, 2, 0, 0, 1, 1, 3])
    # k = np.array([0, 0, 1, 1, 1, 1, 3])
    n = np.array([2, 1, 0, 1, 0, 2])
    k = np.array([0, 0, 1, 1, 1, 2])
    _ = peps.fit_nnls(n, k, ascending=True)
    _ = peps.fit_nnls(n, k, ascending=False)


def test_fit_nnls_zeros():
    # Test the handling of n==0 in fit_nnls

    # Simple three element array
    n = np.array([1, 0, 1])
    k = np.array([0, 0, 1])
    p = peps.fit_nnls(n, k, ascending=True)
    assert p == approx([0, 1 / 2, 1])
    p = peps.fit_nnls(n, k, ascending=True, erase_zeros=True)
    assert p == approx([0, 1, 1])

    # Two zeros after each other
    n = np.array([1, 1, 0, 0, 1, 1])
    k = np.array([0, 0, 0, 0, 1, 1])
    p = peps.fit_nnls(n, k, ascending=True)
    assert p == approx([0, 0, 1 / 3, 2 / 3, 1, 1])
    p = peps.fit_nnls(n, k, ascending=True, erase_zeros=True)
    assert p == approx([0, 0, 1, 1, 1, 1])

    # Tricky: n==0 at the end
    n = np.array([1, 1, 0, 0, 1, 0])
    k = np.array([0, 0, 0, 0, 1, 1])
    p = peps.fit_nnls(n, k, ascending=True)
    assert p == approx([0, 0, 1 / 3, 2 / 3, 1, 1])
    p = peps.fit_nnls(n, k, ascending=True, erase_zeros=True)
    assert p == approx([0, 0, 1, 1, 1, 1])

    # Descending
    n = np.array([1, 1, 0, 0, 1, 1])
    k = np.array([0, 0, 0, 0, 1, 1])
    p = peps.fit_nnls(n, k, ascending=False)
    assert p == pytest.approx([1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2])
    p = peps.fit_nnls(n, k, ascending=False, erase_zeros=True)
    assert p == pytest.approx([1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2])


def test_fit_nnls_zeros_mult(caplog):
    # Run multiple times and check convergence and consistency
    seed_value = np.random.randint(0, 1000000)
    np.random.seed(seed_value)
    for i in range(100):
        N = 6
        n = np.random.randint(size=N, low=0, high=5)
        k = np.random.random(size=N) * n
        p1 = peps.fit_nnls(n, 0.5 * k, ascending=False)
        p2 = peps.fit_nnls(n, 2 * k, ascending=False)
        assert p1 == approx(0.25 * p2)


def test_fit_nnls_peps():
    # This is from a real test case that failed due to a problem in
    # the scipy._nnls implementation
    n0 = np.array(
        [
            3,
            2,
            0,
            1,
            1,
            1,
            3,
            8,
            14,
            16,
            29,
            23,
            41,
            47,
            53,
            57,
            67,
            76,
            103,
            89,
            97,
            94,
            85,
            95,
            78,
            78,
            78,
            77,
            73,
            50,
            50,
            56,
            68,
            98,
            95,
            112,
            134,
            145,
            158,
            172,
            213,
            234,
            222,
            215,
            216,
            216,
            206,
            183,
            135,
            156,
            110,
            92,
            63,
            60,
            52,
            29,
            20,
            16,
            12,
            5,
            5,
            5,
            1,
            2,
            3,
            0,
            2,
        ]
    )
    k0 = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.7205812007860187,
            0.0,
            1.4411624015720375,
            0.7205812007860187,
            2.882324803144075,
            5.76464960628815,
            5.76464960628815,
            12.249880413362318,
            15.132205216506394,
            20.176273622008523,
            27.382085629868712,
            48.27894045266326,
            47.558359251877235,
            68.45521407467177,
            97.99904330689854,
            108.0871801179028,
            135.46926574777152,
            140.51333415327366,
            184.4687874012208,
            171.49832578707245,
            205.36564222401535,
            244.27702706646033,
            214.01261663344755,
            228.42424064916793,
            232.02714665309804,
            205.36564222401535,
            172.9394881886445,
            191.67459940908097,
            162.1307701768542,
            153.48379576742198,
            110.96950492104689,
            103.04311171240067,
            86.46974409432225,
            60.528820866025576,
            43.234872047161126,
            23.779179625938617,
            24.499760826724636,
            17.29394881886445,
            11.5292992125763,
            5.76464960628815,
            5.044068405502131,
            3.6029060039300935,
            0.0,
            2.882324803144075,
            0.0,
            0.0,
            0.0,
        ]
    )
    p = peps.fit_nnls(n0, k0, ascending=True, erase_zeros=True)
    assert np.all(np.diff(p) >= 0)
    assert np.linalg.norm(k0 - n0 * p, np.inf) < 40
    p = peps.fit_nnls(n0, k0, ascending=True, erase_zeros=not True)
    assert np.all(np.diff(p) >= 0)
    assert np.linalg.norm(k0 - n0 * p, np.inf) < 40


def test_peps_qvality():
    scores, targets = get_target_decoy_data()
    peps_values = peps.peps_from_scores_qvality(scores, targets)
    assert np.all(peps_values >= 0)
    assert np.all(peps_values <= 1)
    assert np.all(np.diff(peps_values) * np.diff(scores) <= 0)


def test_peps_kde_nnls():
    np.random.seed(
        1253
    )  # this produced an error with failing iterations in nnls
    scores, targets = get_target_decoy_data()
    peps_values = peps.peps_from_scores_kde_nnls(scores, targets)
    assert np.all(peps_values >= 0)
    assert np.all(peps_values <= 1)
    assert np.all(np.diff(peps_values) * np.diff(scores) <= 0)

    np.random.seed(
        1245
    )  # this produced an assertion error due to peps over 1.0
    scores, targets = get_target_decoy_data()
    peps_values = peps.peps_from_scores_kde_nnls(scores, targets)
    assert np.all(peps_values >= 0)
    assert np.all(peps_values <= 1)
    assert np.all(np.diff(peps_values) * np.diff(scores) <= 0)


@pytest.mark.parametrize(
    "seed",
    [
        # Those were collected seeds from random experiments where nnls failed
        1253,
        41908,
        39831,
        21706,
        38306,
        23020,
        46079,
        96127,
        23472,
        21706,
        38306,
        23020,
        46079,
        96127,
        23472,
        21706,
        38306,
        23020,
        46079,
        96127,
        23472,
        21706,
        38306,
        23020,
        46079,
        96127,
        23472,
        21706,
        38306,
        23020,
        46079,
        96127,
        23472,
        21706,
        38306,
        23020,
        46079,
        96127,
        23472,
        21706,
    ],
)
def test_peps_hist_nnls(seed):
    np.random.seed(seed)
    scores, targets = get_target_decoy_data()
    try:
        peps_values = peps.peps_from_scores_hist_nnls(scores, targets)
        assert np.all(peps_values >= 0)
        assert np.all(peps_values <= 1)
        assert np.all(np.diff(peps_values) * np.diff(scores) <= 0)
    except Exception as e:
        pytest.fail(f"nnls failed on seed {seed}: {str(e)}")
