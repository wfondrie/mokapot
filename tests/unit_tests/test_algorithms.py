import numpy as np
import pytest

from mokapot.algorithms import (
    StoreyQvalueAlgorithm,
    TDCPi0Algorithm,
)
from mokapot.qvalues import tdc
from ..helpers.math import estimate_abs_int


@pytest.mark.parametrize("desc", [True, False])
@pytest.mark.parametrize("N", [100, 10000, 1000000])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("pi0", [0.3, 0.8])
def test_qvalue_algos(N, desc, discrete, pi0):
    # Test cases should be: large and small pi0, large and small sample sizes
    # discrete and continuous null statistics, and for ascending and descending
    # Maybe also for different distributions and target decoy separations

    if (discrete or pi0 < 0.5) and (N == 100):
        return  # doesn't work, no sufficient accuracy from decoys

    np.random.seed(42)
    targets = np.random.rand(N) > pi0
    decoys = ~targets
    # N_targets = targets.sum()
    N_decoys = decoys.sum()
    if discrete:
        scores = np.random.binomial(10, 0.7, N)
        scores[~targets] = np.random.binomial(8, 0.3, N_decoys)
        scores = scores.astype(float)
    else:
        scores = np.random.normal(0, 1, N)
        scores[~targets] = np.random.normal(-1, 1, N_decoys)

    if not desc:
        scores = -scores

    qvals = tdc(scores, targets, desc)
    pi0algo_ratio = TDCPi0Algorithm()
    # pi0algo_smoother = StoreyPi0Algorithm("smoother", eval_lambda=0.5)
    # pi0algo_fixed = StoreyPi0Algorithm("fixed", eval_lambda=0.5)
    # algo = StoreyQvalueAlgorithm(pi0_algo=pi0algo_fixed)
    # algo = StoreyQvalueAlgorithm(pi0_algo=pi0algo_smoother)
    algo = StoreyQvalueAlgorithm(pi0_algo=pi0algo_ratio)
    qvals2 = algo.qvalues(scores, targets, desc)

    np.testing.assert_allclose(qvals[targets], qvals2[targets], atol=0.001)
    np.testing.assert_allclose(qvals[decoys], qvals2[decoys], atol=0.1)
    assert estimate_abs_int(scores, qvals - qvals2, sort=True) < 0.002

    # todo: check the other methods, too
