import numpy as np

from mokapot.qvalues_storey import empirical_pvalues


def test_empirical_pvalues():
    # Basic test for computing empirical pvalues, compare results to
    # results from Storey's R implementation (qvalue::empPvals)
    s = np.arange(0, 13)
    s0 = np.arange(-99, 101)
    p = 0.5 - 0.005 * np.arange(0, 13)
    np.testing.assert_almost_equal(empirical_pvalues(s, s0), p)

    # Now shift and compare
    delta = 1e-12
    np.testing.assert_almost_equal(empirical_pvalues(s + 1, s0), p - 0.005)
    np.testing.assert_almost_equal(empirical_pvalues(s + 1 - delta, s0), p)
    np.testing.assert_almost_equal(empirical_pvalues(s + delta, s0), p)
    np.testing.assert_almost_equal(empirical_pvalues(s - delta, s0), p + 0.005)
    np.testing.assert_almost_equal(empirical_pvalues(s - 1, s0), p + 0.005)

