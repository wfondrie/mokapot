import numpy as np

from mokapot.qvalues_storey import empirical_pvalues


def test_empirical_pvalues():
    # Basic test for computing empirical pvalues, compare results to
    # results from Storey's R implementation (qvalue::empPvals)
    s = np.arange(0, 13)
    s0 = np.arange(-99, 101)
    p = 0.5 - s / len(s0)
    def empPvals(s, s0):
        return empirical_pvalues(s, s0, mode = "storey")
    np.testing.assert_almost_equal(empPvals(s, s0), p)

    # Now shift and compare
    delta = 1e-12
    np.testing.assert_almost_equal(empPvals(s + 1, s0), p - 0.005)
    np.testing.assert_almost_equal(empPvals(s + 1 - delta, s0), p)
    np.testing.assert_almost_equal(empPvals(s + delta, s0), p)
    np.testing.assert_almost_equal(empPvals(s - delta, s0), p + 0.005)
    np.testing.assert_almost_equal(empPvals(s - 1, s0), p + 0.005)

    # Test the different p-value computation modes
    s = np.arange(-200, 200, 23)
    s0 = np.arange(-99, 101)
    N = len(s0)
    p = np.clip(0.5 - s / N, 0, 1)
    np.testing.assert_almost_equal(empirical_pvalues(s, s0, mode="standard"), p)
    p = np.clip(0.5 - s / N, 1.0 / N, 1)
    np.testing.assert_almost_equal(empirical_pvalues(s, s0, mode="storey"), p)
    p = np.clip((101 - s) / (1 + N), 1.0 / (N + 1), 1)
    np.testing.assert_almost_equal(empirical_pvalues(s, s0, mode="best"), p)

