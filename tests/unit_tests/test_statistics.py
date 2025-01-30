import math

import numpy as np
import pytest
import scipy as sp
from pytest import approx

from mokapot.stats.statistics import HistData, OnlineStatistics


def test_init():
    stats = OnlineStatistics()
    assert stats.min == math.inf
    assert stats.max == -math.inf
    assert stats.n == 0
    assert stats.sum == 0.0
    assert stats.mean == 0.0
    assert stats.var == 0.0
    assert stats.sd == 0.0


def test_update():
    stats = OnlineStatistics()
    vals = np.array([1, 2, 3, 4, 5])
    stats.update(vals)

    assert stats.min == 1.0
    assert stats.max == 5.0
    assert stats.n == 5
    assert stats.sum == 15.0
    assert stats.mean == 3.0
    assert math.isclose(stats.var, 2.5)
    assert math.isclose(stats.sd, 1.5811388300841898)


def test_update_multiple():
    stats = OnlineStatistics()
    vals1 = np.array([1, 2, 3, 4, 5])
    stats.update(vals1)
    vals2 = np.array([6, 7, 8, 9, 10])
    stats.update(vals2)

    desc = sp.stats.describe(np.concatenate((vals1, vals2)))
    assert (stats.min, stats.max) == desc.minmax
    assert stats.n == desc.nobs
    assert stats.mean == desc.mean
    assert math.isclose(stats.var, desc.variance)


def test_update_multiple2():
    stats = OnlineStatistics()
    vals = np.array([])
    for _ in range(1000):
        n = np.random.randint(10, 50)
        new_vals = 100 * np.random.random_sample(n)
        stats.update(new_vals)
        vals = np.concatenate((vals, new_vals))

    assert stats.min == vals.min()
    assert stats.max == vals.max()
    assert stats.n == len(vals)
    assert stats.mean == approx(np.mean(vals))
    assert stats.var == approx(np.var(vals, ddof=1), rel=1e-14)
    assert stats.sd == approx(np.std(vals, ddof=1), rel=1e-14)


def test_update_single():
    stats = OnlineStatistics()
    vals = np.arange(10)
    for val in vals:
        stats.update_single(val)

    assert stats.min == vals.min()
    assert stats.max == vals.max()
    assert stats.n == len(vals)
    assert stats.mean == np.mean(vals)
    assert stats.var == approx(np.var(vals, ddof=1))
    assert stats.sd == approx(np.std(vals, ddof=1))


def test_max_likelihood_variance():
    stats = OnlineStatistics(unbiased=False)
    vals = np.arange(10)
    stats.update(vals)
    assert stats.var == approx(np.var(vals, ddof=0))
    assert stats.sd == approx(np.std(vals, ddof=0))


def test_hist_data():
    N = 1000
    x = np.concatenate([np.random.normal(size=N), np.random.normal(2, size=N)])
    counts, bin_edges = np.histogram(x, bins="scott")
    hist = HistData(bin_edges, counts)

    density, _ = np.histogram(x, bins=bin_edges, density=True)
    assert density == approx(hist.density)

    bin_centers = hist.bin_centers
    assert len(bin_centers) == len(counts)
    assert bin_centers[0] == approx((bin_edges[0] + bin_edges[1]) / 2)
    assert bin_centers[-1] == approx((bin_edges[-2] + bin_edges[-1]) / 2)


def test_binning():
    N = 10000
    x = np.random.normal(2, 3, size=N)

    stats = OnlineStatistics()
    stats.update(x)

    assert HistData.get_bin_edges(stats, "scott") == approx(
        np.histogram_bin_edges(x, bins="scott")
    )
    assert HistData.get_bin_edges(stats, "sturges") == approx(
        np.histogram_bin_edges(x, bins="sturges")
    )
    assert HistData.get_bin_edges(stats, "auto") == approx(
        np.histogram_bin_edges(x, bins="scott")
    )

    # If we extend the bins, the new bins should have the same center, it
    # should be one bin more, and they should all have the same size
    edges0 = HistData.get_bin_edges(stats, "scott")
    edges1 = HistData.get_bin_edges(stats, "scott", extend=True)
    assert len(edges1) == len(edges0) + 1
    assert np.diff(edges1).mean() == approx(np.diff(edges0).mean())
    assert edges1.mean() == approx(edges0.mean())

    assert HistData.get_bin_edges(stats, clip=(2, 3)) == approx(
        np.histogram_bin_edges(x, bins=3)
    )
    assert HistData.get_bin_edges(stats, clip=(200, 202)) == approx(
        np.histogram_bin_edges(x, bins=200)
    )

    with pytest.raises(ValueError):
        HistData.get_bin_edges(stats, name="xyz")
