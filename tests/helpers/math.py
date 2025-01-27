import warnings

import numpy as np
import pandas as pd
import scipy


def reduce_linear(x, y, rtol=1e-10):
    """
    Removes unnecessary points from point set w.r.t. to linear interpolation.

    Can be used to test whether two sets of points returned from function
    estimators (e.g. to estimate q-values from scores) and fed into linear
    interpolators are equivalent, i.e. will lead to the same interpolated
    function.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the points.
    y : array-like
        The y-coordinates of the points.
    rtol : float, optional
        The relative tolerance to determine when points can be reduced
        (default is 1e-10).

    Returns
    -------
    x : array-like
        Reduced x-coordinates of the points.
    y : array-like
        Reduced y-coordinates of the points.
    """
    tol = rtol * (y.max() - y.min())
    # Compute linearly interpolated values ym_i at points x_i from the pairs
    # (x_i-1, y_i-1) and (x_i+1, y_i+1)
    ddx = x[:-2] - x[2:]
    ddy = y[:-2] - y[2:]
    ym = y[:-2] + ddy / ddx * (x[1:-1] - x[:-2])
    # Value of y_i at point x_i for comparison. If the difference to ym_i is
    # near zero it can be eliminated.
    yi = y[1:-1]
    # Keep points with near zero diff plus the endpoints
    keep = np.abs(ym - yi) > tol
    keep = np.append(np.insert(keep, 0, True), True)
    x = x[keep]
    y = y[keep]
    return x, y


def estimate_abs_int(x, y, sort=False):
    """Estimate the normalized absolute difference between two curves"""
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    if sort:
        sort_ind = np.argsort(x)
        x = x[sort_ind]
        y = y[sort_ind]

    if all(np.diff(x) <= 0):
        x = x[::-1]
        y = y[::-1]
    assert all(np.diff(x) >= 0)

    a = np.abs(y)
    # Determine possible 0 crossings of y (which means kinks in abs(y))
    with warnings.catch_warnings():
        # we sort out nans later
        warnings.simplefilter("ignore")
        x2 = (a[1:] * x[:-1] + a[:-1] * x[1:]) / (a[1:] + a[:-1])
    x2 = x2[~np.isnan(x2)]
    # Add zero-crossing points to integral eval points (+ sort and make unique)
    xn = np.unique(np.sort(np.concatenate((x2, x))))
    an = np.abs(np.interp(xn, x, y))
    # Trapezoidal rule is exact for piecewise linear functions
    return scipy.integrate.trapezoid(an, xn) / (xn[-1] - xn[0])
