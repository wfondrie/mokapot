from collections import namedtuple
from logging import warning

import numpy as np
import scipy as sp


def empirical_pvalues(
    s: np.ndarray[float], s0: np.ndarray[float], *, mode: str = "best"
) -> np.ndarray[float]:
    """
    Computes the empirical p-values for a set of values.

    Parameters
    ----------
    s : np.ndarray[float]
        Array of data values/test statistics (typically scores) for which
        p-values are to be computed.

    s0 : np.ndarray[float]
        Array of data values (scores) simulated under the null hypothesis
        against which the data values `s` are to be compared.

    mode : str
        Can be one of "standard", "best", or "storey". Default is "best".
        "standard" is to use $r/n$ as estimator where $r$ is the rank of the
        data value in the array of simulated samples. This is unbiased when
        the data value is a concrete value, and not sampled.
        $best$ means $(r+1)/(n+1)$ which is unbiased and conservative when the
        data values are sampled from the null distribution.
        $storey$ is similar to "standard" but includes the "hack" for $r=0$
        found in the implementation of the `qvalue` package.

    Returns
    -------
    np.ndarray[float]
        Array of empirical p-values corresponding to the input data array `s`.
    """
    N = len(s0)
    emp_null = sp.stats.ecdf(s0)
    p = emp_null.sf.evaluate(s)
    mode = mode.lower()
    if mode == "standard":
        return p
    elif mode == "storey":
        return np.maximum(p, 1.0 / N)
    elif mode == "best":
        return (p * N + 1) / (N + 1)
    else:
        raise ValueError(
            f"Unknown mode {mode}. Must be either 'best', 'standard' or 'storey'."
        )


def estimate_pi0(
    p: np.ndarray[float], *, method: str = "smoother", lambdas=np.arange(0.05, 1, 0.05)
) -> np.ndarray[float]:
    """
    Estimate pi0 from p-value using Storey's method.

    Parameters
    ----------
    p : np.ndarray[float]
        Array of p-values for which the proportion of null hypotheses (pi0) is
        estimated.
    method : str, optional
        The method used for smoothing ('smoother' or 'bootstrap'). Default is
        'smoother'. ('bootstrap' is not yet implemented).
    lambdas : np.ndarray, optional
        An array of lambda values used to estimate pi0. Default is an array
        from 0.05 to 0.95 with step 0.05.

    Returns
    -------
    A namedtuple with fields
    - pi0 : float
        The estimated pi0 value.
    - pi0s_smoothed : np.ndarray[float]
        Array of smoothed pi0 values.
    - pi0s_lambda : np.ndarray[float]
        Array of raw pi0 estimates.
    - lambdas : np.ndarray[float]
        Array of lambdas used to estimate pi0.
    """
    N = len(p)
    lambdas = np.sort(lambdas)
    L = len(lambdas)

    assert min(p) >= 0 and max(p) <= 1
    assert min(lambdas) >= 0 and max(lambdas) <= 1
    assert len(lambdas) == 1 or len(lambdas) >= 4

    if max(p) < max(lambdas):
        warning(
            f"The maximum p-value ({max(p)}) should be larger than the "
            f"maximum lambda ({max(lambdas)})"
        )

    # Find for each p in which interval i given by lambdas[i-1]<=p<lambdas[i]
    # is located (set lambdas[-1]==-infinity)
    interval_indices = np.searchsorted(lambdas, p, side="right")

    # Count for each interval, how many p-values fall into it (drop the
    # [-infty,0.05] interval)
    interval_counts = np.bincount(interval_indices, minlength=L + 1)[1 : L + 1]

    # Estimate raw pi0 values ("contaminated" for small lambdas by the true
    # target distribution
    pvals_exceeding_lambda = np.flip(np.cumsum(np.flip(interval_counts)))
    pi0s = pvals_exceeding_lambda / (N * (1.0 - lambdas))

    # Now smooth it with a smoothing spline and evaluate
    smoothed_pi0 = sp.interpolate.UnivariateSpline(lambdas, pi0s, k=3, ext=0)
    pi0s_smooth = smoothed_pi0(lambdas)
    pi0 = pi0s_smooth[-1]

    Pi0Est = namedtuple("Pi0Est", ["pi0", "pi0s_smooth", "pi0s_lambda", "lambdas"])
    return Pi0Est(pi0, pi0s_smooth, pi0s, lambdas)
