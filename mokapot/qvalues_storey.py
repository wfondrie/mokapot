from collections import namedtuple
from logging import warning

import numpy as np
import scipy as sp
from typeguard import typechecked

from mokapot.peps import monotonize_simple


@typechecked
def empirical_pvalues(
    s: np.ndarray[float], s0: np.ndarray[float], *, mode: str = "conservative"
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
        Can be one of "unbiased", "conservative", or "storey". Default is
        "conservative".
        "unbiased" means to use $r/n$ as an estimator, where $r$ is the rank of
        the data value in the array of simulated samples. This is an unbiased
        and good when the data value is a concrete value, and not sampled.
        $conservative$ means $(r+1)/(n+1)$ which is a slightly biased, but
        conservative when the data values are sampled from the null
        distribution, in which case it is to be preferred.
        $storey$ is similar to "unbiased" but includes the "hack" for $r=0$
        found in the implementation of the `qvalue` package. It is neither
        completely unbiased nor convervative.

    Returns
    -------
    np.ndarray[float]
        Array of empirical p-values corresponding to the input data array `s`.

    References
    ----------

    .. [1] B V North, D Curtis, P C Sham, A Note on the Calculation of
       Empirical P Values from Monte Carlo Procedures,  Am J Hum Genet, 2003
       Feb;72(2):498–499. doi: 10.1086/346173
    .. [2] https://en.wikipedia.org/wiki/P-value#Definition
    """
    N = len(s0)

    # The p-value of some test statistic is the probability this or a higher
    # value would be attained under the null hypothesis, i.e. p=Pr(S>=s|H0) (see
    # [2], or p=Pr(S0>=s) if we denote by S0 the sample distribution under H0.
    # The cumulative distribution function (CDF) of S0 is given by
    # F_{S0}(s) = Pr(S0<=s).
    # Since the event S0<=s happens exactly when -S0>=-s, we see that
    # p=Pr(S0>=s)=Pr(-S0<=-s)=F_{-S0}(-s).
    # Note: some derivations out in the wild are not correct, as they don't
    # consider discrete or mixed distributions and compute the p-value via the
    # survival function SF_{S0}(s)=Pr(S0>s), which is okay for continuous
    # distributions, but problematic otherwise, if the distribution has
    # non-zero probability mass at s.
    emp_null = sp.stats.ecdf(-s0)
    p = emp_null.cdf.evaluate(-s)

    mode = mode.lower()
    if mode == "unbiased":
        return p
    elif mode == "storey":
        # Apply Storey's correction for 0 p-values
        return np.maximum(p, 1.0 / N)
    elif mode == "conservative":
        # Make p-values slightly biased, but conservative (see [1])
        return (p * N + 1) / (N + 1)
    else:
        raise ValueError(
            f"Unknown mode {mode}. Must be either 'conservative', 'unbiased' or"
            " 'storey'."
        )


@typechecked
def count_larger(pvals, lambdas):
    """
    Counts the number of elements in `pvals` that are greater than or equal to
    each value in `lambdas`.

    This function calculates the cumulative count of `pvals` values that are
    larger than or equal to each quantile in the `lambdas` array. The `lambdas`
    array is expected to be in strictly ascending order.

    Parameters
    ----------
    pvals : array_like
        Array of values for which cumulative counts are calculated against the
        `lambdas` thresholds.

    lambdas : array_like
        Array of threshold values in strictly ascending order. The counts of
        `pvals` larger than or equal to these values will be computed.

    Returns
    -------
    cumulative_counts : ndarray
        Array of cumulative counts, where each element corresponds to the count
        of `pvals` that are greater than or equal to the corresponding value in
        `lambdas`.
    """
    assert np.all(lambdas[1:] > lambdas[:-1])
    bin_edges = np.append(lambdas, np.inf)
    hist_counts, _ = np.histogram(pvals, bins=bin_edges)
    cumulative_counts = np.cumsum(hist_counts[::-1])[::-1]
    return cumulative_counts


Pi0Est = namedtuple("Pi0Est", ["pi0", "pi0s_smooth", "pi0s_raw", "lambdas", "mse"])


@typechecked
def estimate_pi0(
    pvals: np.ndarray[float],
    *,
    method: str = "smoother",
    lambdas: np.ndarray[float] = np.arange(0.2, 0.8, 0.01),
    eval_lambda: float = 0.5,
) -> Pi0Est:
    """
    Estimate pi0 from p-value using Storey's method.

    Parameters
    ----------
    pvals : np.ndarray[float]
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

    References
    ----------
    .. [1] John D. Storey, Robert Tibshirani, Statistical significance for
        genomewide studies,pp. 9440 –9445 PNAS August 5, 2003, vol. 100, no. 16
        www.pnas.org/cgi/doi/10.1073/pnas.1530509100
    .. [2] Storey JD, Bass AJ, Dabney A, Robinson D (2024). qvalue: Q-value
        estimation for false discovery rate control. R package version 2.38.0,
        http://github.com/jdstorey/qvalue.
    """
    N = len(pvals)
    lambdas = np.sort(lambdas)

    assert min(pvals) >= 0 and max(pvals) <= 1
    assert min(lambdas) >= 0 and max(lambdas) <= 1
    assert len(lambdas) >= 4

    if max(pvals) < max(lambdas):
        warning(
            f"The maximum p-value ({max(pvals)}) should be larger than the "
            f"maximum lambda ({max(lambdas)})"
        )

    pi0s_smooth = pi0s_raw = mse = None

    if method == "fixed":
        pvals_exceeding_lambda = sum(pvals >= eval_lambda)
        pi0 = pvals_exceeding_lambda / (N * (1.0 - eval_lambda))
    else:
        # Estimate raw pi0 values ("contaminated" for small lambdas by the true
        # target distribution
        W = count_larger(pvals, lambdas)
        pi0s_raw = W / (N * (1.0 - lambdas))

        if method == "smoother":
            # Now smooth it with a smoothing spline and evaluate
            pi0_spline_est = sp.interpolate.UnivariateSpline(
                lambdas, pi0s_raw, k=3, ext=0
            )
            pi0s_smooth = pi0_spline_est(lambdas)
            pi0 = pi0_spline_est(eval_lambda)
        elif method == "bootstrap":
            pi0_min = np.quantile(pi0s_raw, 0.1)
            mse = (
                W / (N**2 * (1 - lambdas) ** 2) * (1 - W / N)
                + (pi0s_raw - pi0_min) ** 2
            )
            pi0 = pi0s_raw[np.argmin(mse)]
        else:
            raise ValueError(f"Unknown pi0-estimation method {method}")

    pi0 = np.clip(pi0, 0, 1)

    return Pi0Est(pi0, pi0s_smooth, pi0s_raw, lambdas, mse)


@typechecked
def qvalues(
    pvals: np.ndarray[float],
    *,
    pi0: float | None = None,
    small_p_correction: bool = False,
) -> np.ndarray[float]:
    """
    Calculates q-values, which are an estimate of false discovery rates (FDR),
    for an array of p-values. This function is based on Storey's implementation
    in the R package "qvalue".

    Parameters
    ----------
    pvals : np.ndarray[float]
        An array of p-values to compute q-values for. The array should contain
        values between 0 and 1.
    pi0 : float, optional
        Proportion of true null hypotheses. If not provided, it is estimated
        using the bootstrap method (see estimate_pi0).
    small_p_correction : bool, optional
        Whether to apply a small p-value correction (Storey's pfdr parameter),
        which adjusts for very small p-values in the dataset.

    Returns
    -------
    np.ndarray[float]
        An array of q-values corresponding to the input p-values. The q-values
        are within the range [0, 1].

    References
    ----------
    .. [1] John D. Storey, Robert Tibshirani, Statistical significance for
        genomewide studies,pp. 9440 –9445 PNAS August 5, 2003, vol. 100, no. 16
        www.pnas.org/cgi/doi/10.1073/pnas.1530509100
    .. [2] John D. Storey, A direct approach to false discovery rates,
        J. R. Statist. Soc. B (2002), 64, Part 3, pp. 479–498
    .. [3] Storey JD, Bass AJ, Dabney A, Robinson D (2024). qvalue: Q-value
        estimation for false discovery rate control. R package version 2.38.0,
        http://github.com/jdstorey/qvalue.
    """
    N = len(pvals)
    order = np.argsort(pvals)
    pvals_sorted = pvals[order]
    if pi0 is None:
        pi0 = estimate_pi0(pvals, method="bootstrap")

    fdr_sorted = pi0 * pvals_sorted * N / np.linspace(1, N, N)
    if small_p_correction:
        fdr_sorted /= 1 - (1 - pvals) ** N

    # Note that the monotonization takes also correctly care of repeated pvalues
    # so that they always get the same qvalue
    qvalues_sorted = monotonize_simple(fdr_sorted, ascending=True, reverse=True)

    qvalues = np.zeros_like(qvalues_sorted)
    qvalues[order] = qvalues_sorted

    qvalues = np.clip(qvalues, 0.0, 1.0)
    return qvalues
