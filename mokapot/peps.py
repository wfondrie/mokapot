from __future__ import annotations

import logging
from typing import Callable, Iterator, TypeVar

import numpy as np
import scipy.stats as stats
from scipy.optimize import nnls
from scipy.optimize._nnls import _nnls
from triqler import qvality
from typeguard import typechecked

from mokapot.statistics import HistData

LOGGER = logging.getLogger(__name__)


PEP_ALGORITHM = {
    "qvality": lambda scores, targets, is_tdc: peps_from_scores_qvality(
        scores, targets, is_tdc, use_binary=False
    ),
    "qvality_bin": lambda scores, targets, is_tdc: peps_from_scores_qvality(
        scores, targets, is_tdc, use_binary=True
    ),
    "kde_nnls": lambda scores, targets, is_tdc: peps_from_scores_kde_nnls(
        scores, targets, is_tdc
    ),
    "hist_nnls": lambda scores, targets, is_tdc: peps_from_scores_hist_nnls(
        scores, targets, is_tdc
    ),
}


class PepsConvergenceError(Exception):
    """Raised when nnls iterations do not converge."""

    pass


@typechecked
def peps_from_scores(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    is_tdc: bool,
    pep_algorithm: str = "qvality",
) -> np.ndarray[float]:
    """Compute PEPs from scores.

    Parameters
    ----------
    scores:
        A numpy array containing the scores for each target and decoy peptide.
    targets:
        A boolean array indicating whether each peptide is a target (True) or a
        decoy (False).
    pep_algorithm:
        The PEPS calculation algorithm to use. Defaults to 'qvality'.
    is_tdc:
        Scores and targets come from competition, rather than separate search.
    pep_algorithm:
        PEPS algorithm to use. Defaults to 'qvality'. Possible values are the
        keys of `PEP_ALGORITHM`.

    Returns
    -------
    array:
        The PEPS (Posterior Error Probabilities) calculated using the specified
        algorithm.

    Raises
    ------
    ValueError
        If the specified algorithm is unknown.
    """
    pep_function = PEP_ALGORITHM[pep_algorithm]
    if pep_function is not None:
        return pep_function(scores, targets, is_tdc)
    else:
        raise ValueError(
            f"Unknown pep algorithm in peps_from_scores: {pep_algorithm}"
        )


@typechecked
def peps_from_scores_qvality(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    is_tdc: bool,
    use_binary: bool = False,
) -> np.ndarray[float]:
    """Compute PEPs from scores using the triqler pep algorithm.

    Parameters
    ----------
    scores:
        A numpy array containing the scores for each target and decoy peptide.
    targets:
        A boolean array indicating whether each peptide is a target (True) or a
        decoy (False).
    is_tdc:
        Scores and targets come from competition, rather than separate search.
    use_binary:
        Whether to the binary (Percolator) version of qvality (True), or the
        Python (triqler) version (False). Defaults to False. If True, the
        compiled `qvality` binary must be on the shell search path.

    Returns
    -------
    array:
        A numpy array containing the posterior error probabilities (PEPs)
        calculated using the qvality method. The PEPs are calculated based on
        the provided scores and targets, and are returned in the same order as
        the targets array.
    """
    qvalues_from_scores = (
        qvality.getQvaluesFromScoresQvality
        if use_binary
        else qvality.getQvaluesFromScores
    )

    # Triqler returns the peps for reverse sorted scores, so we sort the scores
    # ourselves and later sort them back
    index = np.argsort(scores)[::-1]
    scores_sorted, targets_sorted = scores[index], targets[index]

    try:
        old_verbosity, qvality.VERB = qvality.VERB, 0
        _, peps_sorted = qvalues_from_scores(
            scores_sorted[targets_sorted],
            scores_sorted[~targets_sorted],
            includeDecoys=True,
            includePEPs=True,
            tdcInput=is_tdc,
        )
        if use_binary:
            peps_sorted = np.array(peps_sorted, dtype=float)

        inverse_idx = np.argsort(index)
        peps = peps_sorted[inverse_idx]
    except SystemExit as msg:
        if "no decoy hits available for PEP calculation" in str(msg):
            peps = np.zeros_like(scores)
        else:
            raise
    finally:
        qvality.VERB = old_verbosity

    return peps


_AnyArray = TypeVar("_AnyArray")


@typechecked
def monotonize_simple(
    x: _AnyArray, ascending: bool, reverse: bool = False
) -> _AnyArray:
    """Monotonizes the input array `x` in either ascending or descending order
    beginning from the start of the array.

    Parameters
    ----------
    x:
        Input array to be monotonized.
    ascending:
        Specifies whether to monotonize in ascending order (`True`) or
        descending order (`False`). Direction is always w.r.t. to the start of
        the array, independently of `reverse`.
    reverse:
         Specify whether the process should run from the start (`False`) or the
         end (`True`) of the array. Defaults to `False`.

    Returns
    -------
    array:
        Monotonized array `x`
    """
    if reverse:
        return monotonize_simple(x[::-1], not ascending, False)[::-1]

    if ascending:
        return np.maximum.accumulate(x)
    else:
        return np.minimum.accumulate(x)


@typechecked
def monotonize(
    x: np.ndarray[float], ascending: bool, simple_averaging: bool = False
) -> np.ndarray[float]:
    """Monotonizes the input array `x` in either ascending or descending order
    averaging over both directions.

    Note: it makes a difference whether you start with monotonization from the
    start or the end of the array.

    Parameters
    ----------
    x:
        Input array to be monotonized.
    ascending:
        Specifies whether to monotonize in ascending order (`True`) or
        descending order (`False`).
    simple_averaging:
        Specifies whether to use a simple average (`True`) or weighted average
        (`False`) when computing the average of the monotonized arrays. Only
        used if `average` is `True`. Default is `False`. Note: the weighted
        average tries to minimize the L2 difference in the change between the
        original and the returned arrays.

    Returns
    -------
    array:
        Monotonized array `x` based on the specified parameters.
    """
    x1 = monotonize_simple(x, ascending)
    if np.all(x1 == x):
        return x  # nothing to do here
    x2 = monotonize_simple(x[::-1], not ascending)[::-1]
    alpha = (
        0.5
        if simple_averaging
        else np.sum((x - x2) * (x1 - x2)) / np.sum((x1 - x2) * (x1 - x2))
    )
    return alpha * x1 + (1 - alpha) * x2


@typechecked
def monotonize_nnls(
    x: np.ndarray[float],
    w: np.ndarray[float] | None = None,
    ascending: bool = True,
) -> np.ndarray[float]:
    """Monotonizes a given array `x` using non-negative least squares (NNLS)
    optimization.

    The returned array is the monotone array `y` that minimized `x-y` in the
    L2-norm. The of all monotone arrays `y` is such that `x-y` has minimum
    mean squared error (MSE).

    Parameters
    ----------
    x:
        numpy array to be monotonized.
    w:
        numpy array containing weights. If None, equal weights are assumed.
    ascending:
        Boolean indicating whether the monotonized array should be in ascending
        order.

    Returns
    -------
    array:
        The monotonized array.
    """
    if not ascending:
        # If descending, just return the reversed output of the algo with
        # reversed inputs.
        return monotonize_nnls(x[::-1], None if w is None else w[::-1])[::-1]

    # Basically the algorithm works by solving
    #   x1 = d1
    #   x2 = d1 + d2
    #   x3 = d1 + d2 + d3
    # and so on for all non-negative di - or rather minimizing the sum of the
    # squared differences - and afterwards # taking xm1 = d1, xm2 = d1 + d2,
    # and so on as the monotonized values.
    # The first is the same as minimizing
    # ||x - A d|| for the matrix A = [1 0 0 0...; 1 1 0 0...; 1 1 1 0...; ...].
    # The second is just computing the cumsum of the d vector.
    N = len(x)
    A = np.tril(np.ones((N, N)))
    if w is not None:
        # We do the weighting by multiplying both sides (i.e. A and x) by
        # a diagonal matrix consisting of the square roots of the weights
        D = np.diag(np.sqrt(w))
        A = D.dot(A)
        x = D.dot(x)
    d, _ = nnls(A, x)
    xm = np.cumsum(d)
    return xm


def estimate_pi0_by_slope(
    target_pdf: np.ndarray[float],
    decoy_pdf: np.ndarray[float],
    threshold: float = 0.9,
):
    r"""Estimate pi0 using the slope of decoy vs target PDFs.

    The idea is that :math:`f_T(s) = \pi_0 f_D(s) + (1-\pi_0) f_{TT}(s)` and
    that for small scores `s` and a scoring function that sufficiently
    separates targets and decoys (or false targets) it holds that
    :math:`f_T(s) \simeq \pi_0 f_D(s)`.
    The algorithm works by determining the maximum of the decoy distribution
    and then estimating the slope of the target vs decoy density for all scores
    left of 90% of the maximum of the decoy distribution.

    Parameters
    ----------
    target_pdf:
        An estimate of the target PDF.
    decoy_pdf:
        An estimate of the decoy PDF.
    threshold:
        The threshold for selecting decoy PDF values (default is 0.9).

    Returns
    -------
    float:
        The estimated value of pi0.
    """
    max_decoy = np.max(decoy_pdf)
    last_index = np.argmax(decoy_pdf >= threshold * max_decoy)
    pi0_est, _ = np.polyfit(decoy_pdf[:last_index], target_pdf[:last_index], 1)
    return max(pi0_est, 1e-10)


@typechecked
def pdfs_from_scores(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    num_eval_scores: int = 500,
) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """Compute target and decoy probability density functions (PDFs) from
    scores using kernel density estimation (KDE).

    Parameters
    ----------
    scores:
        A numpy array containing the scores for each target and decoy peptide.
    targets:
        A boolean array indicating whether each peptide is a target (True) or a
        decoy (False).
    num_eval_scores:
        Number of evaluation scores to compute in the PDFs. Defaults to 500.

    Returns
    -------
    tuple:
        A tuple containing the evaluation scores, the target PDF, and the decoy
        PDF at those scores.
    """
    # Compute score range and evaluation points
    min_score = min(scores)
    max_score = max(scores)
    eval_scores = np.linspace(min_score, max_score, num=num_eval_scores)

    # Compute target and decoy pdfs
    target_scores = scores[targets]
    decoy_scores = scores[~targets]
    target_pdf_estimator = stats.gaussian_kde(target_scores)
    decoy_pdf_estimator = stats.gaussian_kde(decoy_scores)
    target_pdf = target_pdf_estimator.pdf(eval_scores)
    decoy_pdf = decoy_pdf_estimator.pdf(eval_scores)
    return eval_scores, target_pdf, decoy_pdf


@typechecked
def peps_from_scores_kde_nnls(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    is_tdc: bool,
    num_eval_scores: int = 500,
    pi0_estimation_threshold: float = 0.9,
) -> np.ndarray[float]:
    """Estimate peps from scores using density estimates and monotonicity.

    This method computes the estimated probabilities of target being
    incorrect (peps) based on the given scores and targets. It uses the
    following steps:

        1. Compute evaluation scores, target probability density function
           (PDF), and decoy probability density function evaluated at the
           given scores.
        2. Estimate pi0 and the number of correct targets using the target
           PDF, decoy PDF, and pi0EstThresh.
        3. Calculate the number of correct targets by subtracting the decoy
           PDF multiplied by pi0Est from the target PDF, and clip it to
           ensure non-negative values.
        4. Estimate peps by dividing the number of correct targets by the
           target PDF, and clip the result between 0 and 1.
        5. Monotonize the pep estimates.
        6. Linearly interpolate the pep estimates from the evaluation
           scores to the given scores of interest.
        7. Return the estimated probabilities of target being incorrect
           (peps).

    Parameters
    ----------
    scores:
        A numpy array containing the scores for each target and decoy peptide.
    targets:
        A boolean array indicating whether each peptide is a target (True) or
        a decoy (False).
    is_tdc:
        Scores and targets come from competition, rather than separate search.
    num_eval_scores:
        The number of evaluation scores to be computed. Default is 500.
    pi0_estimation_threshold:
        The threshold for pi0 estimation. Default is 0.9.

    Returns
    -------
    array:
        The estimated probabilities of target being incorrect (peps) for the
        given scores.
    """

    # Compute evaluation scores, and target and decoy pdfs
    # (evaluated at given scores)
    eval_scores, target_pdf, decoy_pdf = pdfs_from_scores(
        scores, targets, num_eval_scores
    )

    if is_tdc:
        factor = (~targets).sum() / targets.sum()
    else:
        # Estimate pi0 and estimate number of correct targets
        factor = estimate_pi0_by_slope(
            target_pdf, decoy_pdf, pi0_estimation_threshold
        )

    correct = target_pdf - decoy_pdf * factor
    correct = np.clip(correct, 0, None)

    # Estimate peps from #correct targets, clip it
    pepEst = np.clip(1.0 - correct / target_pdf, 0.0, 1.0)

    # Now monotonize using the NNLS algo putting more weight on areas with high
    # target density
    pepEst = monotonize_nnls(pepEst, w=target_pdf, ascending=False)

    # Linearly interpolate the pep estimates from the eval points to the scores
    # of interest.
    peps = np.interp(scores, eval_scores, pepEst)
    peps = np.clip(peps, 0.0, 1.0)
    return peps


def fit_nnls(n, k, ascending=True, *, weight_exponent=-1.0, erase_zeros=False):
    """Do monotone nnls fit on binomial model.

    This method performs a non-negative least squares (NNLS) fit on given
    input parameters 'n' and 'k', such that `k[i]` is close to `p[i] * n[i]` in
    a weighted least squared sense (weight is determined by
    `n[i] ** weightExponent`) and the `p[i]` are monotone.

    Note: neither `n` nor `k` need to integer valued or positive nor does `k`
    need to be between `0` and `n`.

    Parameters
    ----------
    n:
        The input array of length N
    k:
        The input array of length N
    ascending:
        Optional bool (Default value = True). Whether the result should be
        monotone ascending or descending.
    weight_exponent:
        Optional (Default value = -1.0). The weight exponent to use.
    erase_zeros:
        Optional (Default value = False). Whether 0s in `n` should be erased
        prior to fitting or not.

    Returns
    -------
    array:
        The monotonically increasing or decreasing array `p` of length N.

    """
    # For the basic idea of this algorithm (i.e. monotonize under constraints),
    # see the `monotonize_nnls` algorithm.  This is more or less the same, just
    # that the functional to be minimized is different here.
    if not ascending:
        n = n[::-1]
        k = k[::-1]

    N = len(n)
    D = np.diag(n)
    A = D @ np.tril(np.ones((N, N)))
    w = np.zeros_like(n, dtype=float)
    w[n != 0] = n[n != 0] ** (0.5 * weight_exponent)
    W = np.diag(w)
    R = np.eye(N)

    zeros = (n == 0).nonzero()[0]
    if len(zeros) > 0:
        A[zeros, zeros] = 1
        A[zeros, np.minimum(zeros + 1, N - 1)] = -1
        w[zeros] = 1
        k[zeros] = 0
        W = np.diag(w)

        if erase_zeros:
            # The following lines remove variables that will end up being the
            # same (matrix R) as well as equations that are essentially zero on
            # both sides U). However, since this is a bit tricky, and difficult
            # to maintain and does not seem to lower the condition of the
            # matrix substantially, it is only activated on demand and left
            # here more for further reference, in case it will be needed in the
            # future.
            nnz = n != 0
            nnz[-1] = True
            nnz2 = np.insert(nnz, 0, True)[:-1]

            def make_perm_mat(rows, cols):
                M = np.zeros((np.max(rows) + 1, np.max(cols) + 1))
                M[rows, cols] = 1
                return M

            R = make_perm_mat(np.arange(N), nnz2.cumsum() - 1)
            U = make_perm_mat(np.arange(sum(nnz)), nnz.nonzero()[0])
            W = U @ W

    # The default tolerance of nnls is too low, leading sometimes to
    # non-convergent iterations and subsequent failures. A good tolerance
    # should probably be related to the condition number of `W @ A` and to
    # the error in `W @ k` (numerical and statistical, where the latter is
    # probably much, much larger than the former). Since this is a) difficult
    # to estimate anyway and b) run-time consuming, we settle here for a
    # fixed tolerance, # which a) seems large enough to never lead to
    # non-convergence and b) is fitting for the typical condition numbers and
    # values of k seen in experiments.
    tol = 1e-7
    d, _, mode = _nnls(W @ A @ R, W @ k, tol=tol)
    if mode != 1:
        LOGGER.debug(
            "\t - Warning: nnls went into loop. Taking last solution."
        )
    p = np.cumsum(R @ d)

    if not ascending:
        return p[::-1]
    else:
        return p


@typechecked
class TDHistData:
    """ """

    targets: HistData
    decoys: HistData

    def __init__(
        self,
        bin_edges: np.ndarray[float],
        target_counts: np.ndarray[int],
        decoy_counts: np.ndarray[int],
    ):
        self.targets = HistData(bin_edges, target_counts)
        self.decoys = HistData(bin_edges, decoy_counts)

    @staticmethod
    def from_scores_targets(
        bin_edges: np.ndarray[float],
        scores: np.ndarray[float],
        targets: np.ndarray[bool],
    ) -> TDHistData:
        """Create TDHistData object from scores and targets."""
        return hist_data_from_scores(scores, targets, bin_edges)

    @staticmethod
    def from_score_target_iterator(
        bin_edges: np.ndarray[float], score_target_iterator: Iterator
    ) -> TDHistData:
        """Create TDHistData from an iterator over scores and targets."""
        return hist_data_from_iterator(score_target_iterator, bin_edges)

    def as_counts(
        self,
    ) -> tuple[np.ndarray[float], np.ndarray[int], np.ndarray[int]]:
        """Return bin centers and target and decoy counts."""
        return (
            self.targets.bin_centers,
            self.targets.counts,
            self.decoys.counts,
        )

    def as_densities(
        self,
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """Return bin centers and target and decoy densities."""
        return (
            self.targets.bin_centers,
            self.targets.density,
            self.decoys.density,
        )


@typechecked
def hist_data_from_scores(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    bins: np.ndarray[float] | str | None = None,
) -> TDHistData:
    """Generate histogram data from scores and target/decoy information.

    Parameters
    ----------
    scores:
        A numpy array containing the scores for each target and decoy peptide.
    targets:
        A boolean array indicating whether each peptide is a target (True) or a
        decoy (False).
    bins:
        Either: The number of bins to use for the histogram. Or: the edges of
        the bins to take. Or: None, which lets numpy determines the bins from
        all scores (which is the default).

    Returns
    -------
    TDHistData:
        A `TDHistData` object, encapsulating the histogram data.
    """
    if isinstance(bins, np.ndarray):
        bin_edges = bins
    else:
        bin_edges = np.histogram_bin_edges(scores, bins=bins or "scott")

    target_counts, _ = np.histogram(scores[targets], bins=bin_edges)
    decoy_counts, _ = np.histogram(scores[~targets], bins=bin_edges)

    return TDHistData(bin_edges, target_counts, decoy_counts)


@typechecked
def hist_data_from_iterator(
    score_target_iterator, bin_edges: np.ndarray[float]
) -> TDHistData:
    """Generate histogram data from scores and target/decoy information
    provided by an iterator.

    This is for streaming algorithms.

    Parameters
    ----------
    score_target_iterator:
        An iterator that yields scores and target/decoy information. For each
        iteration a tuple consisting of a score array and a corresponding
        target must be yielded.
    bin_edges:
        The bins to use for the histogram. Must be provided (since they cannot
        be determined at the start of the algorithm).

    Returns
    -------
    TDHistData:
        A `TDHistData` object, encapsulating the histogram data.
    """

    target_counts = np.zeros(len(bin_edges) - 1, dtype=int)
    decoy_counts = np.zeros(len(bin_edges) - 1, dtype=int)
    for scores, targets in score_target_iterator:
        target_counts += np.histogram(scores[targets], bins=bin_edges)[0]
        decoy_counts += np.histogram(scores[~targets], bins=bin_edges)[0]

    return TDHistData(bin_edges, target_counts, decoy_counts)


@typechecked
def estimate_trials_and_successes(
    decoy_counts: np.ndarray[int],
    target_counts: np.ndarray[int],
    is_tdc: bool,
    restrict: bool = True,
):
    """Estimate trials/successes (assuming a binomial model) from decoy and
    target counts.

    Parameters
    ----------
    decoy_counts:
        A numpy array containing the counts of decoy occurrences (histogram).
    target_counts:
        A numpy array containing the counts of target occurrences (histogram).
    is_tdc:
        Scores and targets come from competition, rather than separate search.
    restrict:
        A boolean indicating whether to restrict the estimated trials and
        successes per bin. If True, the estimated values will be bounded by a
        minimum of 0 and a maximum of the corresponding target count. If False,
        the estimated values will be unrestricted.

    Returns
    -------
    tuple:
        A tuple (n, k) where n is a numpy array representing the estimated
        trials per bin, and k is a numpy array representing the estimated
        successes per bin.
    """

    if is_tdc:
        factor = 1
    else:
        # Find correction factor (equivalent to pi0 for target/decoy density)
        factor = estimate_pi0_by_slope(target_counts, decoy_counts)

    # Estimate trials and successes per bin
    if restrict:
        n = np.maximum(target_counts, 1)
        k = np.ceil(factor * decoy_counts).astype(int)
        k = np.clip(k, 0, n)
    else:
        n = target_counts
        k = factor * decoy_counts
    return n, k


@typechecked
def peps_from_scores_hist_nnls(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    is_tdc: bool,
    scale_to_one: bool = False,
    weight_exponent: float = -1.0,
):
    """Calculate the PEP (Posterior Error Probability) estimates from scores
    and targets using the NNLS (Non-negative Least Squares) method.

    The algorithm follows the steps outlined below:
        1. Define joint bins for targets and decoys.
        2. Estimate the number of trials and successes inside each bin.
        3. Perform a monotone fit by minimizing the objective function
           || n - diag(p) * k || with weights n over monotone descending p.
        4. If scaleToOne is True and the first element of the pepEst array is
           less than 1, scale the pepEst array by dividing it by the first
           element.
        5. Linearly interpolate the pep estimates from the evaluation points to
           the scores of interest.
        6. Clip the interpolated pep estimates between 0 and 1 in case they
           went slightly out of bounds.
        7. Return the interpolated and clipped PEP estimates.

    Parameters
    ----------
    scores:
        numpy array containing the scores of interest.
    targets:
        numpy array containing the target values corresponding to each score.
    is_tdc:
        Scores and targets come from competition, rather than separate search.
    scale_to_one:
        Boolean value indicating whether to scale the PEP estimates to 1 for
        small score values. Default is False.

    Returns
    -------
    array:
        Array of PEP estimates at the scores of interest.
    """

    hist_data = hist_data_from_scores(scores, targets)
    peps_func = peps_func_from_hist_nnls(
        hist_data, is_tdc, scale_to_one, weight_exponent
    )
    return peps_func(scores)


@typechecked
def peps_func_from_hist_nnls(
    hist_data: TDHistData,
    is_tdc: bool,
    scale_to_one: bool = False,
    weight_exponent: float = -1.0,
) -> Callable[[np.ndarray[float]], np.ndarray[float]]:
    """Compute a function that calculates the PEP (Posterior Error Probability)
    estimates from scores and targets using the NNLS (Non-negative Least
    Squares) method.

    For a description see `peps_from_scores_hist_nnls`.

    Parameters
    ----------
    hist_data:
        Histogram data as `TDHistData` object.
    is_tdc:
        Scores and targets come from competition, rather than separate search.
    scale_to_one: Scale the result if the maximum PEP is smaller than 1.
         (Default value = False)
    weight_exponent:
         The weight exponent for the `fit_nnls` fit (see there, default 1).

    Returns
    -------
    function:
        A function that computes PEPs, given scores as input. Input must be an
        numpy array.
    """
    # Define joint bins for targets and decoys
    eval_scores, target_counts, decoy_counts = hist_data.as_counts()

    n, k = estimate_trials_and_successes(
        decoy_counts, target_counts, is_tdc, restrict=False
    )

    # Do monotone fit, minimizing || n - diag(p) * k || with weights n over
    # monotone descending p
    try:
        pep_est = fit_nnls(
            n, k, ascending=False, weight_exponent=weight_exponent
        )
    except RuntimeError as e:
        raise PepsConvergenceError from e

    if scale_to_one and pep_est[0] < 1:
        pep_est = pep_est / pep_est[0]

    # Linearly interpolate the pep estimates from the eval points to the scores
    # of interest (keeping monotonicity) clip in case we went slightly out of
    # bounds
    return lambda scores: np.clip(
        np.interp(scores, eval_scores, pep_est), 0, 1
    )
