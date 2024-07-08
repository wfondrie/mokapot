import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from triqler import qvality

# TODO: Remove the next and uncomment the 2nd next line when
#  scipy.optimize.nnls is fixed (see _nnls.py for explanation)
from ._nnls import nnls

# from scipy.optimize import nnls

PEP_ALGORITHM = {
    "qvality": lambda scores, targets: peps_from_scores_qvality(
        scores, targets, use_binary=False
    ),
    "qvality_bin": lambda scores, targets: peps_from_scores_qvality(
        scores, targets, use_binary=True
    ),
    "kde_nnls": lambda scores, targets: peps_from_scores_kde_nnls(
        scores, targets
    ),
    "hist_nnls": lambda scores, targets: peps_from_scores_hist_nnls(
        scores, targets
    ),
}


def peps_from_scores(scores, targets, pep_algorithm="qvality"):
    """
    Compute PEPs from scores.

    :param scores: A numpy array containing the scores for each
        target and decoy peptide.
    :param targets: A boolean array indicating whether each peptide is a target (True)
        or a decoy (False).
    :param pep_algorithm: The PEPS calculation algorithm to use. Defaults to 'qvality'.

    :return: The PEPS (Posterior Error Probabilities) calculated using the specified
        algorithm.

    :raises AssertionError: If the specified algorithm is unknown.
    """  # noqa: E501
    pep_function = PEP_ALGORITHM[pep_algorithm]
    if pep_function is not None:
        return pep_function(scores, targets)
    else:
        raise AssertionError(
            f"Unknown pep algorithm in peps_from_scores: {pep_algorithm}"
        )


def peps_from_scores_qvality(scores, targets, use_binary=False):
    """
    Compute PEPs from scores using the triqler pep algorithm.

    :param scores: A numpy array containing the scores for each target and decoy peptide.
    :param targets: A boolean array indicating whether each peptide is a target (True) or a decoy (False).
    :param use_binary: Whether to the binary (Percolator) version of qvality (True), or the Python (triqler) version
        (False). Defaults to False. If True, the compiled `qvality` binary must be on the shell search path.
    :return: A numpy array containing the posterior error probabilities (PEPs) calculated using the qvality
        method. The PEPs are calculated based on the provided scores and targets, and are returned in the same order
        as the targets array.
    """  # noqa: E501
    # todo: this method should contain the logic of sorting the scores
    #   (and the returned peps afterwards)
    # todo: should also do the error handling, since getQvaluesFromScores may
    # throw a SystemExit exception
    qvalues_from_scores = (
        qvality.getQvaluesFromScoresQvality
        if use_binary
        else qvality.getQvaluesFromScores
    )
    old_verbosity, qvality.VERB = qvality.VERB, 0
    _, peps = qvalues_from_scores(
        scores[targets],
        scores[~targets],
        includeDecoys=True,
        includePEPs=True,
        tdcInput=False,
    )
    qvality.VERB = old_verbosity
    return peps


def monotonize_simple(x, ascending):
    """
    Monotonizes the input array `x` in either ascending or descending order beginning from the start of the array.

    :param x: Input array to be monotonized.
    :param ascending: Specifies whether to monotonize in ascending order (`True`) or descending order (`False`).
    :return: Monotonized array `x`
    """  # noqa: E501
    if ascending:
        return np.maximum.accumulate(x)
    else:
        return np.minimum.accumulate(x)


def monotonize(x, ascending, simple_averaging=False):
    """
    Monotonizes the input array `x` in either ascending or descending order averaging over both directions. Note: it
    makes a difference whether you start with monotonization from the start or the end of the array.

    :param x: Input array to be monotonized.
    :param ascending: Specifies whether to monotonize in ascending order (`True`) or descending order (`False`).
    :param simple_averaging: Specifies whether to use a simple average (`True`) or weighted average (`False`) when
        computing the average of the monotonized arrays. Only used if `average` is `True`. Default is `False`.
        Note: the weighted average tries to minimize the L2 difference in the change between the original and the
        returned arrays.
    :return: Monotonized array `x` based on the specified parameters.
    """  # noqa: E501
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


def monotonize_nnls(x, w=None, ascending=True):
    """
    Monotonizes a given array `x` using non-negative least squares (NNLS) optimization.

    :param x: numpy array to be monotonized.
    :param w: numpy array containing weights. If None, equal weights are assumed.
    :param ascending: Boolean indicating whether the monotonized array should be in ascending order.
    :return: Monotonized array.
    """  # noqa: E501
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


def estimate_pi0_by_slope(target_pdf, decoy_pdf, threshold=0.9):
    """
    Estimate pi0 using the slope of decoy vs target PDFs.
    The idea is that :math:`f_T(s) = \pi_0 f_D(s) + (1-\pi_0) f_{TT}(s)` and that for small scores `s` and a scoring
    function that sufficiently separates targets and decoys (or false targets) it holds that :math:`f_T(s) \simeq \pi_0 f_D(s)`.
    The algorithm works by determining the maximum of the decoy distribution and then estimating the slope of the target
    vs decoy density for all scores left of 90% of the maximum of the decoy distribution.

    :param target_pdf: An estimate of the target PDF.
    :param decoy_pdf: An estimate of the decoy PDF.
    :param threshold: The threshold for selecting decoy PDF values (default is 0.9).
    :return: The estimated value of pi0.

    """  # noqa: E501
    max_decoy = np.max(decoy_pdf)
    last_index = np.argmax(decoy_pdf >= threshold * max_decoy)
    pi0_est, _ = np.polyfit(decoy_pdf[:last_index], target_pdf[:last_index], 1)
    return max(pi0_est, 1e-10)


def pdfs_from_scores(scores, targets, num_eval_scores=500):
    """
    Compute target and decoy probability density functions (PDFs) from scores using kernel density estimation (KDE).

    :param scores: A numpy array containing the scores for each target and decoy peptide.
    :param targets: A boolean array indicating whether each peptide is a target (True) or a decoy (False).
    :param num_eval_scores: Number of evaluation scores to compute in the PDFs. Defaults to 500.
    :return: A tuple containing the evaluation scores, the target PDF, and the decoy PDF at those scores.
    """  # noqa: E501
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


def peps_from_scores_kde_nnls(
    scores, targets, num_eval_scores=500, pi0_estimation_threshold=0.9
):
    """
    :param scores: A numpy array containing the scores for each target and decoy peptide.
    :param targets: A boolean array indicating whether each peptide is a target (True) or a decoy (False).
    :param num_eval_scores: The number of evaluation scores to be computed. Default is 500.
    :param pi0_estimation_threshold: The threshold for pi0 estimation. Default is 0.9.
    :return: The estimated probabilities of target being incorrect (peps) for the given scores.

    This method computes the estimated probabilities of target being incorrect (peps) based on the given scores and targets. It uses the following steps:
        1. Compute evaluation scores, target probability density function (PDF), and decoy probability density function
           (PDF) evaluated at the given scores.
        2. Estimate pi0 and the number of correct targets using the target PDF, decoy PDF, and pi0EstThresh.
        3. Calculate the number of correct targets by subtracting the decoy PDF multiplied by pi0Est from the target PDF,
           and clip it to ensure non-negative values.
        4. Estimate peps by dividing the number of correct targets by the target PDF, and clip the result between 0 and 1.
        5. Monotonize the pep estimates.
        6. Linearly interpolate the pep estimates from the evaluation scores to the given scores of interest.
        7. Return the estimated probabilities of target being incorrect (peps).
    """  # noqa: E501

    # Compute evaluation scores, and target and decoy pdfs
    # (evaluated at given scores)
    eval_scores, target_pdf, decoy_pdf = pdfs_from_scores(
        scores, targets, num_eval_scores
    )

    # Estimate pi0 and estimate number of correct targets
    pi0_est = estimate_pi0_by_slope(
        target_pdf, decoy_pdf, pi0_estimation_threshold
    )

    correct = target_pdf - decoy_pdf * pi0_est
    correct = np.clip(correct, 0, None)

    # Estimate peps from #correct targets, clip it
    pepEst = np.clip(1.0 - correct / target_pdf, 0, 1)

    # Now monotonize using the NNLS algo putting more weight on areas with high
    # target density
    pepEst = monotonize_nnls(pepEst, w=target_pdf, ascending=False)

    # Linearly interpolate the pep estimates from the eval points to the scores
    # of interest.
    peps = np.interp(scores, eval_scores, pepEst)
    peps = np.clip(peps, 0, 1)
    return peps


def fit_nnls(n, k, ascending=True, *, weight_exponent=1, erase_zeros=False):
    """
    This method performs a non-negative least squares (NNLS) fit on given input parameters 'n' and 'k', such that
    `k[i]` is close to `p[i] * n[i]` in a weighted least squared sense (weight is determined by `n[i] ** weightExponent`)
    and the `p[i]` are monotone.

    Note: neither `n` nor `k` need to integer valued or positive nor does `k` need to be between `0` and `n`.

    Parameters:
        - n: The input array of length N.
        - k: The input array of length N.
        - ascending: (optional) A boolean value indicating whether the output array should be in ascending order. Default value is True.
        - weight_exponent: (optional) The exponent to be used for the weight array. Default value is 1.
        - erase_zeros: (optional) If True, rows corresponding to n==0 will be erased from the system of equations,
            whereas if False (default), there will be equations inserted that try to minimize the jump in probabilities,
            i.e. distribute the jumps evenly
    Returns:
        - p: The monotonically increasing or decreasing array of length N.

    """  # noqa: E501
    # For the basic idea of this algorithm, see the `monotonize_nnls`
    # algorithm.
    # This is more or less the same, just with
    # JSPP Q: With what??
    if not ascending:
        return fit_nnls(n[::-1], k[::-1])[::-1]
    N = len(n)
    D = np.diag(n)
    A = D @ np.tril(np.ones((N, N)))
    w = n ** (0.5 * weight_exponent)
    W = np.diag(w)

    nz = (n == 0).nonzero()[0]
    if len(nz) > 0:
        if not erase_zeros:
            A[nz, nz] = 1
            A[nz, np.minimum(nz + 1, N - 1)] = -1
            w[nz] = 1
            k[nz] = 0
            W = np.diag(w)
        else:
            W = np.delete(W, nz, axis=0)

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
    d, _ = nnls(W @ A, W @ k, atol=tol)
    p = np.cumsum(d)
    return p


def hist_data_from_scores(scores, targets, bins=None, density=False):
    """
    Generate histogram data from scores and target/decoy information.

    :param scores: A numpy array containing the scores for each target and decoy peptide.
    :param targets: A boolean array indicating whether each peptide is a target (True) or a decoy (False).
    :param bins: The number of bins to use for the histogram. Defaults to None (which lets numpy determines the bins from all scores).
    :param density: If True, the histogram is normalized to form a probability density. Defaults to False.

    :return: A tuple of three numpy arrays: the evaluation scores, target counts, and decoy counts.
             The evaluation scores are the midpoint of each bin.
             The target counts represent the number of target scores in each bin.
             The decoy counts represent the number of decoy scores in each bin.
    """  # noqa: E501
    if bins is None:
        bins = np.histogram_bin_edges(scores, bins="auto")
    target_counts, _ = np.histogram(
        scores[targets], bins=bins, density=density
    )
    decoy_counts, _ = np.histogram(
        scores[~targets], bins=bins, density=density
    )
    eval_scores = 0.5 * (bins[:-1] + bins[1:])
    return eval_scores, target_counts, decoy_counts


def estimate_trials_and_successes(decoy_counts, target_counts, restrict=True):
    """
    Estimate trials/successes (assuming a binomial model) from decoy and target counts.

    :param decoy_counts: A numpy array containing the counts of decoy occurrences (histogram).
    :param target_counts: A numpy array containing the counts of target occurrences (histogram).
    :param restrict: A boolean indicating whether to restrict the estimated trials and successes per bin.
                     If True, the estimated values will be bounded by a minimum of 0 and a maximum of the corresponding target count.
                     If False, the estimated values will be unrestricted.
    :return: A tuple (n, k) where n is a numpy array representing the estimated trials per bin, and k is a numpy array representing the estimated successes per bin.

    """  # noqa: E501
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


def peps_from_scores_hist_nnls(scores, targets, scale_to_one=True):
    """
    :param scores: numpy array containing the scores of interest.
    :param targets: numpy array containing the target values corresponding to each score.
    :param scale_to_one: Boolean value indicating whether to scale the PEP estimates to 1 for small score values. Default is True.
    :return: Array of PEP estimates at the scores of interest.

    This method calculates the PEP (Posterior Error Probability) estimates from scores and targets using the NNLS
    (Non-negative Least Squares) method. The algorithm follows the steps outlined below.

    1. Define joint bins for targets and decoys.
    2. Estimate the number of trials and successes inside each bin.
    3. Perform a monotone fit by minimizing the objective function || n - diag(p) * k || with weights n over monotone descending p.
    4. If scaleToOne is True and the first element of the pepEst array is less than 1, scale the pepEst array by dividing it by the first element.
    5. Linearly interpolate the pep estimates from the evaluation points to the scores of interest.
    6. Clip the interpolated pep estimates between 0 and 1 in case they went slightly out of bounds.
    7. Return the interpolated and clipped PEP estimates.
    """  # noqa: E501
    # Define joint bins for targets and decoys
    eval_scores, target_counts, decoy_counts = hist_data_from_scores(
        scores, targets
    )
    n, k = estimate_trials_and_successes(
        decoy_counts, target_counts, restrict=False
    )

    # Do monotone fit, minimizing || n - diag(p) * k || with weights n over
    # monotone descending p
    pep_est = fit_nnls(n, k, ascending=False)

    if scale_to_one and pep_est[0] < 1:
        pep_est = pep_est / pep_est[0]

    # Linearly interpolate the pep estimates from the eval points to the
    # scores of interest (keeping monotonicity) clip in case we went
    # slightly out of bounds
    return np.clip(np.interp(scores, eval_scores, pep_est), 0, 1)


def peps_from_scores_hist_direct(scores, targets):
    """
    Compute a PEP estimate directly from the binned scores (histogram) without any monotonization, just restricting
    peps between 0 and 1.

    :param scores: A numpy array of scores.
    :param targets: A numpy array of target labels corresponding to the scores.
    :return: A numpy array of estimated PEP (Posterior Error Probability) values based on the scores.
    """  # noqa: E501
    # Define joint bins for targets and decoys
    eval_scores, target_counts, decoy_counts = hist_data_from_scores(
        scores, targets
    )

    # Estimate number of trials and successes per bin
    n, k = estimate_trials_and_successes(decoy_counts, target_counts)

    # Direct "no-frills" estimation of the PEP without monotonization
    # or anything else
    pep_est = k / n

    # Linearly interpolate the pep estimates from the eval points to the
    # scores of interest (keeping monotonicity) clip in case we went slightly
    # out of bounds
    return np.clip(np.interp(scores, eval_scores, pep_est), 0, 1)


def plot_peps(
    scores,
    targets,
    ax=None,
    peps_true=None,
    show_pdfs=True,
    show_hists=True,
    show_qvality=True,
    show_kde_nnls=True,
    show_hist_nnls=True,
    show_peps_direct=True,
):
    if ax is None:
        plt.cla()
        plt.clf()
        ax = plt.gca()
        ax.clear()
    if show_pdfs:
        eval_scores, target_pdf, decoy_pdf = pdfs_from_scores(
            scores, targets, 200
        )
        ax.plot(eval_scores, target_pdf, label="Target PDF")
        ax.plot(eval_scores, decoy_pdf, label="Decoy PDF")
    if show_hists:
        bins = np.histogram_bin_edges(scores, bins="auto")
        ax.hist(scores[~targets], bins=bins, density=True, color=("C1", 0.5))
        ax.hist(scores[targets], bins=bins, density=True, color=("C0", 0.5))
    if show_qvality:
        peps_qv = (
            peps_from_scores_qvality(scores, targets, use_binary=False) + 0.01
        )
        ax.plot(scores, peps_qv, label="Mokapot (triqler)")
    if show_kde_nnls:
        peps_km = peps_from_scores_kde_nnls(scores, targets, 200)
        ax.plot(scores, peps_km, label="KDEnnls")
    if show_hist_nnls:
        peps_bg = peps_from_scores_hist_nnls(scores, targets)
        ax.plot(scores, peps_bg, label="HistNNLS")
    if show_qvality:
        import shutil

        if shutil.which("qvality") is not None:
            peps_qv = peps_from_scores_qvality(
                scores, targets, use_binary=True
            )
            ax.plot(scores, peps_qv, label="QVality C++")
    if show_peps_direct:
        peps_bg = peps_from_scores_hist_direct(scores, targets)
        ax.plot(
            scores, peps_bg, label="Direct", color=("k", 0.5), linewidth=0.5
        )
    if peps_true is not None:
        ax.plot(
            scores, peps_true, color=("k", 0.5), linestyle="--", label="Truth"
        )
    ax.set_xlabel("Score")
    ax.set_ylabel("Prob.")
    ax.legend(loc="lower right")

    bin_edges, target_counts, decoy_counts = hist_data_from_scores(
        scores, targets
    )
    delta = bin_edges[1] - bin_edges[0]

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    inset_ax = inset_axes(ax, loc="upper right", width="30%", height="30%")
    # Plot some data in the inset subplot
    if show_pdfs:
        inset_ax.plot(decoy_pdf, target_pdf, color="C0", label="pdf")
    inset_ax.plot(
        decoy_counts / (delta * sum(decoy_counts)),
        target_counts / (delta * sum(target_counts)),
        color="C1",
        label="hist",
    )
    # Set labels, legend, grid, and aspect
    inset_ax.set_xlabel("decoy")
    inset_ax.set_ylabel("target")
    inset_ax.grid(True)
    inset_ax.set_aspect("equal")
    return ax
