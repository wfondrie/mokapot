"""
This module estimates q-values.
"""

from typing import Callable

import numba as nb
import numpy as np
from typeguard import typechecked

from mokapot.stats.peps import (
    estimate_pi0_by_slope,
    hist_data_from_scores,
    monotonize_simple,
    peps_from_scores_hist_nnls,
    TDHistData,
)

QVALUE_ALGORITHM = {
    "tdc": lambda scores, targets: tdc(scores, targets, desc=True),
    "from_peps": lambda scores, targets: qvalues_from_peps(
        scores, targets, is_tdc=True
    ),
    "from_counts": lambda scores, targets: qvalues_from_counts(
        scores, targets, is_tdc=True
    ),
}


@typechecked
def tdc(
    scores: np.ndarray[np.floating | np.integer],
    target: np.ndarray[bool | np.floating | np.integer],
    desc: bool = True,
):
    """Estimate q-values using target decoy competition.

    Estimates q-values using the simple target decoy competition method.
    For set of target and decoy PSMs meeting a specified score threshold,
    the false discovery rate (FDR) is estimated as:

    ...math:
        FDR = \frac{Decoys + 1}{Targets}

    More formally, let the scores of target and decoy PSMs be indicated as
    :math:`f_1, f_2, ..., f_{m_f}` and :math:`d_1, d_2, ..., d_{m_d}`,
    respectively. For a score threshold :math:`t`, the false discovery
    rate is estimated as:

    ...math:
        E\\{FDR(t)\\} = \frac{|\\{d_i > t; i=1, ..., m_d\\}| + 1}
        {\\{|f_i > t; i=1, ..., m_f|\\}}

    The reported q-value for each PSM is the minimum FDR at which that
    PSM would be accepted.

    Parameters
    ----------
    scores : numpy.ndarray of float
        A 1D array containing the score to rank by
    target : numpy.ndarray of bool
        A 1D array indicating if the entry is from a target or decoy
        hit. This should be boolean, where `True` indicates a target
        and `False` indicates a decoy. `target[i]` is the label for
        `metric[i]`; thus `target` and `metric` should be of
        equal length.
    desc : bool
        Are higher scores better? `True` indicates that they are,
        `False` indicates that they are not.

    Returns
    -------
    numpy.ndarray
        A 1D array with the estimated q-value for each entry. The
        array is the same length as the `scores` and `target` arrays.
    """
    # todo: I think the allowed data types are way to general and lenient. scores
    # should me maximally integer|floating (but better just float) and targets
    # should only be bool, nothing else. The rest is the job of the calling code.

    scores = np.array(scores)
    target = np.array(target)

    if scores.shape[0] != target.shape[0]:
        raise ValueError("'scores' and 'target' must be the same length")

    # Sort and estimate FDR
    if desc:
        srt_idx = np.argsort(-scores)
    else:
        srt_idx = np.argsort(scores)

    scores = scores[srt_idx]
    target = target[srt_idx]
    cum_targets = target.cumsum()
    cum_decoys = ((target - 1) ** 2).cumsum()
    num_total = cum_targets + cum_decoys

    # Handles zeros in denominator
    fdr = np.divide(
        (cum_decoys + 1),
        cum_targets,
        out=np.ones_like(cum_targets, dtype=np.float32),
        where=(cum_targets != 0),
    )

    # Calculate q-values
    unique_metric, indices = np.unique(scores, return_counts=True)

    # Some arrays need to be flipped so that we can loop through from
    # worse to best score.
    fdr = np.flip(fdr)
    num_total = np.flip(num_total)
    if not desc:
        unique_metric = np.flip(unique_metric)
        indices = np.flip(indices)

    qvals = _fdr2qvalue(fdr, num_total, unique_metric, indices)
    qvals = np.flip(qvals)
    qvals = qvals[np.argsort(srt_idx)]

    return qvals


@nb.njit
def _fdr2qvalue(fdr, num_total, met, indices):
    """Quickly turn a list of FDRs to q-values.

    All of the inputs are assumed to be sorted.

    Parameters
    ----------
    fdr : numpy.ndarray
        A vector of all unique FDR values.
    num_total : numpy.ndarray
        A vector of the cumulative number of PSMs at each score.
    met : numpy.ndarray
        A vector of the scores for each PSM.
    indices : tuple of numpy.ndarray
        Tuple where the vector at index i indicates the PSMs that
        shared the unique FDR value in `fdr`.

    Returns
    -------
    numpy.ndarray
        A vector of q-values.
    """
    min_q = 1
    qvals = np.ones(len(fdr))
    group_fdr = np.ones(len(fdr))
    prev_idx = 0
    for idx in range(met.shape[0]):
        next_idx = prev_idx + indices[idx]
        group = slice(prev_idx, next_idx)
        prev_idx = next_idx

        fdr_group = fdr[group]
        n_group = num_total[group]
        curr_fdr = fdr_group[np.argmax(n_group)]
        if curr_fdr < min_q:
            min_q = curr_fdr

        group_fdr[group] = curr_fdr
        qvals[group] = min_q

    return qvals


@typechecked
def qvalues_from_scores(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    qvalue_algorithm: str = "tdc",
):
    """Compute q-values from scores.

    Parameters
    ----------
    scores:
        A numpy array containing the scores for each target and decoy peptide.
    targets:
        A boolean array indicating whether each peptide is a target (True) or a
        decoy (False).
    qvalue_algorithm:
        The q-value calculation algorithm to use. Defaults to 'tdc' (mokapot
        builtin).

    Returns
    -------
    array:
        The q-values calculated using the specified algorithm.

    Raises
    ------
    ValueError
        If the specified algorithm is unknown.
    """
    qvalue_function = QVALUE_ALGORITHM[qvalue_algorithm]
    if qvalue_function is not None:
        return qvalue_function(scores, targets)
    else:
        raise ValueError(
            "Unknown qvalue algorithm in qvalues_from_scores:" f" {qvalue_algorithm}"
        )


@typechecked
def qvalues_from_peps(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    is_tdc: bool,
    peps: np.ndarray[float] | None = None,
):
    r"""Compute q-values from peps.

    Computation is done according to Käll et al. (Section 3.2, first formula)
    Non-parametric estimation of posterior error probabilities associated with
    peptides identified by tandem mass spectrometry Bioinformatics, Volume 24,
    Issue 16, August 2008, Pages i42-i48
    https://doi.org/10.1093/bioinformatics/btn294

    The formula used is:

    .. math:: q_{PEP}(x^t) = \min_{x'\ge {x^t}}
        \frac{\sum_{x\in\{y|y\ge x',y\in T\}}P(H_0|X=x)}
        {|\{y|y\ge x',y\in T\}|}

    Note: the formula in the paper has an :math:`x^t` in the denominator, which
    does not make a whole lot of sense. Shall probably be :math:`x'` instead.

    Parameters
    ----------
    scores:
        Array-like object representing the scores.
    targets:
        Boolean array-like object indicating the targets.
    peps:
        Array-like object representing the posterior error probabilities
        associated with the peptides. Default is None (then it's computed via
        the HistNNLS algorithm).
    is_tdc:
        Scores and targets come from competition, rather than separate search.

    Returns
    -------
    array:
        Array of q-values computed from peps.
    """

    if peps is None:
        peps = peps_from_scores_hist_nnls(scores, targets, is_tdc)

    # We need to sort scores in descending order for the formula to work
    # (it involves to cumsum's from the maximum scores downwards)
    ind = np.argsort(-scores)
    scores_sorted = scores[ind]
    targets_sorted = targets[ind]
    peps_sorted = peps[ind]

    target_scores = scores_sorted[targets_sorted]
    target_peps = peps_sorted[targets_sorted]
    target_fdr = target_peps.cumsum() / np.arange(1, len(target_peps) + 1, dtype=float)
    target_qvalues = monotonize_simple(target_fdr, ascending=True, reverse=True)

    # Note: we need to flip the arrays again, since interp needs scores in
    #   ascending order
    qvalues = np.interp(scores, np.flip(target_scores), np.flip(target_qvalues))
    return qvalues


@typechecked
def qvalues_from_counts(
    scores: np.ndarray[float], targets: np.ndarray[bool], is_tdc: bool
):
    r"""
    Compute qvalues from target/decoy counts.

    Computed according to Käll et al. (Section 3.2, second formula)
    Non-parametric estimation of posterior error probabilities associated with
    peptides identified by tandem mass spectrometry Bioinformatics, Volume 24,
    Issue 16, August 2008, Pages i42–i48
    https://doi.org/10.1093/bioinformatics/btn294

    The formula used is:

    .. math:: q_{TD}(x^t) = \pi_0 \frac{\#T}{\#D} \min_{x\ge {x^t}}
        {|\{y|y\ge x, y\in D\}|}/
        {|\{y|y\ge x, y\in T\}|}

    Note: the factor :math:`\frac{\#T}{\#D}` is not in the original equation,
    but should be there to account of lists of targets and decoys of different
    lengths.

    Note 2: for tdc the estimator #D/#T is used for $\pi_0$, effectively
    cancelling out the factor.

    Parameters
    ----------
    scores :
        Array-like object representing the scores.
    targets :
        Boolean array-like object indicating the targets.
    is_tdc:
        Scores and targets come from competition, rather than separate search.

    Returns
    -------
    array:
        Array of q-values computed from peps.
    """

    if is_tdc:
        factor = 1
    else:
        hist_data = hist_data_from_scores(scores, targets)
        eval_scores, target_density, decoy_density = hist_data.as_densities()
        pi0 = estimate_pi0_by_slope(target_density, decoy_density)
        target_decoy_ratio = targets.sum() / (~targets).sum()
        factor = pi0 * target_decoy_ratio

    # Sort by score, but take also care of multiple targets/decoys per score
    ind = np.lexsort((targets, -scores))
    targets_sorted = targets[ind]
    scores_sorted = scores[ind]
    fdr_sorted = (
        factor
        * ((~targets_sorted).cumsum() + 1)
        / np.maximum(targets_sorted.cumsum(), 1)
    )
    qvalues_sorted = monotonize_simple(fdr_sorted, ascending=True, reverse=True)

    # Extract unique scores and take qvalue from the last of them
    # (for each unique score we need the unique qvalue, the extra return values
    # are needed to get the correct one and to map them back later to the full
    # array. See np.unique and do the math ;)
    # Note: if all scores are uniq this step could be skipped and maybe some
    # cpu cycles saved.
    scores_uniq, idx_uniq, rev_uniq, cnt_uniq = np.unique(
        scores_sorted,
        return_index=True,
        return_counts=True,
        return_inverse=True,
    )
    qvalues_uniq = qvalues_sorted[idx_uniq + cnt_uniq - 1]

    # Map unique values back, but in original order
    qvalues = np.zeros_like(qvalues_sorted)
    qvalues[ind] = qvalues_uniq[rev_uniq]

    return np.clip(qvalues, 0.0, 1.0)


@typechecked
def qvalues_func_from_hist(
    hist_data: TDHistData, is_tdc: bool
) -> Callable[[np.ndarray[float]], np.ndarray[float]]:
    r"""Compute q-values from histogram counts.

    Compute qvalues from target/decoy counts according to Käll et al. (Section
    3.2, second formula), but from the histogram counts.

    Note that the formula is exact for the left edges of each histogram bin.
    For the interiors of the bins the q-values are linearly interpolated.

    Parameters
    ----------
    scores :
        Array-like object representing the scores.
    targets :
        Boolean array-like object indicating the targets.
    hist_data:
        Histogram data in form of a `TDHistData` object.
    is_tdc:
        Scores and targets come from competition, rather than separate search.

    Returns
    -------
    function:
        Function the computes an array of q-values from an array of scores.
    """

    _, target_counts, decoy_counts = hist_data.as_counts()
    if is_tdc:
        factor = 1
    else:
        factor = estimate_pi0_by_slope(target_counts, decoy_counts)

    targets_sum = np.flip(target_counts).cumsum()
    decoys_sum = np.flip(decoy_counts).cumsum()

    fdr_flipped = factor * (decoys_sum + 1) / np.maximum(targets_sum, 1)
    fdr_flipped = np.clip(fdr_flipped, 0.0, 1.0)
    qvalues_flipped = monotonize_simple(fdr_flipped, ascending=True, reverse=True)
    qvalues = np.flip(qvalues_flipped)

    # We need to append zero to end of the qvalues for right edge of the last
    # bin, the other q-values correspond to the left edges of the bins
    # (because of the >= in the formula for the counts)
    qvalues = np.append(qvalues, qvalues[-1])
    eval_scores = hist_data.targets.bin_edges

    return lambda scores: np.interp(scores, eval_scores, qvalues)


@typechecked
def qvalues_from_storeys_algo(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    pi0: float | None = None,
    decoy_qvals_by_interp: bool = True,
    pvalue_method: str = "conservative",
):
    """
    Calculates q-values for a set of scores using Storey's algorithm.

    The function computes empirical p-values, estimates the proportion of
    null hypotheses (`pi0`), and subsequently calculates the q-values,
    which are adjusted p-values used in multiple hypothesis testing.

    Parameters
    ----------
    scores : np.ndarray[float]
        Array of scores for which q-values are computed. Must be of
        type float.
    targets : np.ndarray[bool]
        Binary array indicating whether a score is a target (`True`) or
        a decoy (`False`). Must have the same shape as `scores`.
    pi0 : float, optional
        Proportion of null hypotheses in the dataset. If not provided,
        it is estimated from the p-values using the "smoother" method
        with evaluation at lambda = 0.5.
    decoy_qvals_by_interp : bool, optional
        Whether to calculate q-values for decoys by interpolating from
        target q-values (`True`) or calculate decoy q-values independently
        as 1:1 p-values (`False`). Defaults to `True`.
    pvalue_method : str, optional
        Method to calculate empirical p-values. Acceptable values are
        mode-based methods such as "conservative". Defaults to
        "conservative".

    Returns
    -------
    np.ndarray[float]
        An array of q-values corresponding to the input `scores`, where
        lower scores generally indicate a higher likelihood of being a
        target. The q-values are adjusted for multiple hypothesis testing.
    """
    import mokapot.stats.qvalues_storey as qvalues_storey

    stat1 = scores[targets]
    stat0 = scores[~targets]

    pvals1 = qvalues_storey.empirical_pvalues(stat1, stat0, mode=pvalue_method)

    if pi0 is None:
        pi0est = qvalues_storey.estimate_pi0(pvals1, method="smoother", eval_lambda=0.5)
        pi0 = pi0est.pi0

    qvals1 = qvalues_storey.qvalues(pvals1, pi0=pi0, small_p_correction=False)

    if decoy_qvals_by_interp:
        ind = np.argsort(stat1)
        qvals0 = np.interp(stat0, stat1[ind], qvals1[ind])
    else:
        pvals0 = qvalues_storey.empirical_pvalues(stat0, stat0, mode=pvalue_method)
        qvals0 = qvalues_storey.qvalues(pvals0, pi0=1)

    qvals = np.zeros_like(scores, dtype=float)
    qvals[targets] = qvals1
    qvals[~targets] = qvals0
    return qvals
