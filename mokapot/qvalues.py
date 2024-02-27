"""
This module estimates q-values.
"""

import numpy as np
import numba as nb

from .peps import peps_from_scores_hist_nnls, monotonize_simple, hist_data_from_scores, estimate_pi0_by_slope

QVALUE_ALGORITHM = {
    'tdc': lambda scores, targets: tdc(scores, targets, desc=True),
    'from_peps': lambda scores, targets: qvalues_from_peps(scores, targets),
    'from_counts': lambda scores, targets: qvalues_from_counts(scores, targets),
}


def tdc(scores, target, desc=True):
    """
    Estimate q-values using target decoy competition.

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
    scores = np.array(scores)

    try:
        target = np.array(target, dtype=bool)
    except ValueError:
        raise ValueError("'target' should be boolean.")

    if scores.shape[0] != target.shape[0]:
        raise ValueError("'scores' and 'target' must be the same length")

    # Unsigned integers can cause weird things to happen.
    # Convert all scores to floats to for safety.
    if np.issubdtype(scores.dtype, np.integer):
        scores = scores.astype(np.float_)

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
        out=np.ones_like(cum_targets, dtype=float),
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


def crosslink_tdc(scores, num_targets, desc=True):
    """
    Estimate q-values using the Walzthoeni et al method.

    This q-value method is specifically for cross-linked peptides.

    Parameters
    ----------
    num_targets : numpy.ndarray
        The number of target sequences in the cross-linked pair.

    metric : numpy.ndarray
        The metric to used to rank elements.

    desc : bool
        Is a higher metric better?

    Returns
    -------
    numpy.ndarray
        A 1D array of q-values in the same order as `num_targets` and
        `metric`.
    """
    scores = np.array(scores, dtype=np.float)
    num_targets = np.array(num_targets, dtype=np.int)

    if num_targets.max() > 2 or num_targets.min() < 0:
        raise ValueError("'num_targets' must be 0, 1, or 2.")

    if scores.shape[0] != num_targets.shape[0]:
        raise ValueError("'scores' and 'num_targets' must be the same length.")

    if desc:
        srt_idx = np.argsort(-scores)
    else:
        srt_idx = np.argsort(scores)

    scores = scores[srt_idx]
    num_targets = num_targets[srt_idx]
    num_total = np.ones(len(num_targets)).cumsum()
    cum_targets = (num_targets == 2).astype(int).cumsum()
    one_decoy = (num_targets == 1).astype(int).cumsum()
    two_decoy = (num_targets == 0).astype(int).cumsum()

    fdr = np.divide(
        (one_decoy - two_decoy),
        cum_targets,
        out=np.ones_like(cum_targets, dtype=float),
        where=(cum_targets != 0),
    )

    fdr[fdr < 0] = 0

    # Calculate q-values
    unique_metric, indices = np.unique(scores, return_counts=True)

    # Flip arrays to loop from worst to best score
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
    """
    Quickly turn a list of FDRs to q-values.

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


def qvalues_from_scores(scores, targets, qvalue_algorithm="tdc"):
    """
    Compute q-values from scores.

    :param scores: A numpy array containing the scores for each target and decoy peptide.
    :param targets: A boolean array indicating whether each peptide is a target (True) or a decoy (False).
    :param qvalue_algorithm: The q-value calculation algorithm to use. Defaults to 'tdc' (mokapot builtin).

    :return: The q-values calculated using the specified algorithm.

    :raises AssertionError: If the specified algorithm is unknown.
    """
    qvalue_function = QVALUE_ALGORITHM[qvalue_algorithm]
    if qvalue_function is not None:
        return qvalue_function(scores, targets)
    else:
        raise AssertionError(f"Unknown qvalue algorithm in qvalues_from_scores: {qvalue_algorithm}")


def qvalues_from_peps(scores, targets, peps=None):
    r"""
    Compute q-values from peps according to Käll et al. (Section 3.2, first formula)
    Non-parametric estimation of posterior error probabilities associated with peptides
    identified by tandem mass spectrometry
    Bioinformatics, Volume 24, Issue 16, August 2008, Pages i42–i48
    https://doi.org/10.1093/bioinformatics/btn294

    The formula used is:

    .. math:: q_{PEP}(x^t) = \min_{x'\ge {x^t}} \frac{\sum_{x\in\{y|y\ge x',y\in T\}}P(H_0|X=x)}{|\{y|y\ge x',y\in T\}|}

    Note: the formula in the paper has an :math:`x^t` in the denominator, which does not make a whole lot of sense.
    Shall probably be :math:`x'` instead.

    :param scores: Array-like object representing the scores.
    :param targets: Boolean array-like object indicating the targets.
    :param peps: Array-like object representing the posterior error probabilities associated
                 with the peptides. Default is None (then it's computed via the HistNNLS algorithm).
    :return: Array of q-values computed from peps.
    """
    if peps is None:
        peps = peps_from_scores_hist_nnls(scores, targets)

    # We need to sort scores in descending order for the formula to work (it involves to cumsum's from the maximum
    # scores downwards)
    ind = np.argsort(-scores)
    scores_sorted = scores[ind]
    targets_sorted = targets[ind]
    peps_sorted = peps[ind]

    target_scores = scores_sorted[targets_sorted]
    target_peps = peps_sorted[targets_sorted]
    target_fdr = target_peps.cumsum() / np.arange(1, len(target_peps) + 1, dtype=float)
    target_qvalues = monotonize_simple(target_fdr, ascending=True)

    # Note: we need to flip the arrays again, since interp needs scores in ascending order
    qvalues = np.interp(scores, np.flip(target_scores), np.flip(target_qvalues))
    return qvalues


def qvalues_from_counts(scores, targets):
    r"""
    Compute qvalues from target/decoy counts according to Käll et al. (Section 3.2, second formula)
    Non-parametric estimation of posterior error probabilities associated with peptides
    identified by tandem mass spectrometry
    Bioinformatics, Volume 24, Issue 16, August 2008, Pages i42–i48
    https://doi.org/10.1093/bioinformatics/btn294

    The formula used is:

    .. math:: q_{TD}(x^t) = \min_{x\ge {x^t}} \pi_0 \frac{\#T}{\#D} {|\{y|y\ge x, y\in D\}|}/{|\{y|y\ge x, y\in T\}|}

    Note: the factor :math:`\frac{\#T}{\#D}` is not in the original equation, but should be there to account of lists
    of targets and decoys of different lengths.

    :param scores: Array-like object representing the scores.
    :param targets: Boolean array-like object indicating the targets.
    :return: Array of q-values computed from peps.
    """

    eval_scores, target_density, decoy_density = hist_data_from_scores(scores, targets, density=True)
    pi0 = estimate_pi0_by_slope(target_density, decoy_density)

    target_decoy_ratio = targets.sum() / (~targets).sum()

    ind = np.argsort(-scores)
    targets_sorted = targets[ind]
    scores_sorted = scores[ind]
    fdr_sorted = pi0 * target_decoy_ratio * (~targets_sorted).cumsum() / targets_sorted.cumsum()
    qvalues_sorted = monotonize_simple(fdr_sorted, ascending=True)

    # Note: we need to flip the arrays again, since interp needs scores in ascending order
    qvalues = np.interp(scores, np.flip(scores_sorted), np.flip(qvalues_sorted))
    return qvalues
