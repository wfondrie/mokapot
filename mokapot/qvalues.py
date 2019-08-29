"""
This module estimates q-values.
"""
import numpy as np
import numba as nb

def tdc(metric, target, desc=True):
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
        E\{FDR(t)\} = \frac{|\{d_i > t; i=1, ..., m_d\}| + 1}
        {\{|f_i > t; i=1, ..., m_f|\}}

    The reported q-value for each PSM is the minimum FDR at which that
    PSM would be accepted.

    Parameters
    ----------
    metric : numpy.ndarray
        A 1D array containing the score to rank by (`float`)

    target : numpy.ndarray
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
        array is the same length as the `metric` and `target` arrays.
    """
    # Check arguments
    msg = "'{}' must be a 1D numpy.ndarray or pandas.Series"
    if not isinstance(metric, np.ndarray) or len(metric.shape) != 1:
        raise ValueError(msg.format("metric"))

    if not isinstance(target, np.ndarray) or len(target.shape) != 1:
        raise ValueError(msg.format("target"))

    if not isinstance(desc, bool):
        raise ValueError("'desc' must be boolean (True or False)")

    if metric.shape[0] != target.shape[0]:
        raise ValueError("'metric' and 'target' must be the same length")

    try:
        target = np.array(target, dtype=bool)
    except ValueError:
        raise ValueError("'target' should be boolean.")

    # Sort and estimate FDR
    if desc:
        srt_idx = np.argsort(-metric)
    else:
        srt_idx = np.argsort(metric)

    metric = metric[srt_idx]
    target = target[srt_idx]
    cum_targets = target.cumsum()
    cum_decoys = ((target-1)**2).cumsum()
    num_total = cum_targets + cum_decoys

    fdr = (cum_decoys + 1) / cum_targets

    # Calculate q-values
    unique_metric, indices = np.unique(metric, return_counts=True)

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
    """Quickly turn a list of FDRs to q-values"""
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
