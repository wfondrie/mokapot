"""Estimate q-values."""
import numba as nb
import numpy as np


def tdc(
    scores: np.ndarray,
    target: np.ndarray,
    desc: bool | None = True,
) -> np.ndarray:
    r"""
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

    desc : bool or None, optional
        Are higher scores better? `True` indicates that they are,
        `False` indicates that they are not. `None` indicates that
        the data are already sorted.

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

    # Sort only if required:
    if desc is not None:
        if desc:
            srt_idx = np.argsort(-scores)
        else:
            srt_idx = np.argsort(scores)

        scores = scores[srt_idx]
        target = target[srt_idx]

    # Compute FDR
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

    if desc is not None:
        qvals = qvals[np.argsort(srt_idx)]

    return qvals


@nb.njit
def _fdr2qvalue(
    fdr: np.ndarry,
    num_total: np.ndarray,
    met: np.ndarray,
    indices: tuple[np.ndarray],
) -> np.ndarray:
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
