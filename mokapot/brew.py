"""
Defines a function to run the Percolator algorithm.
"""
import logging
import copy
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .model import Model
from .dataset import PsmDataset

LOGGER = logging.getLogger(__name__)


# Functions -------------------------------------------------------------------
def brew(psms,
         model=None,
         train_fdr=0.01,
         test_fdr=0.01,
         max_iter=10,
         folds=3,
         max_workers=1):
    """
    Re-score one or more collection of PSMs.

    The provided PSMs analyzed using the semi-supervised learning
    algorithm that was introduced by
    `Percolator <http://percolator.ms>`_. Cross-validation is used to
    ensure that the learned models to not overfit to the PSMs used for
    model training. If a multiple collections of PSMs are provided, they
    are aggregated for model training, but the confidence estimates are
    calculated separately for each collection.

    Parameters
    ----------
    psms : PsmDataset or list of PsmDataset
        One or more :doc:`PsmDataset <dataset>` objects. PSMs are
        aggregated across all of the collections for model training,
        but the confidence estimates are calculated and returned
        separately.
    model : Model, optional
        The :doc:`Model <model>` object to be fit. The default is
        `None`, which attempts to mimic the same support vector machine
        models used by Percolator.
    train_fdr : float, optional
        The false-discovery rate threshold to define positive examples
        during model training.
    test_fdr : float, optional
        The false-discovery rate threshold to evaluate whether the
        learned models yield more PSMs than the best feature.
    max_iter : int, optional
        The maximum number of iterations to use for training.
    folds : int, optional
        The number of cross-validation folds to use. PSMs originating
        from the same mass spectrum are always in the same fold.
    max_workers : int, optional
        The number of processes to use for model training. More workers
        will require more memory, but will typically decrease the total
        run time. An integer exceeding the number of folds will have
        no additional effect.

    Returns
    -------
    PsmConfidence or list of PsmConfidence
        An object or a list of objects containing the
        :doc:`confidence estimates <confidence>` at various levels
        (i.e. PSMs, peptides) when assessed using the learned score.
        If a list, they will be in the same order as provided in the
        `psms` parameter.
    """
    if model is None:
        model = Model()

    # TODO: Add group FDR estimates.
    all_idx = set(range(len(psms.data)))
    test_idx = psms._split(folds)

    train_sets = []
    test_sets = []
    for idx in test_idx:
        train_set = copy.copy(psms)
        train_set._data = psms.data.iloc[list(all_idx - set(idx)), :]
        train_sets.append(train_set)

        test_set = copy.copy(psms)
        test_set._data = psms.data.iloc[list(idx), :]
        test_sets.append(test_set)

    # Create args for map:
    map_args = [_fit_model,
                train_sets,
                [copy.deepcopy(model) for _ in range(folds)],
                [train_fdr]*folds,
                [max_iter]*folds]

    # Train models in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as prc:
        if max_workers == 1:
            map_fun = map
        else:
            map_fun = prc.map

        models = map_fun(*map_args)

    scores = [_predict(p, m, test_fdr) for p, m in zip(test_sets, models)]
    rev_idx = np.argsort(sum(test_idx, [])).tolist()
    scores = np.concatenate(scores)[rev_idx]

    return psms.assign_confidence(scores, desc=True)


# Utility Functions -----------------------------------------------------------
def _predict(psms, model, test_fdr):
    """
    Return calibrated scores for the PSMs.

    Parameters
    ----------
    psms : PsmDataset
         The PsmDataset to use.
    model : Model
         The model to use.
    test_fdr : float
         The FDR use for calibration.
    """
    return psms._calibrate_scores(model.predict(psms), fdr_threshold=test_fdr)

def _fit_model(train_set: PsmDataset, model: Model, train_fdr: float,
               max_iter: int) -> Model:
    """
    Fit the estimator using the training data.

    Parameters
    ----------
    train_set : PsmDataset
        A PsmDataset that specifies the training data
    model : Model
        A Classifier to train.
    train_fdr : float
        The FDR threshold used to define positive examples during the
        Percolator algorithm.
    max_iter : int
        The maximum number of iterations to run the algorithm.
    """
    model.fit(train_set, train_fdr=train_fdr, max_iter=max_iter)
    return model
