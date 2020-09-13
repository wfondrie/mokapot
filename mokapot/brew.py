"""
Defines a function to run the Percolator algorithm.
"""
import logging
import copy
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np

from .model import PercolatorModel

LOGGER = logging.getLogger(__name__)


# Functions -------------------------------------------------------------------
def brew(psms, model=None, test_fdr=0.01, folds=3, max_workers=1):
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
    psms : PsmDataset object or list of PsmDataset objects
        One or more :doc:`collections of PSMs <dataset>` objects.
        PSMs are aggregated across all of the collections for model
        training, but the confidence estimates are calculated and
        returned separately.
    model: Model object, optional
        The :py:class:`mokapot.Model` object to be fit. The default is
        :code:`None`, which attempts to mimic the same support vector
        machine models used by Percolator.
    test_fdr : float, optional
        The false-discovery rate threshold at which to evaluate
        the learned models.
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
    Confidence object or list of Confidence objects
        An object or a list of objects containing the
        :doc:`confidence estimates <confidence>` at various levels
        (i.e. PSMs, peptides) when assessed using the learned score.
        If a list, they will be in the same order as provided in the
        `psms` parameter.
    list of Model objects
        The learned :py:class:`~mokapot.model.Model` objects, one
        for each fold.
    """
    if model is None:
        model = PercolatorModel()

    try:
        iter(psms)
    except TypeError:
        psms = [psms]

    # Check that all of the datasets have the same features:
    feat_set = set(psms[0].features.columns)
    if not all([set(p.features.columns) == feat_set for p in psms]):
        raise ValueError("All collections of PSMs must use the same features.")

    if len(psms) > 1:
        LOGGER.info("")
        LOGGER.info("Found %i total PSMs.", sum([len(p.data) for p in psms]))

    LOGGER.info("Splitting PSMs into %i folds...", folds)
    test_idx = [p._split(folds) for p in psms]
    train_sets = _make_train_sets(psms, test_idx)

    # Create args for map:
    map_args = [
        _fit_model,
        train_sets,
        [copy.deepcopy(model) for _ in range(folds)],
        range(folds),
    ]

    # Train models in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as prc:
        if max_workers == 1:
            map_fun = map
        else:
            map_fun = prc.map

        models = list(map_fun(*map_args))

    # Determine if the models need to be reset:
    reset = any([m[1] for m in models])
    if reset:
        # If we reset, just use the original model on all the folds:
        scores = [
            p._calibrate_scores(model.predict(p), test_fdr) for p in psms
        ]
    elif all([m[0].is_trained for m in models]):
        # If we don't reset, assign scores to each fold:
        models = [m for m, _ in models]
        scores = [
            _predict(p, i, models, test_fdr) for p, i in zip(psms, test_idx)
        ]
    else:
        # If model training has failed
        scores = [np.zeros(len(p.data)) for p in psms]

    # Find which is best: the learned model, the best feature, or
    # a pretrained model.
    if not model.override:
        LOGGER.info("")
        best_feats = [p._find_best_feature(test_fdr) for p in psms]
        feat_total = sum([best_feat[1] for best_feat in best_feats])
    else:
        feat_total = 0

    preds = [p._update_labels(s, test_fdr) for p, s in zip(psms, scores)]
    pred_total = sum([(pred == 1).sum() for pred in preds])

    # Here, f[0] is the name of the best feature, and f[3] is a boolean
    if feat_total > pred_total:
        using_best_feat = True
        scores = [
            p.data[f[0]].values * int(f[3]) for p, f in zip(psms, best_feats)
        ]
    else:
        using_best_feat = False

    if using_best_feat:
        logging.warning(
            "Learned model did not improve over the best feature. "
            "Now scoring by the best feature for each collection "
            "of PSMs."
        )
    elif reset:
        logging.warning(
            "Learned model did not improve upon the pretrained "
            "input model. Now re-scoring each collection of PSMs "
            "using the original model."
        )

    LOGGER.info("")
    res = [
        p.assign_confidence(s, eval_fdr=test_fdr, desc=True)
        for p, s in zip(psms, scores)
    ]

    if len(res) == 1:
        return res[0], models

    return res, models


# Utility Functions -----------------------------------------------------------
def _make_train_sets(psms, test_idx):
    """
    Parameters
    ----------
    psms : list of PsmDataset
        The PsmDataset to get a subset of.
    test_idx : list of list of numpy.ndarray
        The indicies of the test sets

    Yields
    ------
    PsmDataset
        The training set.
    """
    train_set = copy.copy(psms[0])
    all_idx = [set(range(len(p.data))) for p in psms]
    for idx in zip(*test_idx):
        train_set._data = None
        data = []
        for i, j, dset in zip(idx, all_idx, psms):
            data.append(dset.data.loc[list(j - set(i)), :])

        train_set._data = pd.concat(data, ignore_index=True)
        yield train_set


def _predict(dset, test_idx, models, test_fdr):
    """
    Return the new scores for the dataset

    Parameters
    ----------
    dset : PsmDataset
        The dataset to rescore
    test_idx : list of numpy.ndarray
        The indicies of the test sets
    models : list of Model
        The models for each dataset and whether it
        was reset or not.
    test_fdr : the fdr to calibrate at.
    """
    test_set = copy.copy(dset)
    scores = []
    for fold_idx, mod in zip(test_idx, models):
        test_set._data = dset.data.loc[list(fold_idx), :]
        scores.append(
            test_set._calibrate_scores(mod.predict(test_set), test_fdr)
        )

    rev_idx = np.argsort(sum(test_idx, [])).tolist()
    return np.concatenate(scores)[rev_idx]


def _fit_model(train_set, model, fold):
    """
    Fit the estimator using the training data.

    Parameters
    ----------
    train_set : PsmDataset
        A PsmDataset that specifies the training data
    model : tuple of Model
        A Classifier to train.

    Returns
    -------
    model : mokapot.model.Model
        The trained model.
    reset : bool
        Whether the models should be reset to their original parameters.
    """
    LOGGER.info("")
    LOGGER.info("=== Analyzing Fold %i ===", fold + 1)
    reset = False
    try:
        model.fit(train_set)
    except RuntimeError as msg:
        if str(msg) != "Model performs worse after training.":
            raise

        if model.is_trained:
            reset = True

    return model, reset
