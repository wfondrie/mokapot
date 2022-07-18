"""
Defines a function to run the Percolator algorithm.
"""
import logging
import copy

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

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

    A list of previously trained models can be provided to the ``model``
    argument to rescore the PSMs in each fold. Note that the number of
    models must match ``folds``. Furthermore, it is valid to use the
    learned models on the same dataset from which they were trained,
    but they must be provided in the same order, such that the
    relationship of the cross-validation folds is maintained.

    Parameters
    ----------
    psms : PsmDataset object or list of PsmDataset objects
        One or more :doc:`collections of PSMs <dataset>` objects.
        PSMs are aggregated across all of the collections for model
        training, but the confidence estimates are calculated and
        returned separately.
    model: Model object or list of Model objects, optional
        The :py:class:`mokapot.Model` object to be fit. The default is
        :code:`None`, which attempts to mimic the same support vector
        machine models used by Percolator. If a list of
        :py:class:`mokapot.Model` objects is provided, they are assumed
        to be previously trained models and will and one will be
        used to rescore each fold.
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
        no additional effect. Note that logging messages will be garbled
        if more than one worker is enabled.

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
    if max_workers != 1:
        # train_sets can't be a generator for joblib :(
        train_sets = list(train_sets)

    # If trained models are provided, use the them as-is.
    try:
        fitted = [[m, False] for m in model if m.is_trained]
        assert len(fitted) == len(model)  # Test that all models are fitted.
        assert len(model) == folds
    except AssertionError as orig_err:
        if len(model) != folds:
            err = ValueError(
                f"The number of trained models ({len(model)}) "
                f"must match the number of folds ({folds})."
            )
        else:
            err = RuntimeError(
                "One or more of the provided models was not previously trained"
            )

        raise err from orig_err
    except TypeError:
        fitted = Parallel(n_jobs=max_workers, require="sharedmem")(
            delayed(_fit_model)(d, copy.deepcopy(model), f)
            for f, d in enumerate(train_sets)
        )

    # Sort models to have deterministic results with multithreading.
    fitted.sort(key=lambda x: x[0].fold)
    models, resets = list(zip(*fitted))

    # Determine if the models need to be reset:
    reset = any(resets)

    # If we reset, just use the original model on all the folds:
    if reset:
        scores = [
            p._calibrate_scores(model.predict(p), test_fdr) for p in psms
        ]

    # If we don't reset, assign scores to each fold:
    elif all([m.is_trained for m in models]):
        scores = [
            _predict(p, i, models, test_fdr) for p, i in zip(psms, test_idx)
        ]

    # If model training has failed
    else:
        scores = [np.zeros(len(p.data)) for p in psms]

    # Find which is best: the learned model, the best feature, or
    # a pretrained model.
    if not all([m.override for m in models]):
        best_feats = [p._find_best_feature(test_fdr) for p in psms]
        feat_total = sum([best_feat[1] for best_feat in best_feats])
    else:
        feat_total = 0

    preds = [p._update_labels(s, test_fdr) for p, s in zip(psms, scores)]
    pred_total = sum([(pred == 1).sum() for pred in preds])

    # Here, f[0] is the name of the best feature, and f[3] is a boolean
    if feat_total > pred_total:
        using_best_feat = True
        scores = []
        descs = []
        for dat, (feat, _, _, desc) in zip(psms, best_feats):
            scores.append(dat.data[feat].values)
            descs.append(desc)

    else:
        using_best_feat = False
        descs = [True] * len(psms)

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
        p.assign_confidence(s, eval_fdr=test_fdr, desc=d)
        for p, s, d in zip(psms, scores, descs)
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

    Returns
    -------
    numpy.ndarray
        A :py:class:`numpy.ndarray` containing the new scores.
    """
    test_set = copy.copy(dset)
    scores = []
    for fold_idx, mod in zip(test_idx, models):
        test_set._data = dset.data.loc[list(fold_idx), :]

        # Don't calibrate if using predict_proba.
        try:
            mod.estimator.decision_function
            scores.append(
                test_set._calibrate_scores(mod.predict(test_set), test_fdr)
            )
        except AttributeError:
            scores.append(mod.predict(test_set))
        except RuntimeError:
            raise RuntimeError(
                "Failed to calibrate scores between cross-validation folds, "
                "because no target PSMs could be found below 'test_fdr'. Try "
                "raising 'test_fdr'."
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
    model.fold = fold + 1
    reset = False
    try:
        model.fit(train_set)
    except RuntimeError as msg:
        if str(msg) != "Model performs worse after training.":
            raise

        if model.is_trained:
            reset = True

    return model, reset
