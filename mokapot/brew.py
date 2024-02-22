"""
Defines a function to run the Percolator algorithm.
"""
import logging
import copy
from operator import itemgetter

import numpy as np
from joblib import Parallel, delayed

from .model import PercolatorModel
from . import utils
from .dataset import (
    LinearPsmDataset,
    calibrate_scores,
    update_labels,
    read_file,
)
from .parsers.pin import (
    parse_in_chunks,
    read_file_in_chunks,
)
from .constants import (
    CHUNK_SIZE_ROWS_PREDICTION,
    CHUNK_SIZE_READ_ALL_DATA,
)

LOGGER = logging.getLogger(__name__)


# Functions -------------------------------------------------------------------
def brew(
    psms,
    model=None,
    test_fdr=0.01,
    folds=3,
    max_workers=1,
    rng=None,
    subset_max_train=None,
    ensemble=False,
):
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
    psms : Dict
        Contains all required info about the input data
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
    rng : int, np.random.Generator, optional
        A seed or generator used to generate splits, or None to use the
        default random number generator state.

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
    rng = np.random.default_rng(rng)
    if model is None:
        model = PercolatorModel()

    try:
        iter(psms)
    except TypeError:
        psms = [psms]

    try:
        model.estimator
        model.rng = rng
    except AttributeError:
        pass

        # Check that all of the datasets have the same features:
    feat_set = set(psms[0].feature_columns)
    if not all([set(_psms.feature_columns) == feat_set for _psms in psms]):
        raise ValueError("All collections of PSMs must use the same features.")

    data_size = [len(_psms.spectra_dataframe) for _psms in psms]
    if sum(data_size) > 1:
        LOGGER.info("Found %i total PSMs.", sum(data_size))
        num_targets = sum(
            [
                (_psms.spectra_dataframe[_psms.target_column]).sum()
                for _psms in psms
            ]
        )
        num_decoys = sum(
            [
                (~_psms.spectra_dataframe[_psms.target_column]).sum()
                for _psms in psms
            ]
        )
        LOGGER.info(
            "  - %i target PSMs and %i decoy PSMs detected.",
            num_targets,
            num_decoys,
        )
    LOGGER.info("Splitting PSMs into %i folds...", folds)
    test_folds_idx = [_psms._split(folds, rng) for _psms in psms]

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
        train_sets = list(
            make_train_sets(
                test_idx=test_folds_idx,
                subset_max_train=subset_max_train,
                data_size=data_size,
                rng=rng,
            )
        )
        train_psms = parse_in_chunks(
            psms=psms,
            train_idx=train_sets,
            chunk_size=CHUNK_SIZE_READ_ALL_DATA,
            max_workers=max_workers
        )
        del train_sets
        fitted = Parallel(n_jobs=max_workers, require="sharedmem")(
            delayed(_fit_model)(d, psms, copy.deepcopy(model), f)
            for f, d in enumerate(train_psms)
        )

    # Sort models to have deterministic results with multithreading.
    fitted.sort(key=lambda x: x[0].fold)
    models, resets = list(zip(*fitted))

    # Determine if the models need to be reset:
    reset = any(resets)

    # If we reset, just use the original model on all the folds:
    if reset:
        scores = [
            _psms.calibrate_scores(
                _predict_with_ensemble(
                    psms=_psms,
                    models=[model],
                    max_workers=max_workers),
                test_fdr
            )
            for _psms in psms
        ]

    # If we don't reset, assign scores to each fold:
    elif all([m.is_trained for m in models]):
        if ensemble:
            scores = [
                _predict_with_ensemble(
                    psms=_psms,
                    models=models,
                    max_workers=max_workers)
                for _psms in psms
            ]
        else:
            # generate model index for each psm in all folds
            model_to_psm_idx = [
                [[i] * len(idx) for i, idx in enumerate(test_fold_idx)]
                for test_fold_idx in test_folds_idx
            ]
            # sort test indices and model indices in the original order
            # (order of input data)
            original_order_idx = [
                np.argsort(utils.flatten(test_fold_idx)).tolist()
                for test_fold_idx in test_folds_idx
            ]
            del test_folds_idx
            model_to_psm_idx = [
                np.concatenate(model_idx)[idx]
                for model_idx, idx in zip(model_to_psm_idx, original_order_idx)
            ]
            del original_order_idx
            scores = list(
                _predict(
                    models_idx=model_to_psm_idx,
                    psms=psms,
                    models=models,
                    test_fdr=test_fdr,
                    max_workers=max_workers
                )
            )
    # If model training has failed
    else:
        scores = [np.zeros(data_size) for _ in psms]
    # Find which is best: the learned model, the best feature, or
    # a pretrained model.
    if not all([m.override for m in models]):
        best_feats = [[m.best_feat, m.feat_pass, m.desc] for m in models]
        best_feat_idx, feat_total = max(
            enumerate(map(itemgetter(1), best_feats)), key=itemgetter(1)
        )
    else:
        feat_total = 0

    preds = [
        update_labels(_psms.filename, s, _psms.target_column, test_fdr)
        for _psms, s in zip(psms, scores)
    ]

    pred_total = sum([(pred == 1).sum() for pred in preds])

    # Here, f[0] is the name of the best feature, and f[3] is a boolean
    if feat_total > pred_total:
        using_best_feat = True
        feat, _, desc = best_feats[best_feat_idx]
        descs = [desc] * len(psms)
        scores = [
            read_file(
                _psms.filename,
                use_cols=[feat],
            ).values
            for _psms in psms
        ]

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

    return psms, models, scores, descs


# Utility Functions -----------------------------------------------------------
def make_train_sets(test_idx, subset_max_train, data_size, rng):
    """
    Parameters
    ----------
    test_idx : list of list of numpy.ndarray
        The indicies of the test sets
    subset_max_train : int or None
        The number of PSMs for training.
    data_size : list[int]
        size of the input data

    Yields
    ------
    PsmDataset
        The training set.
    """
    subset_max_train_per_file = []
    if subset_max_train is not None:
        subset_max_train_per_file = [
            subset_max_train // len(data_size) for _ in range(len(data_size))
        ]
        subset_max_train_per_file[-1] += subset_max_train - sum(
            subset_max_train_per_file
        )
    chunk_range = 5000000
    for fold_idx in zip(*test_idx):
        train_idx = [[] for _ in data_size]
        train_idx_size = 0
        for file_idx, idx in enumerate(fold_idx):
            ds = data_size[file_idx]
            k = 0
            while k + chunk_range < ds:
                train_idx[file_idx] += list(
                    set(range(k, k + chunk_range)) - set(idx)
                )
                k += chunk_range
            train_idx[file_idx] += list(set(range(k, ds)) - set(idx))
            train_idx_size += len(train_idx[file_idx])
        if len(subset_max_train_per_file) > 0 and train_idx_size > sum(
            subset_max_train_per_file
        ):
            LOGGER.info(
                "Subsetting PSMs (%i) to (%i).",
                train_idx_size,
                subset_max_train,
            )
            for i, current_subset_max_train in enumerate(
                subset_max_train_per_file
            ):
                if current_subset_max_train < train_idx_size:
                    train_idx[i] = rng.choice(
                        train_idx[i], current_subset_max_train, replace=False
                    )
        yield train_idx


def _create_psms(psms, data, enforce_checks=True):
    utils.convert_targets_column(data=data, target_column=psms.target_column)
    return LinearPsmDataset(
        psms=data,
        target_column=psms.target_column,
        spectrum_columns=psms.spectrum_columns,
        peptide_column=psms.peptide_column,
        protein_column=psms.protein_column,
        group_column=psms.group_column,
        feature_columns=psms.feature_columns,
        filename_column=psms.filename_column,
        scan_column=psms.scan_column,
        calcmass_column=psms.calcmass_column,
        expmass_column=psms.expmass_column,
        rt_column=psms.rt_column,
        charge_column=psms.charge_column,
        copy_data=False,
        enforce_checks=enforce_checks,
    )


def get_index_values(df, col_name, val, orig_idx):
    df = df[df[col_name] == val].drop(col_name, axis=1)
    orig_idx[val] += list(df.index)
    return df


def predict_fold(model, fold, psms, scores):
    scores[fold].append(model.predict(psms))


def _predict(models_idx, psms, models, test_fdr, max_workers):
    """
    Return the new scores for the dataset

    Parameters
    ----------
    psms : Dict
        Contains all required info about the dataset to rescore
    models_idx : list of numpy.ndarray
        The indicies of the models to predict with
    models : list of Model
        The models for each dataset and whether it
        was reset or not.
    test_fdr : the fdr to calibrate at.
    max_workers : maximum threads for parallelism

    Returns
    -------
    numpy.ndarray
        A :py:class:`numpy.ndarray` containing the new scores.
    """
    for _psms, mod_idx in zip(psms, models_idx):
        scores = []

        model_test_idx = utils.create_chunks(
            data=mod_idx, chunk_size=CHUNK_SIZE_ROWS_PREDICTION
        )
        n_folds = len(models)
        fold_scores = [[] for _ in range(n_folds)]
        targets = [[] for _ in range(n_folds)]
        orig_idx = [[] for _ in range(n_folds)]
        reader = read_file_in_chunks(
            file=_psms.filename,
            chunk_size=CHUNK_SIZE_ROWS_PREDICTION,
            use_cols=_psms.columns,
        )
        for i, psms_slice in enumerate(reader):
            psms_slice["fold"] = model_test_idx.pop(0)
            psms_slice = [
                get_index_values(psms_slice, "fold", i, orig_idx)
                for i in range(n_folds)
            ]
            psms_slice = [
                _create_psms(_psms, psm_slice, enforce_checks=False)
                for psm_slice in psms_slice
            ]
            [
                targets[i].append(psm_slice.targets)
                for i, psm_slice in enumerate(psms_slice)
            ]

            Parallel(n_jobs=max_workers, require="sharedmem")(
                delayed(predict_fold)(
                    model=models[mod_idx],
                    fold=mod_idx,
                    psms=_psms,
                    scores=fold_scores,
                )
                for mod_idx, _psms in enumerate(psms_slice)
            )
            del psms_slice
        del reader
        del model_test_idx
        for mod in models:
            try:
                mod.estimator.decision_function
                scores.append(
                    calibrate_scores(
                        np.hstack(fold_scores.pop(0)),
                        np.hstack(targets.pop(0)),
                        test_fdr,
                    )
                )
            except AttributeError:
                scores.append(np.hstack(fold_scores.pop(0)))
            except RuntimeError:
                raise RuntimeError(
                    "Failed to calibrate scores between cross-validation folds, "
                    "because no target PSMs could be found below 'test_fdr'. Try "
                    "raising 'test_fdr'."
                )
        del targets
        del fold_scores
        orig_idx = np.argsort(sum(orig_idx, [])).tolist()
        yield np.concatenate(scores)[orig_idx]


def _predict_with_ensemble(psms, models, max_workers):
    """
    Return the new scores for the dataset using ensemble of all trained models

    Parameters
    ----------
    max_workers
    psms : Dict
        Contains all required info about the dataset to rescore
    models : list of Model
        The models for each dataset and whether it
        was reset or not.
    """
    scores = [[] for _ in range(len(models))]
    reader = read_file_in_chunks(
        file=psms.filename,
        chunk_size=CHUNK_SIZE_ROWS_PREDICTION,
        use_cols=psms.columns,
    )
    for data in reader:
        data = _create_psms(psms, data, enforce_checks=False)
        fold_scores = Parallel(n_jobs=max_workers, require="sharedmem")(
            delayed(mod.predict)(psms=data) for mod in models
        )
        [score.append(fs) for score, fs in zip(scores, fold_scores)]
    del fold_scores
    scores = [np.hstack(score) for score in scores]
    return np.mean(scores, axis=0)


def _fit_model(train_set, psms, model, fold):
    """
    Fit the estimator using the training data.

    Parameters
    ----------
    train_set : PsmDataset
        A PsmDataset that specifies the training data
    model : tuple of Model
        A Classifier to train.
    fold : int
        The fold number. Only used for logging.

    Returns
    -------
    model : mokapot.model.Model
        The trained model.
    reset : bool
        Whether the models should be reset to their original parameters.
    """
    model.fold = fold + 1
    LOGGER.debug("")
    LOGGER.debug("=== Analyzing Fold %i ===", fold + 1)
    reset = False
    train_set = _create_psms(psms[0], train_set)
    try:
        model.fit(train_set)
    except RuntimeError as msg:
        if str(msg) != "Model performs worse after training.":
            raise

        if model.is_trained:
            reset = True

    return model, reset
