"""Tests that the brew function works"""

import copy

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

import mokapot
from mokapot import PercolatorModel, Model

np.random.seed(42)


@pytest.fixture
def svm():
    """A simple Percolator model"""
    return PercolatorModel(train_fdr=0.05, max_iter=10, rng=2)


def test_brew_simple(psms_ondisk, svm):
    """Test with mostly default parameters of brew"""
    models, scores = mokapot.brew([psms_ondisk], svm, test_fdr=0.05)
    assert len(models) == 3
    assert isinstance(models[0], PercolatorModel)


def test_brew_simple_parquet(psms_ondisk_from_parquet, svm):
    """Test with mostly default parameters of brew"""
    models, scores = mokapot.brew(
        [psms_ondisk_from_parquet], svm, test_fdr=0.05
    )
    assert len(models) == 3
    assert isinstance(models[0], PercolatorModel)


def test_brew_decision_tree(psms_ondisk):
    """Verify there are no dependencies on the SVM."""
    rfm = Model(
        # Changed from RF bc it is faster to run. 2024-12-06
        # RandomForestClassifier(),
        DecisionTreeClassifier(min_samples_leaf=10, min_samples_split=20),
        train_fdr=0.1,
    )
    models, scores = mokapot.brew([psms_ondisk], model=rfm, test_fdr=0.1)
    assert len(models) == 3
    assert isinstance(models[0], Model)


@pytest.mark.slow
def test_brew_joint(psms_ondisk, svm):
    """Test that the multiple input PSM collections yield multiple out"""
    collections = [psms_ondisk, copy.copy(psms_ondisk), copy.copy(psms_ondisk)]
    models, scores = mokapot.brew(collections, svm, test_fdr=0.05)
    assert len(scores) == 3
    assert len(models) == 3


def test_brew_joint_parquet(psms_ondisk_from_parquet, svm):
    """Test that the multiple input PSM collections yield multiple out"""
    collections = [
        psms_ondisk_from_parquet,
        copy.copy(psms_ondisk_from_parquet),
        copy.copy(psms_ondisk_from_parquet),
    ]
    models, scores = mokapot.brew(collections, svm, test_fdr=0.05)
    assert len(scores) == 3
    assert len(models) == 3


def test_brew_folds(psms_ondisk, svm):
    """Test that changing the number of folds works"""
    models, scores = mokapot.brew([psms_ondisk], svm, test_fdr=0.05, folds=4)
    assert len(scores) == 1
    assert len(models) == 4


@pytest.mark.slow
def test_brew_seed(psms_ondisk, svm):
    """Test that (not) changing the split selection seed works"""
    folds = 3
    seed = 0
    psms_ondisk_b = copy.copy(psms_ondisk)
    psms_ondisk_c = copy.copy(psms_ondisk)
    models_a, scores_a = mokapot.brew(
        [psms_ondisk], svm, test_fdr=0.05, folds=folds, rng=seed
    )
    assert len(models_a) == folds

    models_b, scores_b = mokapot.brew(
        [psms_ondisk_b], svm, test_fdr=0.05, folds=folds, rng=seed
    )
    assert len(models_b) == folds

    assert np.array_equal(scores_a[0], scores_b[0]), (
        "Results differed with same seed"
    )

    models_c, scores_c = mokapot.brew(
        [psms_ondisk_c], svm, test_fdr=0.05, folds=folds, rng=seed + 2
    )
    assert len(models_c) == folds
    assert not (np.array_equal(scores_a[0], scores_c[0])), (
        "Results were identical with different seed!"
    )


def test_brew_seed_parquet(psms_ondisk_from_parquet, svm):
    """Test that (not) changing the split selection seed works"""
    folds = 3
    seed = 0
    psms_ondisk_b = copy.copy(psms_ondisk_from_parquet)
    psms_ondisk_c = copy.copy(psms_ondisk_from_parquet)
    models_a, scores_a = mokapot.brew(
        [psms_ondisk_from_parquet],
        svm,
        test_fdr=0.05,
        folds=folds,
        rng=seed,
    )
    assert len(models_a) == folds

    models_b, scores_b = mokapot.brew(
        [psms_ondisk_b],
        svm,
        test_fdr=0.05,
        folds=folds,
        rng=seed,
    )
    assert len(models_b) == folds

    assert np.array_equal(scores_a[0], scores_b[0]), (
        "Results differed with same seed"
    )

    models_c, scores_c = mokapot.brew(
        [psms_ondisk_c],
        svm,
        test_fdr=0.05,
        folds=folds,
        rng=seed + 2,
    )
    assert len(models_c) == folds
    assert not (np.array_equal(scores_a[0], scores_c[0])), (
        "Results were identical with different seed!"
    )


def test_brew_test_fdr_error(psms_ondisk, svm):
    """Test that we get a sensible error message"""
    with pytest.raises(RuntimeError) as err:
        mokapot.brew([psms_ondisk], svm, test_fdr=0.001, rng=2)
    assert "Failed to calibrate" in str(err)


def test_brew_test_fdr_error_parquet(psms_ondisk_from_parquet, svm):
    """Test that we get a sensible error message"""
    with pytest.raises(RuntimeError) as err:
        mokapot.brew(
            [psms_ondisk_from_parquet],
            svm,
            test_fdr=0.001,
            rng=2,
        )
    assert "Failed to calibrate" in str(err)


def test_brew_multiprocess(psms_ondisk, svm):
    """Test that multiprocessing doesn't yield an error"""
    models, _ = mokapot.brew([psms_ondisk], svm, test_fdr=0.05, max_workers=2)
    # The models should not be the same:
    assert_not_close(models[0].estimator.coef_, models[1].estimator.coef_)
    assert_not_close(models[1].estimator.coef_, models[2].estimator.coef_)
    assert_not_close(models[2].estimator.coef_, models[0].estimator.coef_)


def test_brew_multiprocess_parquet(psms_ondisk_from_parquet, svm):
    """Test that multiprocessing doesn't yield an error"""
    models, _ = mokapot.brew(
        [psms_ondisk_from_parquet],
        svm,
        test_fdr=0.05,
        max_workers=2,
    )
    # The models should not be the same:
    assert_not_close(models[0].estimator.coef_, models[1].estimator.coef_)
    assert_not_close(models[1].estimator.coef_, models[2].estimator.coef_)
    assert_not_close(models[2].estimator.coef_, models[0].estimator.coef_)


def test_brew_trained_models(psms_ondisk, svm):
    """Test that using trained models reproduces same results"""
    # fix a seed to have the same random split for each run
    (
        models_with_training,
        scores_with_training,
    ) = mokapot.brew([copy.copy(psms_ondisk)], svm, test_fdr=0.05, rng=2)
    models = list(models_with_training)
    models.reverse()  # Change the model order
    (
        models_without_training,
        scores_without_training,
    ) = mokapot.brew([psms_ondisk], models, test_fdr=0.05, rng=2)
    assert models_with_training == models_without_training
    assert np.array_equal(scores_with_training[0], scores_without_training[0])


def test_brew_using_few_models_error(psms_ondisk, svm):
    """Test that if the number of trained models less than the number of
    folds we get the expected error message.
    """

    with pytest.raises(ValueError) as err:
        mokapot.brew([psms_ondisk], [svm, svm], test_fdr=0.05)
    assert (
        "The number of trained models (2) must match the number of folds (3)."
        in str(err)
    )


def test_brew_using_untrained_models_error(psms_ondisk, svm):
    """Test that all models all trained."""

    with pytest.raises(RuntimeError) as err:
        mokapot.brew([psms_ondisk], [svm, svm, svm], test_fdr=0.05)
    assert "not previously trained" in str(err)


def test_brew_using_non_trained_models_error(psms_ondisk, svm):
    """Test that using non trained models gives the expected error message"""
    svm.is_trained = False
    with pytest.raises(RuntimeError) as err:
        mokapot.brew([psms_ondisk], [svm, svm, svm], test_fdr=0.05)
    assert (
        "One or more of the provided models was not previously trained"
        in str(err)
    )


def assert_not_close(x, y):
    """Assert that two arrays are not equal"""
    np.testing.assert_raises(AssertionError, np.testing.assert_allclose, x, y)
