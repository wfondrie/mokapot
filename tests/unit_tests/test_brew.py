"""Tests that the brew function works"""
import pytest
import numpy as np
import mokapot
from mokapot import PercolatorModel, Model
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)


@pytest.fixture
def svm():
    """A simple Percolator model"""
    return PercolatorModel(train_fdr=0.05, max_iter=10)


def test_brew_simple(psms, svm):
    """Test with mostly default parameters of brew"""
    results, models = mokapot.brew(psms, svm, test_fdr=0.05)
    assert isinstance(results, mokapot.confidence.LinearConfidence)
    assert len(models) == 3
    assert isinstance(models[0], PercolatorModel)


def test_brew_random_forest(psms):
    """Verify there are no dependencies on the SVM."""
    rfm = Model(
        RandomForestClassifier(),
        train_fdr=0.1,
    )
    results, models = mokapot.brew(psms, model=rfm, test_fdr=0.1)
    assert isinstance(results, mokapot.confidence.LinearConfidence)
    assert len(models) == 3
    assert isinstance(models[0], Model)


def test_brew_joint(psms, svm):
    """Test that the multiple input PSM collections yield multiple out"""
    collections = [psms, psms, psms]
    results, models = mokapot.brew(collections, svm, test_fdr=0.05)
    assert len(results) == 3
    assert len(models) == 3


def test_brew_folds(psms, svm):
    """Test that changing the number of folds works"""
    results, models = mokapot.brew(psms, svm, test_fdr=0.1, folds=4)
    assert isinstance(results, mokapot.confidence.LinearConfidence)
    assert len(models) == 4


def test_brew_seed(psms, svm):
    """Test that (not) changing the split selection seed works"""
    folds = 3
    seed = 0

    results_a, models_a = mokapot.brew(
        psms, svm, test_fdr=0.05, folds=folds, rng=seed
    )
    assert isinstance(results_a, mokapot.confidence.LinearConfidence)
    assert len(models_a) == folds

    results_b, models_b = mokapot.brew(
        psms, svm, test_fdr=0.05, folds=folds, rng=seed
    )
    assert isinstance(results_b, mokapot.confidence.LinearConfidence)
    assert len(models_b) == folds

    assert (
        results_a.accepted == results_b.accepted
    ), "Results differed with same seed"

    results_c, models_c = mokapot.brew(
        psms, svm, test_fdr=0.05, folds=folds, rng=seed + 2
    )
    assert isinstance(results_c, mokapot.confidence.LinearConfidence)
    assert len(models_c) == folds

    assert (
        results_a.accepted != results_c.accepted
    ), "Results were identical with different seed!"


def test_brew_test_fdr_error(psms, svm):
    """Test that we get a sensible error message"""
    with pytest.raises(RuntimeError) as err:
        results, models = mokapot.brew(psms, svm)

    assert "Failed to calibrate" in str(err)


# @pytest.mark.skip(reason="Not currently working, at least on MacOS.")
def test_brew_multiprocess(psms, svm):
    """Test that multiprocessing doesn't yield an error"""
    _, models = mokapot.brew(psms, svm, test_fdr=0.05, max_workers=2)

    # The models should not be the same:
    assert_not_close(models[0].estimator.coef_, models[1].estimator.coef_)
    assert_not_close(models[1].estimator.coef_, models[2].estimator.coef_)
    assert_not_close(models[2].estimator.coef_, models[0].estimator.coef_)


def test_brew_trained_models(psms, svm):
    """Test that using trained models reproduces same results"""
    # fix a seed to have the same random split for each run
    results_with_training, models_with_training = mokapot.brew(
        psms, svm, test_fdr=0.05, rng=3
    )
    models = list(models_with_training)
    models.reverse()  # Change the model order
    results_without_training, models_without_training = mokapot.brew(
        psms, models, test_fdr=0.05, rng=3
    )
    assert models_with_training == models_without_training
    assert results_with_training.accepted == results_without_training.accepted


def test_brew_using_few_models_error(psms, svm):
    """Test that if the number of trained models less than the number of
    folds we get the expected error message.
    """
    with pytest.raises(ValueError) as err:
        mokapot.brew(psms, [svm, svm], test_fdr=0.05)
    assert (
        "The number of trained models (2) must match the number of folds (3)."
        in str(err)
    )


def test_brew_using_non_trained_models_error(psms, svm):
    """Test that using non trained models gives the expected error message"""
    svm.is_trained = False
    with pytest.raises(RuntimeError) as err:
        mokapot.brew(psms, [svm, svm, svm], test_fdr=0.05)
    assert (
        "One or more of the provided models was not previously trained"
        in str(err)
    )


def assert_not_close(x, y):
    """Assert that two arrays are not equal"""
    np.testing.assert_raises(AssertionError, np.testing.assert_allclose, x, y)
