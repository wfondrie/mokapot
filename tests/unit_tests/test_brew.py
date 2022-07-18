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
    results, models = mokapot.brew(psms, svm, test_fdr=0.05, folds=4)
    assert isinstance(results, mokapot.confidence.LinearConfidence)
    assert len(models) == 4


def test_brew_test_fdr_error(psms, svm):
    """Test that we get a sensible error message"""
    with pytest.raises(RuntimeError) as err:
        results, models = mokapot.brew(psms, svm)

    assert "Failed to calibrate" in str(err)


# @pytest.mark.skip(reason="Not currently working, at least on MacOS.")
def test_brew_multiprocess(psms, svm):
    """Test that multiprocessing doesn't yield an error"""
    mokapot.brew(psms, svm, test_fdr=0.05, max_workers=2)


def test_brew_trained_models(psms, svm):
    """Test that using trained models reproduces same results"""
    # fix a seed to have the same random split for each run
    np.random.seed(3)
    results_with_training, models_with_training = mokapot.brew(
        psms, svm, test_fdr=0.05
    )
    np.random.seed(3)
    models = list(models_with_training)
    models.reverse()  # Change the model order
    results_without_training, models_without_training = mokapot.brew(
        psms, models, test_fdr=0.05
    )
    assert models_with_training == models_without_training
    assert results_with_training.accepted == results_without_training.accepted


def test_brew_using_few_models_error(psms, svm):
    """Test that if the number of trained models less than the number of folds we get the expected error message"""
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
