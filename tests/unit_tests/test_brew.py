"""Tests that the brew function works"""
import copy

import pytest
import numpy as np
import mokapot
from mokapot import PercolatorModel, Model
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)


@pytest.fixture
def svm():
    """A simple Percolator model"""
    return PercolatorModel(train_fdr=0.05, max_iter=10, rng=2)


def test_brew_simple(psms_ondisk, svm):
    """Test with mostly default parameters of brew"""
    psms, models, scores, desc = mokapot.brew(psms_ondisk, svm, test_fdr=0.05)
    assert len(models) == 3
    assert isinstance(models[0], PercolatorModel)


def test_brew_random_forest(psms_ondisk):
    """Verify there are no dependencies on the SVM."""
    rfm = Model(
        RandomForestClassifier(),
        train_fdr=0.1,
    )
    psms, models, scores, desc = mokapot.brew(
        psms_ondisk, model=rfm, test_fdr=0.1
    )
    assert len(models) == 3
    assert isinstance(models[0], Model)


def test_brew_joint(psms_ondisk, svm):
    """Test that the multiple input PSM collections yield multiple out"""
    collections = [psms_ondisk, copy.copy(psms_ondisk), copy.copy(psms_ondisk)]
    psms, models, scores, desc = mokapot.brew(collections, svm, test_fdr=0.05)
    assert len(scores) == 3
    assert len(psms) == 3
    assert len(models) == 3
    assert len(desc) == 3


def test_brew_folds(psms_ondisk, svm):
    """Test that changing the number of folds works"""
    psms, models, scores, desc = mokapot.brew(
        psms_ondisk, svm, test_fdr=0.05, folds=4
    )
    assert len(scores) == 1
    assert len(psms) == 1
    assert len(models) == 4


def test_brew_seed(psms_ondisk, svm):
    """Test that (not) changing the split selection seed works"""
    folds = 3
    seed = 0
    psms_ondisk_b = copy.copy(psms_ondisk)
    psms_ondisk_c = copy.copy(psms_ondisk)
    psms_a, models_a, scores_a, desc_a = mokapot.brew(
        psms_ondisk, svm, test_fdr=0.05, folds=folds, rng=seed
    )
    assert len(models_a) == folds

    psms_b, models_b, scores_b, desc_b = mokapot.brew(
        psms_ondisk_b, svm, test_fdr=0.05, folds=folds, rng=seed
    )
    assert len(models_b) == folds

    assert np.array_equal(
        scores_a[0], scores_b[0]
    ), "Results differed with same seed"

    psms_c, models_c, scores_c, desc_c = mokapot.brew(
        psms_ondisk_c, svm, test_fdr=0.05, folds=folds, rng=seed + 2
    )
    assert len(models_c) == folds
    assert ~(
        np.array_equal(scores_a[0], scores_c[0])
    ), "Results were identical with different seed!"


def test_brew_test_fdr_error(psms_ondisk, svm):
    """Test that we get a sensible error message"""
    with pytest.raises(RuntimeError) as err:
        mokapot.brew(psms_ondisk, svm, test_fdr=0.001, rng=2)
    assert "Failed to calibrate" in str(err)


# @pytest.mark.skip(reason="Not currently working, at least on MacOS.")
def test_brew_multiprocess(psms_ondisk, svm):
    """Test that multiprocessing doesn't yield an error"""
    mokapot.brew(psms_ondisk, svm, test_fdr=0.05, max_workers=2)


def test_brew_trained_models(psms_ondisk, svm):
    """Test that using trained models reproduces same results"""
    # fix a seed to have the same random split for each run
    (
        psms_with_training,
        models_with_training,
        scores_with_training,
        desc_with_training,
    ) = mokapot.brew(copy.copy(psms_ondisk), svm, test_fdr=0.05, rng=2)
    models = list(models_with_training)
    models.reverse()  # Change the model order
    (
        psms_without_training,
        models_without_training,
        scores_without_training,
        desc_without_training,
    ) = mokapot.brew(psms_ondisk, models, test_fdr=0.05, rng=2)
    assert models_with_training == models_without_training
    assert np.array_equal(scores_with_training[0], scores_without_training[0])


def test_brew_using_few_models_error(psms_ondisk, svm):
    """Test that if the number of trained models less than the number of
    folds we get the expected error message.
    """
    with pytest.raises(ValueError) as err:
        mokapot.brew(psms_ondisk, [svm, svm], test_fdr=0.05)
    assert (
        "The number of trained models (2) must match the number of folds (3)."
        in str(err)
    )


def test_brew_using_non_trained_models_error(psms_ondisk, svm):
    """Test that using non trained models gives the expected error message"""
    svm.is_trained = False
    with pytest.raises(RuntimeError) as err:
        mokapot.brew(psms_ondisk, [svm, svm, svm], test_fdr=0.05)
    assert (
        "One or more of the provided models was not previously trained"
        in str(err)
    )
