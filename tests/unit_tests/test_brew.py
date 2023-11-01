"""Tests that the brew function works."""
import copy

import numpy as np
import pytest
from polars.testing import assert_frame_equal, assert_frame_not_equal
from sklearn.ensemble import RandomForestClassifier

import mokapot
from mokapot import Model, PercolatorModel, PsmConfidence

np.random.seed(42)


@pytest.fixture
def svm():
    """A simple Percolator model."""
    return svm_model()


def svm_model():
    """The percolator model."""
    return PercolatorModel(train_fdr=0.05, max_iter=10)


def rfm_model():
    """A random forest model."""
    return Model(RandomForestClassifier(), train_fdr=0.1)


@pytest.mark.parametrize(
    ("num", "model", "folds"),
    [
        (1, svm_model(), 3),
        (3, svm_model(), 3),
        (1, rfm_model(), 3),
        (1, svm_model(), 4),
    ],
)
def test_basic_params(psms, num, model, folds):
    """Test with mostly default parameters of brew."""
    psms = [copy.deepcopy(psms) for _ in range(num)]
    results, models = mokapot.brew(psms, model=model, folds=folds, rng=42)

    if num == 1:
        assert isinstance(results, PsmConfidence)
    else:
        assert isinstance(results, list)
        assert len(results) == num, f"Expected {num} confidence objects."

    assert len(models) == folds
    assert type(models[0]) is type(model)


def test_seed(psms, svm):
    """Test that (not) changing the split selection seed works."""
    folds = 3
    results_a, _ = mokapot.brew(psms, svm, folds=folds, rng=42)
    results_b, _ = mokapot.brew(psms, svm, folds=folds, rng=42)

    assert_frame_equal(results_a.results.psms, results_b.results.psms)

    psms.rng = 1
    results_c, _ = mokapot.brew(psms, svm, folds=folds, rng=1)
    assert_frame_not_equal(results_a.results.psms, results_c.results.psms)


def test_fdr_error(psms, svm):
    """Test that we get a sensible error message."""
    psms.rng = 1
    with pytest.raises(RuntimeError) as err:
        # This threshold is was determined by trial and error.
        psms.eval_fdr = 0.015
        mokapot.brew(psms, svm, rng=1)

    assert "Failed to calibrate" in str(err)


# @pytest.mark.skip(reason="Not currently working, at least on MacOS.")
def test_multiprocess(psms, svm):
    """Test that multiprocessing doesn't yield an error."""
    _, models = mokapot.brew(psms, svm, max_workers=2)

    # The models should not be the same:
    assert_not_close(models[0].estimator.coef_, models[1].estimator.coef_)
    assert_not_close(models[1].estimator.coef_, models[2].estimator.coef_)
    assert_not_close(models[2].estimator.coef_, models[0].estimator.coef_)


def test_trained_models(psms, svm):
    """Test that using trained models reproduces same results."""
    # fix a seed to have the same random split for each run
    results_with_training, models_with_training = mokapot.brew(
        psms, svm, rng=3
    )
    models = list(models_with_training)
    models.reverse()  # Change the model order
    results_without_training, models_without_training = mokapot.brew(
        psms, models, rng=3
    )
    assert models_with_training == models_without_training
    assert_frame_equal(
        results_with_training.results.psms,
        results_without_training.results.psms,
    )


def test_using_few_models_error(psms, svm):
    """The number of trained models less than the number of folds."""
    svm.is_trained = True
    with pytest.raises(ValueError) as err:
        mokapot.brew(psms, [svm, svm])
    assert (
        "The number of trained models (2) must match the number of folds (3)"
        in str(err)
    )


def test_using_untrained_models_error(psms, svm):
    """Test that using untrained models gives the expected error message."""
    svm.is_trained = False
    with pytest.raises(ValueError) as err:
        mokapot.brew(psms, [svm, svm, svm])

    assert "Only one untrained model is allowed" in str(err)


def assert_not_close(x, y):
    """Assert that two arrays are not equal."""
    np.testing.assert_raises(AssertionError, np.testing.assert_allclose, x, y)
