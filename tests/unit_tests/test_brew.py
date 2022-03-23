"""Tests that the brew function works"""
import pytest
import numpy as np
import mokapot
from mokapot import PercolatorModel

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
