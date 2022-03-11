"""Test that models work as expected"""
import pytest
import mokapot
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError


def test_model_init():
    """Test that a model initializes correctly"""
    model = mokapot.Model(
        LogisticRegression(),
        scaler=MinMaxScaler(),
        train_fdr=0.05,
        max_iter=1,
        direction="score",
        override=True,
        subset_max_train=500,
        shuffle=False,
    )

    assert isinstance(model.estimator, LogisticRegression)
    assert isinstance(model.scaler, MinMaxScaler)
    assert model.train_fdr == 0.05
    assert model.max_iter == 1
    assert model.direction == "score"
    assert model.override
    assert model.subset_max_train == 500
    assert not model.shuffle
    assert not model.is_trained

    model = mokapot.Model(LogisticRegression(), scaler="as-is")
    assert isinstance(model.scaler, mokapot.model.DummyScaler)

    model = mokapot.Model(LogisticRegression())
    assert isinstance(model.scaler, StandardScaler)

    print(model)


def test_perc_init():
    """Test the initialization of a PercolatorModel"""
    model = mokapot.PercolatorModel(
        scaler="as-is",
        train_fdr=0.05,
        max_iter=1,
        direction="score",
        override=True,
        subset_max_train=500,
    )
    assert isinstance(model.estimator, GridSearchCV)
    assert isinstance(model.estimator.estimator, LinearSVC)
    assert isinstance(model.scaler, mokapot.model.DummyScaler)
    assert model.train_fdr == 0.05
    assert model.max_iter == 1
    assert model.direction == "score"
    assert model.override
    assert model.subset_max_train == 500


def test_model_fit(psms):
    """Test that model fitting works"""
    model = mokapot.Model(LogisticRegression(), train_fdr=0.05, max_iter=1)
    model.fit(psms)

    assert model.is_trained

    model = mokapot.Model(LogisticRegressionCV(), train_fdr=0.05, max_iter=1)
    model.fit(psms)

    assert isinstance(model.estimator, LogisticRegression)
    assert model.is_trained

    no_targets = pd.DataFrame({"targets": [False] * 100})
    with pytest.raises(ValueError):
        model.fit(no_targets)

    no_decoys = pd.DataFrame({"targets": [True] * 100})
    with pytest.raises(ValueError):
        model.fit(no_decoys)


def test_model_fit_large_subset(psms):
    model = mokapot.Model(
        LogisticRegression(),
        train_fdr=0.05,
        max_iter=1,
        subset_max_train=2_000_000_000,
    )
    model.fit(psms)

    assert model.is_trained


def test_model_predict(psms):
    """Test predictions"""
    model = mokapot.Model(LogisticRegression(), train_fdr=0.05, max_iter=1)

    try:
        model.predict(psms)
    except NotFittedError:
        pass

    # The normal case
    model.fit(psms)
    scores = model.predict(psms)
    assert len(scores) == len(psms)

    # The case where a model is trained on a dataset with different features:
    psms._data["blah"] = np.random.randn(len(psms))
    psms._feature_columns = ("score", "blah")
    with pytest.raises(ValueError):
        model.predict(psms)


def test_model_persistance(tmp_path):
    """test that we can save and load a model"""
    model_file = str(tmp_path / "model.pkl")

    model = mokapot.Model(LogisticRegression(), train_fdr=0.05, max_iter=1)
    mokapot.save_model(model, model_file)
    loaded = mokapot.load_model(model_file)

    assert isinstance(loaded, mokapot.Model)


def test_dummy_scaler():
    """Test the DummyScaler class"""
    data = np.random.default_rng(42).normal(0, 1, (20, 10))
    scaler = mokapot.model.DummyScaler()
    assert (data == scaler.fit_transform(data)).all()
    assert (data == scaler.transform(data)).all()
