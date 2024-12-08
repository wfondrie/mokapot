"""Test that models work as expected"""

import pytest
import mokapot
import numpy as np
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
        shuffle=False,
    )

    assert isinstance(model.estimator, LogisticRegression)
    assert isinstance(model.scaler, MinMaxScaler)
    assert model.train_fdr == 0.05
    assert model.max_iter == 1
    assert model.direction == "score"
    assert model.override
    assert not model.shuffle
    assert not model.is_trained

    model = mokapot.Model(LogisticRegression(), scaler="as-is")
    assert isinstance(model.scaler, mokapot.model.DummyScaler)

    model = mokapot.Model(LogisticRegression())
    assert isinstance(model.scaler, StandardScaler)


def test_perc_init():
    """Test the initialization of a PercolatorModel"""
    model = mokapot.PercolatorModel(
        scaler="as-is",
        train_fdr=0.05,
        max_iter=1,
        direction="score",
        override=True,
    )
    assert isinstance(model.estimator, GridSearchCV)
    assert isinstance(model.estimator.estimator, LinearSVC)
    assert isinstance(model.scaler, mokapot.model.DummyScaler)
    assert model.train_fdr == 0.05
    assert model.max_iter == 1
    assert model.direction == "score"
    assert model.override


def test_model_fit(psms_dataset):
    """Test that model fitting works"""
    model = mokapot.Model(LogisticRegression(), train_fdr=0.05, max_iter=1)
    model.fit(psms_dataset)

    assert model.is_trained

    model = mokapot.Model(LogisticRegressionCV(), train_fdr=0.05, max_iter=1)
    model.fit(psms_dataset)

    assert isinstance(model.estimator, LogisticRegression)
    assert model.is_trained

    psms_dataset._data[psms_dataset._target_column] = False
    with pytest.raises(ValueError):
        model.fit(psms_dataset)  # no targets

    psms_dataset._data[psms_dataset._target_column] = True
    with pytest.raises(ValueError):
        model.fit(psms_dataset)  # no decoys


def test_model_fit_large_subset(psms_dataset):
    model = mokapot.Model(
        LogisticRegression(),
        train_fdr=0.05,
        max_iter=1,
    )
    model.fit(psms_dataset)

    assert model.is_trained


def test_model_predict(psms_dataset):
    """Test predictions"""
    model = mokapot.Model(LogisticRegression(), train_fdr=0.05, max_iter=1)

    try:
        model.predict(psms_dataset)
    except NotFittedError:
        pass

    # The normal case
    model.fit(psms_dataset)
    scores = model.predict(psms_dataset)
    assert len(scores) == len(psms_dataset)

    # The case where a model is trained on a dataset with different features:
    psms_dataset._data["blah"] = np.random.randn(len(psms_dataset))
    psms_dataset._feature_columns = ("score0", "blah")
    with pytest.raises(ValueError):
        model.predict(psms_dataset)


def test_model_persistance(tmp_path):
    """test that we can save and load a model"""
    model_file = tmp_path / "model.pkl"

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
