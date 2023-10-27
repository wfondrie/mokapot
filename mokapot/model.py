"""mokapot implements an algorithm for training machine learning models to
distinguish high-scoring target peptide-spectrum matches (PSMs) from decoy PSMs
using an iterative procedure. It is the :py:class:`Model` class that contains
this logic. A :py:class:`Model` instance can be created from any object with a
`scikit-learn estimator interface
<https://scikit-learn.org/stable/developers/develop.html>`_, allowing a wide
variety of models to be used. Once initialized, the :py:meth:`Model.fit` method
trains the underyling classifier using :doc:`a collection of PSMs <dataset>`
with this iterative approach.

Additional subclasses of the :py:class:`Model` class are available for
typical use cases. For example, use :py:class:`PercolatorModel` if you
want to emulate the behavior of Percolator.

"""
from __future__ import annotations

import logging
import pickle
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.base import ClassifierMixin, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .dataset import PsmDataset

LOGGER = logging.getLogger(__name__)
PERC_GRID = {
    "class_weight": [
        {0: neg, 1: pos} for neg in (0.1, 1, 10) for pos in (0.1, 1, 10)
    ]
}


class Model:
    """A machine learning model to re-score PSMs.

    Any classifier with a `scikit-learn estimator interface
    <https://scikit-learn.org/stable/developers/develop.html#estimators>`_
    can be used. This class also supports hyper parameter optimization
    using classes from the :py:mod:`sklearn.model_selection`
    module, such as the :py:class:`~sklearn.model_selection.GridSearchCV`
    and :py:class:`~sklearn.model_selection.RandomizedSearchCV` classes.

    Parameters
    ----------
    estimator : classifier object
        A classifier that is assumed to implement the scikit-learn
        estimator interface. To emulate Percolator (an SVM model) use
        :py:class:`PercolatorModel` instead.
    scaler : scaler object or "as-is", optional
        Defines how features are normalized before model fitting and
        prediction. The default, :code:`None`, subtracts the mean and scales
        to unit variance using
        :py:class:`sklearn.preprocessing.StandardScaler`.
        Other scalers should follow the `scikit-learn transformer
        interface
        <https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects>`_
        , implementing :code:`fit_transform()` and :code:`transform()` methods.
        Alternatively, the string :code:`"as-is"` leaves the features in
        their original scale.
    train_fdr : float, optional
        The maximum false discovery rate at which to consider a target PSM as a
        positive example.
    max_iter : int, optional
        The number of iterations to perform.
    direction : str or None, optional
        The name of the feature to use as the initial direction for ranking
        PSMs. The default, :code:`None`, automatically
        selects the feature that finds the most PSMs below the
        `train_fdr`. This
        will be ignored in the case the model is already trained.
    override : bool, optional
        If the learned model performs worse than the best feature, should
        the model still be used?
    shuffle : bool, optional
        Should the order of PSMs be randomized for training? For deterministic
        algorithms, this will have no effect.
    rng : int or numpy.random.Generator, optional
        The seed or generator used for model training.

    Attributes
    ----------
    estimator : classifier object
        The classifier used to re-score PSMs.
    scaler : scaler object
        The scaler used to normalize features.
    features : list of str or None
        The name of the features used to fit the model. None if the
        model has yet to be trained.
    is_trained : bool
        Indicates if the model has been trained.
    train_fdr : float
        The maximum false discovery rate at which to consider a target PSM as a
        positive example.
    max_iter : int
        The number of iterations to perform.
    direction : str or None
        The name of the feature to use as the initial direction for ranking
        PSMs.
    override : bool
        If the learned model performs worse than the best feature, should
        the model still be used?
    shuffle : bool
        Is the order of PSMs shuffled for training?
    fold : int or None
        The CV fold on which this model was fit, if any.
    rng : numpy.random.Generator
        The random number generator.
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        scaler: TransformerMixin | Literal["as-is"] | None = None,
        train_fdr: float = 0.01,
        max_iter: int = 10,
        direction: str = None,
        override: bool = False,
        shuffle: bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        """Initialize a Model object."""
        self.estimator = clone(estimator)
        self.features = None
        self.is_trained = False

        if scaler == "as-is":
            self.scaler = DummyScaler()
        elif scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = clone(scaler)

        self.train_fdr = train_fdr
        self.max_iter = max_iter
        self.direction = direction
        self.override = override
        self.shuffle = shuffle
        self.rng = rng

        # To keep track of the fold that this was trained on.
        # Needed to ensure reproducibility in brew() with
        # multiprocessing.
        self.fold = None

        # Sort out whether we need to optimize hyperparameters:
        if isinstance(self.estimator, BaseSearchCV):
            self._needs_cv = True
        else:
            self._needs_cv = False

    def __repr__(self) -> str:
        """How to print the class."""
        trained = {True: "A trained", False: "An untrained"}
        return (
            f"{trained[self.is_trained]} mokapot.model.Model object:\n"
            f"  estimator: {self.estimator}\n"
            f"  scaler: {self.scaler}\n"
            f"  features: {self.features}"
        )

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator for model training."""
        return self._rng

    @rng.setter
    def rng(self, rng: np.random.Generator | int | None) -> None:
        """Set the random number generator."""
        self._rng = np.random.default_rng(rng)

    def save(self, out_file: PathLike) -> Path:
        """Save the model to a file.

        Parameters
        ----------
        out_file : PathLike
            The name of the file for the saved model.

        Returns
        -------
        Path
            The output file name.

        Notes
        -----
        Because classes may change between mokapot and scikit-learn
        versions, a saved model may not work when either is changed
        from the version that created the model.
        """
        out_file = Path(out_file)
        with out_file.open("wb+") as out:
            pickle.dump(self, out)

        return out_file

    def decision_function(self, dataset: PsmDataset) -> np.ndarray:
        """Score examples from a dataset.

        Parameters
        ----------
        dataset : PsmDataset object
            The dataset to score.

        Returns
        -------
        numpy.ndarray
            A :py:class:`numpy.ndarray` containing the score for each example.
        """
        if not self.is_trained:
            raise NotFittedError("This model is untrained. Run fit() first.")

        if set(dataset.schema.features) != set(self.features):
            raise ValueError(
                "Features of the datset do not match the "
                "features of this Model."
            )

        # Verifies that the columns are in the correct order:
        data = (
            dataset.data.select(self.features)
            .collect(streaming=True)
            .to_numpy()
        )

        feat = self.scaler.transform(data)
        return self._compute_scores(feat)

    def predict(self, dataset: PsmDataset) -> np.ndarray:
        """Alias for :py:meth:`decision_function`."""
        return self.decision_function(dataset)

    def fit(self, dataset: PsmDataset) -> Model:
        """Fit the model using the Percolator algorithm.

        The model if trained by iteratively learning to separate decoy
        PSMs from high-scoring target examples. By default, an initial
        direction is chosen as the feature that best separates target
        from decoy examples. A false discovery rate threshold is used to
        define how high a target must score to be used as a positive
        example in the next training iteration.

        Parameters
        ----------
        dataset : PsmDataset object
            The dataset from which to train the model.

        Returns
        -------
        self
        """
        if not dataset.targets.sum():
            raise ValueError(
                f"No target {dataset.unit} were available for training."
            )

        if dataset.targets.sum() == len(dataset):
            raise ValueError(
                f"No decoy {dataset.unit} were available for training."
            )

        if len(dataset) <= 200:
            LOGGER.warning(
                "Few %s are available for model training (%i). "
                "The learned models may be unstable.",
                dataset.unit,
                len(dataset.data),
            )

        # Scale features and choose the initial direction
        self.features = dataset.schema.features
        norm_feat = self.scaler.fit_transform(dataset.features)
        start_labels = self.starting_labels(dataset, norm_feat)

        # Shuffle order
        shuffled_idx = self.rng.permutation(np.arange(len(start_labels)))
        original_idx = np.argsort(shuffled_idx)
        if self.shuffle:
            norm_feat = norm_feat[shuffled_idx, :]
            start_labels = start_labels[shuffled_idx]

        # Prepare the model:
        model = _find_hyperparameters(self, norm_feat, start_labels)

        # Begin training loop
        target = start_labels
        num_passed = []
        LOGGER.info("Beginning training loop...")
        for i in range(self.max_iter):
            # Fit the model
            samples = norm_feat[target.astype(bool), :]
            iter_targ = (target[target.astype(bool)] + 1) / 2
            model.fit(samples, iter_targ)

            # Update scores
            scores = self._compute_scores(norm_feat)
            scores = scores[original_idx]

            # Update target
            target = dataset.update_labels(scores, desc=True)
            target = target[shuffled_idx]
            num_passed.append((target == 1).sum())

            LOGGER.info(
                "  - Iteration %i: %i training PSMs passed.", i, num_passed[i]
            )

        # If the model performs worse than what was initialized:
        if num_passed[-1] < (start_labels == 1).sum():
            if self.override:
                LOGGER.warning("Model performs worse after training.")
            else:
                raise RuntimeError("Model performs worse after training.")

        self.estimator = model
        weights = _get_weights(self.estimator, self.features)
        if weights is not None:
            LOGGER.info("Normalized feature weights in the learned model:")
            for line in weights:
                LOGGER.info("    %s", line)

        self.is_trained = True
        LOGGER.info("Done training.")
        return self

    def starting_labels(
        self,
        dataset: PsmDataset,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Find the labels to use for the initial direction.

        If the model hasn't been trained and no direction was provided, the
        feature with the most examples at the dataset's :code:`eval_fdr` will
        be used. If the model has been trained, then the output of the model is
        used for further model fitting. Finally, if a direction is provided,
        then that feature is used compute the starting labels for model
        training.

        Parameters
        ----------
        dataset : PsmDataset
            The examples to get labels from.
        features : np.ndarray
            The scaled features to be used as model input.

        Returns
        -------
        np.ndarray
            The starting labels for model training.

        """
        LOGGER.info("Finding initial direction...")
        if self.is_trained:
            prefix = "The pretrained model"
            scores = self._compute_scores(features)
            start_labels = dataset.update_labels(scores)
        elif self.direction is None:
            best_feat, desc = dataset.best_feature
            prefix = f"Selected feature '{best_feat}'"
            score = (
                dataset.data.select(best_feat)
                .collect(streaming=True)
                .to_numpy()
            )
            start_labels = dataset.update_labels(score, desc)
        else:
            prefix = f"Provided feature '{self.direction}'"
            score = (
                dataset.data.select(self.direction)
                .collect(streaming=True)
                .to_numpy()
            )
            start_labels = dataset.update_labels(score, desc=True)
            asc_labels = dataset.update_labels(score, desc=False)
            if (asc_labels == 1).sum() > (start_labels == 1).sum():
                start_labels = asc_labels

        LOGGER.info(
            "  - %s yielded %i %s at q<=%g.",
            prefix,
            (start_labels == 1).sum(),
            dataset.unit,
            self.train_fdr,
        )

        if not (start_labels == 1).sum():
            raise RuntimeError(
                f"No PSMs accepted at train_fdr={self.train_fdr}. "
                "Consider changing it to a higher value."
            )

        return start_labels

    def _compute_scores(self, features: np.ndarray) -> np.ndarry:
        """Compute the scores with the model.

        We want to use the `predict_proba` method if it is available, but fall
        back to the `decision_function` method if it isn't. In sklearn,
        `predict_proba` for a binary classifier returns a two-column numpy
        array, where the second column is the probability we want. However,
        skorch (and other tools) sometime do this differently, returning only a
        single column. This function makes it so mokapot can work with either.

        Parameters
        ----------
        features : np.ndarray
            The normalized features

        Returns
        -------
        np.ndarray
            A :py:class:`numpy.ndarray` containing the score for each row.

        """
        try:
            scores = self.estimator.predict_proba(features).squeeze()
            if len(scores.shape) == 2:
                return scores[:, 1]

            if len(scores.shape) == 1:
                return scores

            raise RuntimeError("'predict_proba' returned too many dimensions.")
        except AttributeError:
            return self.estimator.decision_function(features)

    @classmethod
    def load(cls, model_file: PathLike) -> Model:
        """
        Load a saved model for mokapot.

        Parameters
        ----------
        model_file : PathLike
            The name of file from which to load the model.

        Returns
        -------
        mokapot.model.Model
            The loaded mokapot model.

        Warnings
        --------
        Unpickling data in Python is unsafe. Make sure that the model is from
        a source that you trust.
        """
        LOGGER.info("Loading mokapot model...")
        with open(model_file, "rb") as mod_in:
            model = pickle.load(mod_in)

        if isinstance(model, Model):
            return model

        raise TypeError("This file did not contain a mokapot Model.")


class PercolatorModel(Model):
    """A model that emulates Percolator.

    Create linear support vector machine (SVM) model that is similar
    to the one used by Percolator. This is the default model used by
    mokapot.

    Parameters
    ----------
    scaler : scaler object or "as-is", optional
        Defines how features are normalized before model fitting and
        prediction. The default, :code:`None`, subtracts the mean and scales
        to unit variance using
        :py:class:`sklearn.preprocessing.StandardScaler`.
        Other scalers should follow the `scikit-learn transformer
        interface
        <https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects>`_
        , implementing :code:`fit_transform()` and :code:`transform()` methods.
        Alternatively, the string :code:`"as-is"` leaves the features in
        their original scale.
    train_fdr : float, optional
        The maximum false discovery rate at which to consider a target PSM as a
        positive example.
    max_iter : int, optional
        The number of iterations to perform.
    direction : str or None, optional
        The name of the feature to use as the initial direction for ranking
        PSMs. The default, :code:`None`, automatically
        selects the feature that finds the most PSMs below the
        `train_fdr`. This
        will be ignored in the case the model is already trained.
    override : bool, optional
        If the learned model performs worse than the best feature, should
        the model still be used?
    n_jobs : int, optional
        The number of jobs used to parallelize the hyperparameter grid search.
    rng : int or numpy.random.Generator, optional
        The seed or generator used for model training.

    Attributes
    ----------
    estimator : classifier object
        The classifier used to re-score PSMs.
    scaler : scaler object
        The scaler used to normalize features.
    features : list of str or None
        The name of the features used to fit the model. None if the
        model has yet to be trained.
    is_trained : bool
        Indicates if the model has been trained.
    train_fdr : float
        The maximum false discovery rate at which to consider a target PSM as a
        positive example.
    max_iter : int
        The number of iterations to perform.
    direction : str or None
        The name of the feature to use as the initial direction for ranking
        PSMs.
    override : bool
        If the learned model performs worse than the best feature, should
        the model still be used?
    n_jobs : int
        The number of jobs to use for parallizing the hyperparameter
        grid search.
    rng : numpy.random.Generator
        The random number generator.
    """

    def __init__(
        self,
        scaler: TransformerMixin | Literal["as-is"] | None = None,
        train_fdr: float = 0.01,
        max_iter: int = 10,
        direction: str = None,
        override: bool = False,
        n_jobs: int = -1,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        """Initialize a PercolatorModel."""
        self.n_jobs = n_jobs
        rng = np.random.default_rng(rng)
        svm_model = LinearSVC(dual=False, random_state=7)
        estimator = GridSearchCV(
            svm_model,
            param_grid=PERC_GRID,
            refit=False,
            cv=KFold(3, shuffle=True, random_state=rng.integers(1, 1e6)),
            n_jobs=n_jobs,
        )

        super().__init__(
            estimator=estimator,
            scaler=scaler,
            train_fdr=train_fdr,
            max_iter=max_iter,
            direction=direction,
            override=override,
            rng=rng,
        )


class DummyScaler:
    """
    Implements the interface of scikit-learn scalers, but does
    nothing to the data. This simplifies the training code.

    :meta private:
    """

    def fit(self, x: np.ndarray) -> DummyScaler:
        """Does nothing."""
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Does nothing."""
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Does nothing."""
        return x


def save_model(model: Model, out_file: PathLike):
    """
    Save a :py:class:`mokapot.model.Model` object to a file.

    Parameters
    ----------
    model : mokapot.Model
        The model to save.
    out_file : PathLike
        The path at which to the saved model.

    Returns
    -------
    Path
        The output file name.

    Notes
    -----
    Because classes may change between mokapot and scikit-learn versions,
    a saved model may not work when either is changed from the version
    that created the model.
    """
    return model.save(out_file)


def _find_hyperparameters(model, features, labels):
    """
    Find the hyperparameters for the model.

    Parameters
    ----------
    model : a mokapot.Model
        The model to fit.
    features : array-like
        The features to fit the model with.
    labels : array-like
        The labels for each PSM (1, 0, or -1).

    Returns
    -------
    An estimator.
    """
    if model._needs_cv:
        LOGGER.info("Selecting hyperparameters...")
        cv_samples = features[labels.astype(bool), :]
        cv_targ = (labels[labels.astype(bool)] + 1) / 2

        # Fit the model
        model.estimator.fit(cv_samples, cv_targ)

        # Extract the best params.
        best_params = model.estimator.best_params_
        new_est = model.estimator.estimator
        new_est.set_params(**best_params)
        model._needs_cv = False
        for param, value in best_params.items():
            LOGGER.info("\t- %s = %s", param, value)
    else:
        new_est = model.estimator

    return new_est


def _get_weights(model, features):
    """
    If the model is a linear model, parse the weights to a list of strings.

    Parameters
    ----------
    model : estimator
        An sklearn linear_model object
    features : list of str
        The feature names, in order.

    Returns
    -------
    list of str
        The weights associated with each feature.
    """
    try:
        weights = model.coef_
        intercept = model.intercept_
        assert weights.shape[0] == 1
        assert weights.shape[1] == len(features)
        assert len(intercept) == 1
        weights = list(weights.flatten())
    except (AttributeError, AssertionError):
        return None

    col_width = max([len(f) for f in features]) + 2
    txt_out = ["Feature" + " " * (col_width - 7) + "Weight"]
    for weight, feature in zip(weights, features):
        space = " " * (col_width - len(feature))
        txt_out.append(feature + space + str(weight))

    txt_out.append("intercept" + " " * (col_width - 9) + str(intercept[0]))
    return txt_out
