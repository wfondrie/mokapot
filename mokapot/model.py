"""
This module defines the model classes to used mokapot.
"""
import logging
from typing import Union

import numpy as np
import pandas as pd
import sklearn.base as base
import sklearn.svm as svm
import sklearn.model_selection as ms
from sklearn.exceptions import NotFittedError

from . import utils
from .dataset import PsmDataset

LOGGER = logging.getLogger(__name__)

# Constants -------------------------------------------------------------------
MODEL_TYPES = Union[base.BaseEstimator, ms.GridSearchCV, ms.RandomizedSearchCV]
PERC_GRID = {"class_weight": [{0: neg, 1: pos} for neg in (0.1, 1, 10) for pos in (0.1, 1, 10)]}

# Classes ---------------------------------------------------------------------
class Model():
    """
    Create a Linear SVM model for classifying PSMs.
    """
    def __init__(self, estimator: MODEL_TYPES = None,
                 standardize_features: bool = True) -> None:
        """Initialize a Model object"""
        if estimator is None:
            svm_model = svm.LinearSVC(dual=False)
            estimator = ms.GridSearchCV(svm_model, param_grid=PERC_GRID,
                                        refit=False)

        self.estimator = estimator
        self.feature_names = None

        # private attributes
        self._base_estimator = base.clone(estimator)
        self._train_mean = None   # mean of training set features
        self._train_std = None    # standard deviation of training set features
        self._trained = False     # Has the model been fit?
        self._standardize_features = standardize_features

    def decision_function(self, psms: PsmDataset) -> np.ndarray:
        """
        Score the PSMs in a PsmDataset

        Parameters
        ----------
        psms
            The collection of PSMs to score.
        """
        if not self._trained:
            raise NotFittedError("This model is untrained. Run fit() first.")

        feat_names = psms.feature_columns
        if set(feat_names) != set(self.feature_names):
            raise ValueError("Features of the PsmDataset do not match the "
                             "features of this Model.")

        feat = psms.features.loc[:, self.feature_names].values
        if self._standardize_features:
            feat = utils.safe_divide((feat - self._train_mean.values),
                                     self._train_std.values)

        return self.estimator.decision_function(feat)

    def predict(self, psms: PsmDataset) -> np.ndarray:
        """Score the psms"""
        return self.decision_function(psms)

    def fit(self, psms: PsmDataset, train_fdr: float = 0.01,
            max_iter: int = 10) -> None:
        """Fit an SVM model using the Percolator procedure"""
        best_feat, feat_pass, feat_labels = psms._find_best_feature(train_fdr)

        # Normalize Features
        self.feature_names = psms.feature_columns
        self._train_mean = psms.features.mean(axis=0)
        self._train_std = psms.features.std(axis=0)
        norm_feat = utils.safe_divide((psms.features - self._train_mean),
                                      self._train_std)

        # Initialize Model and Training Variables
        if hasattr(self._base_estimator, "estimator"):
            cv_samples = norm_feat[feat_labels.astype(bool), :]
            cv_targ = (feat_labels[feat_labels.astype(bool)]+1)/2
            self._base_estimator.fit(cv_samples, cv_targ)
            best_params = self._base_estimator.best_params_
            model = self._base_estimator.estimator
            model.set_params(**best_params)
            print(best_params)
        else:
            model = base.clone(self._base_estimator)

        # Begin training loop
        target = feat_labels
        num_passed = []
        for _ in range(max_iter):
            # Fit the model
            samples = norm_feat[target.astype(bool), :]
            iter_targ = (target[target.astype(bool)]+1)/2
            model.fit(samples, iter_targ)

            # Update scores
            scores = model.decision_function(norm_feat)

            # Update target
            target = psms._update_labels(scores, fdr_threshold=train_fdr)
            num_passed.append((target == 1).sum())
            print(iter_targ.sum())

        self.estimator = model
        self._trained = True
