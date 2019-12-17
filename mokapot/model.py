"""
This module defines the model classes to used mokapot.
"""
from __future__ import annotations
import logging
from typing import Tuple, Dict, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import sklearn.base as base
import sklearn.svm as svm
import sklearn.model_selection as ms
from sklearn.exceptions import NotFittedError

from mokapot.qvalues import tdc
import mokapot.utils as utils

if TYPE_CHECKING:
    from mokapot.dataset import PsmDataset


LOGGER = logging.getLogger(__name__)
MODEL_TYPES = Union[base.BaseEstimator, ms.GridSearchCV, ms.RandomizedSearchCV]


class Classifier():
    """
    Create a Linear SVM model for classifying PSMs.
    """
    def __init__(self, estimator: MODEL_TYPES = \
                 svm.LinearSVC(dual=False, class_weight="balanced")) \
                 -> None:
        """Initialize a MokapotModel object"""
        if not base.is_classifier(estimator):
            raise ValueError("The 'estimator' must be a classifier. See "
                             "https://scikit-learn.org/stable/supervised_learn"
                             "ing.html#supervised-learning for options.")

        self.estimator = None
        self.feature_names = None

        # private-ish attributes
        self._base_estimator = base.clone(estimator)
        self._train_mean = None # mean of training set features
        self._train_std = None  # standard deviation of training set features
        self._trained = False   # Has the model been fit?
        self._standardize = True  # Standardize features for predictions?

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

        model = self.estimator
        train_mean = self._train_mean
        train_std = self._train_std

        feat_names = psms.features.columns.tolist()
        if set(feat_names) != set(self.feature_names):
            raise ValueError("Features of the PsmDataset do not match the "
                             "features of this Classifier.")

        feat = psms.features[self.feature_names].values
        if self._standardize:
            feat = utils.safe_divide((feat-train_mean.values), train_std)

        return self.estimator.decision_function(feat)

    def predict(self, psms: PsmDataset) -> np.ndarray:
        """Score the psms"""
        return self.decision_function(psms)

    def fit(self, psms: PsmDataset, train_fdr: float = 0.01,
            max_iter: int = 10) -> None:
        """Fit an SVM model using the Percolator procedure"""
        best_feat, feat_pass, feat_target = psms.find_best_feature(train_fdr)
        _best_feat_msg(best_feat, feat_pass)

        # Normalize Features
        self.feature_names = psms.features.columns.tolist()
        train_mean = psms.features.mean(axis=0)
        train_std = psms.features.std(axis=0)
        norm_feat = utils.safe_divide((psms.features.copy() - train_mean),
                                      train_std)

        # Initialize Model and Training Variables
        target = feat_target

        if isinstance(self._base_estimator, ms._search.BaseSearchCV):
            LOGGER.info("Choosing hyper-parameters by grid search...")
            self._base_estimator.fit(norm_feat, feat_target)

            best_params = self._base_estimator.best_params_
            model = self._base_estimator.estimator
            model.set_params(**best_params)

            pstring = ", ".join([f"{k} = {v}" for k, v in best_params.items()])
            LOGGER.info("Best parameters: %s", pstring)

        else:
            model = base.clone(self._base_estimator)

        # Begin training loop
        target = feat_target
        num_passed = []
        for _ in range(max_iter):
            # Fit the model
            samples = norm_feat[target.astype(bool), :]
            iter_targ = target[target.astype(bool)]
            model.fit(samples, iter_targ)

            # Update scores
            scores = model.decision_function(norm_feat)

            # Update target
            qvals = tdc(scores, target=(psms.label+1)/2)
            unlabeled = np.logical_and(qvals > train_fdr, psms.label == 1)
            target = psms.label.copy()
            target[unlabeled] = 0
            num_pass = (target == 1).sum()
            num_passed.append(num_pass)

        _fold_msg(best_feat, feat_pass, num_passed)

        if feat_pass > num_passed[-1]:
            raise LOGGER.warning("No improvement was detected with model "
                                 "training. Consider a less stringent value "
                                 "for 'train_fdr'")

        self.estimator = model
        self._train_mean = train_mean
        self._train_std = train_std
        self._trained = True


def _best_feat_msg(best_feat, num_pass):
    """Log the best feature and the number of positive PSMs."""
    LOGGER.info("Selected feature '%s' as initial direction.", best_feat)
    LOGGER.info("Could separate %i training set positives in that direction.",
                num_pass)


def _fold_msg(best_feat, best_feat_pass, num_pass):
    """Logging messages for each fold"""
    LOGGER.info("Positive PSMs by iteration:")
    num_passed = [f"{n}" for i, n in enumerate(num_pass)]
    num_passed = "->".join(num_passed)
    LOGGER.info("%s\n", num_passed)
