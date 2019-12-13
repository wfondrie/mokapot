"""
This module defines the model classes to used mokapot.
"""
import logging
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.svm as svm
from sklearn.exceptions import NotFittedError
from sklearn.linear_model.base import LinearClassifierMixin

from .dataset import PsmDataset, _best_feat_msg, _fold_msg
from .qvalues import tdc
from .utils import unnormalize_weights

class Classifier():
    """
    Create a Linear SVM model for classifying PSMs.
    """
    def __init__(self, estimator: sklearn.base.BaseEstimator) -> None:
        """Initialize a MokapotModel object"""
        if not sklearn.base.is_classifier(estimator):
            raise ValueError("The 'estimator' must be a classifier. See "
                             "https://scikit-learn.org/stable/supervised_learn"
                             "ing.html#supervised-learning for options.")

        self.estimator = None
        self.feature_names = None

        # private-ish attributes
        self._base_estimator = sklearn.base.clone(estimator)
        self._train_mean = None # mean of training set features
        self._train_std = None  # standard deviation of training set features
        self._trained = False   # Has the model been fit?
        self._normalize = True  # Normalize features for predictions?

    def predict(self, psms: PsmDataset) -> None:
        """
        Score the PSMs in a PsmDataset

        Parameters
        ----------
        psms
            The collection of PSMs to score.
        """
        if not self._trained:
            raise NotFittedError("This model is untrained. Run fit() or "
                                 "load_weights() first.")

        # For users, 'psms' should be a PsmDataset. However, if the
        # MokapotModel was fit using the 'percolate()' method, self.model will
        # be a tuple. In this case, this method should only use the first model
        # NOTE: 'PsmDataset.percolate()' calls 'predict' on a tuple of
        # PsmDataset objects of equal length to self.estimator.

        # The standard user case:
        if isinstance(psms, PsmDataset):
            psms = (psms,)
            if not isinstance(self.estimator, tuple):
                model = (self.estimator,)
                train_mean = (self._train_mean,)
                train_std = (self._train_std,)
            else:
                model = self.estimator[0]
                train_mean = self._train_mean[0]
                train_std = self._train_std[0]

        # As used in 'percolate()'
        else:
            model = self.estimator
            train_mean = self._train_mean
            train_std = self._train_std

        # This should not be raised by users...
        if len(psms) != len(model):
            raise ValueError("psms does not match the number of models.")

        predictions = []
        for psm_set, model, tmean, tstd in zip(psms, model, train_mean, train_std):
            feat_names = psm_set.features.columns.tolist()
            if set(feat_names) != set(self.feature_names):
                raise ValueError("Features of the PsmDataset do not match the "
                                 "features used to fit this MokapotModel")

            feat = psm_set.features[self.feature_names].values
            if self._normalize:
                feat = np.divide((feat - tmean), tstd,
                                 out=np.zeros_like(feat),
                                 where=(tstd != 0))

            pred = model.decision_function(feat)
            predictions.append(pred)

        if len(predictions) == 1:
            predictions = predictions[0]

        return predictions

    def fit(self, psms: PsmDataset, train_fdr: float = 0.01,
            max_iter: int = 10, **kwargs) -> None:
        """Fit an SVM model using the Percolator procedure"""
        res = self._fit(psms=psms, train_fdr=train_fdr, max_iter=max_iter,
                        msg=True, kwargs=kwargs)

        model, feat_pass, num_pass, _, train_mean, train_std = res
        if feat_pass > num_pass[-1]:
            raise logging.warning("No improvement was detected with model "
                                  "training. Consider a less stringent value "
                                  "for 'train_fdr'")

        self.estimator = model
        self._train_mean = train_mean
        self._train_std = train_std
        self._trained = True

    def _fit(self, psms: PsmDataset, train_fdr: float = 0.01,
             max_iter: int = 10, msg: bool = True,
             kwargs: Dict = None) -> Tuple[pd.DataFrame, str]:
        """Fit the model using the Percolator procedure"""
        print(1)
        best_feat, feat_pass, feat_target = psms.find_best_feature(train_fdr)
        print(2)
        if msg:
            _best_feat_msg(best_feat, feat_pass)

        # Normalize Features
        self.feature_names = psms.features.columns.tolist()
        train_mean = psms.features.mean(axis=0)
        train_std = psms.features.std(axis=0)
        norm_feat = psms.features.copy()
        norm_feat = np.divide((norm_feat - train_mean), train_std,
                              out=np.zeros_like(norm_feat),
                              where=(train_std != 0))

        # Initialize Model and Training Variables
        target = feat_target
        model = sklearn.base.clone(self._base_estimator)

        # Grid search parameters, if required:
        if kwargs:
            if msg:
                logging.info("Choosing hyper-parameters by grid search...")

            grid_model = sklearn.model_selection.GridSearchCV(model, **kwargs)
            grid_model.fit(norm_feat, feat_target)
            model = grid_model.best_estimator_

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

        if msg:
            _fold_msg(best_feat, feat_pass, num_passed)

        return (model, feat_pass, num_passed, best_feat, train_mean, train_std)


class LinearClassifier(Classifier):
    """Create a linear classifier"""
    def __init__(self, estimator: sklearn.base.BaseEstimator) -> None:
        """Inititialize a linear classifier for mokapot"""
        super().__init__(estimator)

        # For linear models, there's no need to normalize the data again,
        # since the weights can be transformed to the original feature
        # scale. This also means you can load weights with no danger.
        self._normalize = False

    def load_weights(weights: pd.Series, intercept: float) -> None:
        """Load weights into a linear model"""
        self.feature_names = weights.index.tolist()
        self.estimator = self._base_estimator
        self.estimator.coef_ = weights
        self.estimator.intercept_ = intercept
        self._trained = True

    def fit(psms: PsmDataset, train_fdr: float = 0.01,
            max_iter: int = 10, **kwargs) -> None:
        """Fit the model"""
        res = super()._fit(psms, train_fdr, max_iter, kwargs)
        model, feat_pass, num_pass, _, train_mean, train_std = res
        if feat_pass > num_pass[-1]:
            raise logging.warning("No improvement was detected with model "
                                  "training. Consider a less stringent value "
                                  "for 'train_fdr'")

        weights = unnormalize_weights(model.coef_, model.intercept_,
                                      feat_mean, feat_std)

        model.coef_ = weights[0]
        model.intercept_ = weights[1]
        self.estimator = model
        self._trained = True
