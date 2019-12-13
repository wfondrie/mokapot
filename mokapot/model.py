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

from .dataset import PsmDataset
from .qvalues import tdc

class MokapotModel():
    """
    Create a Linear SVM model for classifying PSMs.
    """
    def __init__(self, estimator: sklearn.base.BaseEstimator = \
                 svm.LinearSVC(dual=False, class_weight="balanced")) -> None:
        """Initialize a MokapotModel object"""
        self.estimator = None
        self.feature_names = None

        # private-ish attributes
        self._base_estimator = estimator # The untrained model
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
            else:
                model = self.estimator[0]

        # As used in 'percolate()'
        else:
            model = self.estimator

        # This should not be raised by users...
        if len(psms) != len(model):
            raise ValueError("psms does not match the number of models.")

        predictions = []
        for psm_set, model in zip(psms, model):
            feat_names = psm_set.features.columns.tolist()
            if set(feat_names) != set(self.feature_names):
                raise ValueError("Features of the PsmDataset do not match the "
                                 "features used to fit this MokapotModel")

            feat = psm_set.features[self.feature_names].values
            if self._normalize:
                feat = (feat - self._train_mean) / self._train_std

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

        model, feat_pass, num_pass, _ = res
        if feat_pass > num_pass[-1]:
            raise logging.warning("No improvement was detected with model "
                                  "training. Consider a less stringent value "
                                  "for 'train_fdr'")

        self.estimator = model
        self._trained = True


    def _fit(self, psms: PsmDataset, train_fdr: float = 0.01,
             max_iter: int = 10, msg: bool = True,
             kwargs: Dict = None) -> Tuple[pd.DataFrame, str]:
        """Fit an SVM model using the Percolator procedure"""
        best_feat, feat_pass, feat_target = psms.find_best_feature(train_fdr)

        if msg:
            _best_feat_msg(best_feat, feat_pass)

        # Normalize Features
        self.feature_names = psms.features.columns.tolist()
        self._train_mean = psms.features.mean(axis=0)
        self._train_std = psms.features.std(axis=0)
        norm_feat = psms.features.copy()
        norm_feat = np.divide((norm_feat - self._train_mean),
                              self._train_std,
                              out=np.zeros_like(norm_feat),
                              where=(self._train_std != 0))

        # Initialize Model and Training Variables
        target = feat_target
        model = sklearn.base.clone(self._base_estimator)

        # Grid search parameters, if required:
        if kwargs is not None:
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

        # Wrap up: Transform weights to original feature scale.
        #weights = pd.DataFrame({"Normalized": model.coef_.flatten()},
        #                       index=feat_names)

        #weights["Unnormalized"] = np.divide(weights.Normalized, feat_stdev,
        #                                    out=np.zeros_like(feat_stdev),
        #                                    where=(feat_stdev != 0))

        #int_sub = (feat_mean / feat_stdev * weights.Normalized).sum()
        #norm_int = model.intercept_ - int_sub

        #weights = weights.append(pd.DataFrame({"Normalized": model.intercept_,
        #                                       "Unnormalized": norm_int},
        #                                      index=["m0"]))

        return (model, feat_pass, num_passed, best_feat)

# Utility Functions -----------------------------------------------------------
def _best_feat_msg(best_feat, num_pass):
    """Log the best feature and the number of positive PSMs."""
    logging.info("Selected feature '%s' as initial direction.", best_feat)
    logging.info("Could separate %i training set positives in that direction.",
                 num_pass)

def _fold_msg(best_feat, best_feat_pass, num_pass, fold=None):
    """Logging messages for each fold"""
    if fold is not None:
        logging.info("Fold %i", fold)
        _best_feat_msg(best_feat, best_feat_pass)

    logging.info("Positive PSMs by iteration:")
    num_passed = [f"{n}" for i, n in enumerate(num_pass)]
    num_passed = "->".join(num_passed)
    logging.info("%s\n", num_passed)
