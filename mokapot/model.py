"""
This module defines the model class to used my Molokai.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn.svm as svm

from mokapot.dataset import PsmDataset
from mokapot.qvalues import tdc

class MokapotSVM():
    """
    Create a Linear SVM model for classifying PSMs.
    """
    def __init__(self, weights: pd.Series = None, intercept: float = 0,) \
        -> None:
        """Initialize a MokapotSVM object"""
        self.weights = weights
        self.intercept = intercept

    def predict(self, psms: PsmDataset) -> np.ndarray:
        """
        Score the PSMs in a PsmDataset

        Parameters
        ----------
        psms : mokapot.PsmDataset
            The collection of PSMs to score.

        Returns
        -------
        The SVM score for each sample.
        """
        if self.weights is None:
            raise ValueError("This model is untrained and unitialized. Run "
                             "fit() or load_weights() first.")

        # Load features
        feat_names = psms.features.columns.tolist()
        if set(feat_names) != set(self.feature_names):
            raise ValueError("Features of the PsmDataset do not match the "
                             "features used to fit the model.")

        feat = psms.features[self.weights.index].values

        # Make predictions
        return np.dot(feat, self.weights.values) + self.intercept

    def fit(self, psms: PsmDataset, train_fdr: float = 0.01,
            max_iter: int = 10) -> None:
        """Fit an SVM model using the Percolator procedure"""
        best_feat, feat_pass, feat_target = psms.find_best_feature(train_fdr)
        logging.info("Selected feature '%s' as initial direction.\n"
                     "-> Could separate %i training set positives with q<%f "
                     "in that direction.", best_feat, feat_pass, train_fdr)

        # Normalize Features
        feat_names = psms.features.columns.tolist()
        feat_mean = psms.features.mean(axis=1)
        feat_stdev = psms.features.std(axis=1) + np.finfo(float).tiny
        norm_feat = (psms.features - feat_mean) / feat_stdev

        # Initialize Model and Training Variables
        target = feat_target
        model = svm.LinearSVC(dual=psms.dual, class_weight="balanced")

        # Begin training loop
        target = feat_target
        for i in range(max_iter):
            # Fit the model
            samples = norm_feat.values[target.astype(bool), :]
            iter_targ = target[target.astype(bool)]
            model.fit(samples, iter_targ)

            # Update scores
            scores = model.decision_function(norm_feat)

            # Update target
            qvals = tdc(scores, target=(psms.label+1)/2)
            unlabeled = np.logical_and(qvals > train_fdr, psms.label == 1)
            target = self.label
            target[unlabeled] = 0
            num_pass = (target == 1).sum()

            logging.info("Iteration %i:\t Estimated %i PSMs with q<%f.",
                         i+1, num_pass, train_fdr)

        # Wrap up
        if feat_pass > num_pass:
            raise RuntimeError("No improvement was detected with model "
                               "training. Consider a less stringent value for "
                               "'train_fdr'.")

        feat_mean = feat_mean.append(pd.Seri)
        weights = np.append(model.coef_, model.intercept_)
        weights = pd.DataFrame({"Normalized": weights},
                               index=feat_names + ["m0"])

        weights["Unnormalized"] = weights.Normalized

