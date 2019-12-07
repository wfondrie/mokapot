"""
This module defines the model class to used my Molokai.
"""
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Union

import numpy as np
import pandas as pd
import sklearn.svm as svm

from mokapot.dataset import PsmDataset
from mokapot.qvalues import tdc

class MokapotSVM():
    """
    Create a Linear SVM model for classifying PSMs.
    """
    def __init__(self, weights: pd.Series = None, intercept: float = 0) \
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
        if set(feat_names) != set(self.weights.index.tolist()):
            raise ValueError("Features of the PsmDataset do not match the "
                             "features used to fit the model.")

        feat = psms.features[self.weights.index].values

        # Make predictions
        return np.dot(feat, self.weights.values) + self.intercept

    def fit(self, psms: PsmDataset, train_fdr: float = 0.01,
            max_iter: int = 10) -> None:
        """Fit an SVM model using the Percolator procedure"""
        weights, feat_pass, num_pass = self.__fit(psms, train_fdr, max_iter)

        if feat_pass > num_pass:
            raise RuntimeError("No improvement was detected with model "
                               "training. Consider a less stringent value for "
                               "'train_fdr'")

        logging.info("Learned SVM weights:\n%s", weights)
        self.weights = weights.Unnormalized[:-1]
        self.intercept = weights.Unnormalized[-1]

    def __fit(self, psms: PsmDataset, train_fdr: float = 0.01,
              max_iter: int = 10) -> Tuple[pd.DataFrame, str]:
        """Fit an SVM model using the Percolator procedure"""
        best_feat, feat_pass, feat_target = psms.find_best_feature(train_fdr)
        logging.info("Selected feature '%s' as initial direction.\n"
                     "\t-> Could separate %i training set positives "
                     "in that direction.", best_feat, feat_pass)

        # Normalize Features
        feat_names = psms.features.columns.tolist()
        feat_mean = psms.features.mean(axis=0)
        feat_stdev = psms.features.std(axis=0) + np.finfo(float).tiny
        norm_feat = (psms.features.copy() - feat_mean) / feat_stdev

        # Initialize Model and Training Variables
        target = feat_target
        model = svm.LinearSVC(dual=psms.dual, class_weight="balanced")

        # Begin training loop
        target = feat_target
        num_passed = []
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
            target = psms.label.copy()
            target[unlabeled] = 0
            num_pass = (target == 1).sum()
            num_passed.append(num_pass)

        # Wrap up
        feat_mean = feat_mean.append(pd.Series([0], index=["m0"]))
        feat_stdev = feat_stdev.append(pd.Series([1], index=["m0"]))
        weights = np.append(model.coef_, model.intercept_)
        weights = pd.DataFrame({"Normalized": weights},
                               index=feat_names + ["m0"])

        weights["Unnormalized"] = weights.Normalized * feat_stdev - feat_mean

        return (weights, feat_pass, num_passed)

    def predict(self, psms: Union[PsmDataset, Tuple[PsmDataset]]) -> None:
        """Apply the learned model to """

    def percolate(self, psms: PsmDataset, train_fdr: float = 0.01,
                  test_fdr: float = 0.01, max_iter: int = 10, folds: int = 3):
        """Run the tradiational Percolator algorithm with cross-validation"""
        train_sets, test_sets = psms.split(folds)

        # Need kwargs for map:
        map_args = [self.__fit, train_sets, [train_fdr]*folds, [max_iter]*folds]
        print(map_args)

        # Train models in parallel:
        trained_models = []
        logging.info("Training SVM models by %i-fold cross-validation...", folds)
        with ProcessPoolExecutor() as executor:
            for split, results in enumerate(executor.map(*map_args)):
                num_passed = str(results[2]).join("->")
                trained_models.append(results[0])
                logging.info("Split %i positive PSMs by iteration:\n\t%s",
                             split+1, num_passed)

        logging.info("Scoring PSMs...")
        
