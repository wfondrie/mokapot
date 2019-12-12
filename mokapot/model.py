"""
This module defines the model class to used my Molokai.
"""
import os
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Union

import numpy as np
import pandas as pd
import sklearn.svm as svm

from .dataset import PsmDataset, merge
from .qvalues import tdc

class MokapotSVM():
    """
    Create a Linear SVM model for classifying PSMs.
    """
    def __init__(self, weights: pd.Series = None, intercept: float = 0) \
        -> None:
        """Initialize a MokapotSVM object"""
        self.weights = weights
        self.intercept = intercept

    def predict(self, psms: PsmDataset) -> None:
        """
        Score the PSMs in a PsmDataset

        Parameters
        ----------
        psms
            The collection of PSMs to score.
        """
        if self.weights is None:
            raise ValueError("This model is untrained and unitialized. Run "
                             "fit() or load_weights() first.")

        # For users, 'psms' should be a PsmDataset. However, if the MokapotSVM
        # was fit using the 'percolate()' method, self.weights and
        # self.intercept will be a tuple. In this case, this method should only
        # use the first set of weights.
        # NOTE: 'percolate()' calls 'predict' on a tuple of PsmDataset objects
        # of equal length to self.weights.

        # The standard user case:
        if isinstance(psms, PsmDataset):
            psms = (psms,)
            if isinstance(self.weights, pd.Series):
                weights = (self.weights,)
                intercepts = (self.intercept,)
            else:
                weights = self.weights[0]
                intercepts = self.intercept[0]

        # As used in 'percolate()'
        else:
            weights = self.weights
            intercepts = self.intercept

        # This should not be raised by users...
        if len(psms) != len(weights):
            raise ValueError("psms does not match weights")

        predictions = []
        for psm_set, weight, intercept in zip(psms, weights, intercepts):
            feat_names = psm_set.features.columns.tolist()
            if set(feat_names) != set(weight.index.tolist()):
                raise ValueError("Features of the PsmDataset do not match the "
                                 "features used to fit the MokapotSVM model")

            feat = psm_set.features[weight.index].values
            pred = np.dot(feat, weight.values) + intercept
            predictions.append(pred)

        if len(predictions) == 1:
            predictions = predictions[0]

        return predictions

    def fit(self, psms: PsmDataset, train_fdr: float = 0.01,
            max_iter: int = 10, progress_bar: bool = True) -> None:
        """Fit an SVM model using the Percolator procedure"""
        res = self._fit(psms, train_fdr, max_iter, progress_bar)
        weights, feat_pass, num_pass, _ = res

        if feat_pass > num_pass[-1]:
            raise RuntimeError("No improvement was detected with model "
                               "training. Consider a less stringent value for "
                               "'train_fdr'")

        logging.info("Learned SVM weights:\n%s", weights)
        self.weights = weights.Unnormalized[:-1]
        self.intercept = weights.Unnormalized[-1]

    def _fit(self, psms: PsmDataset, train_fdr: float = 0.01,
             max_iter: int = 10, msg: bool = True) -> Tuple[pd.DataFrame, str]:
        """Fit an SVM model using the Percolator procedure"""
        best_feat, feat_pass, feat_target = psms.find_best_feature(train_fdr)

        if msg:
            _best_feat_msg(best_feat, feat_pass)

        # Normalize Features
        feat_names = psms.features.columns.tolist()
        feat_mean = psms.features.mean(axis=0)
        feat_stdev = psms.features.std(axis=0)
        norm_feat = psms.features.copy()
        norm_feat = np.divide((norm_feat - feat_mean), feat_stdev,
                              out=np.zeros_like(norm_feat),
                              where=(feat_stdev != 0))

        #norm_feat = (psms.features.copy() - feat_mean) / feat_stdev

        # Initialize Model and Training Variables
        target = feat_target
        model = svm.LinearSVC(dual=psms.dual, class_weight="balanced")

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
        weights = pd.DataFrame({"Normalized": model.coef_.flatten()},
                               index=feat_names)

        weights["Unnormalized"] = np.divide(weights.Normalized, feat_stdev,
                                            out=np.zeros_like(feat_stdev),
                                            where=(feat_stdev != 0))

        int_sub = (feat_mean / feat_stdev * weights.Normalized).sum()
        norm_int = model.intercept_ - int_sub

        weights = weights.append(pd.DataFrame({"Normalized": model.intercept_,
                                               "Unnormalized": norm_int},
                                              index=["m0"]))

        return (weights, feat_pass, num_passed, best_feat)


    def percolate(self, psms: PsmDataset, train_fdr: float = 0.01,
                  test_fdr: float = 0.01, max_iter: int = 10,
                  folds: int = 3, max_workers: int = None) \
                  -> Tuple[pd.DataFrame]:
        """Run the tradiational Percolator algorithm with cross-validation"""
        logging.info(f"Splitting PSMs into {folds} folds...")
        train_sets, test_sets = psms.split(folds)

        # Need args for map:
        map_args = [self._fit, train_sets, [train_fdr]*folds,
                    [max_iter]*folds, [False]*folds]

        # Train models in parallel:
        self.weights = []
        self.intercept = []
        logging.info("Training SVM models by %i-fold cross-validation...\n", folds)
        with ProcessPoolExecutor(max_workers=max_workers) as prc:
            for split, results in enumerate(prc.map(*map_args)):
                _fold_msg(results[3], results[1], results[2], split+1)
                self.weights.append(results[0].Unnormalized[:-1])
                self.intercept.append(results[0].Unnormalized[-1])

        self.weights = tuple(self.weights)
        self.intercept = tuple(self.intercept)
        del train_sets

        logging.info("Scoring PSMs...")
        scores = self.predict(test_sets)

        # Add scores to test sets
        for test_set, score in zip(test_sets, scores):
            test_set.normalization_fdr = test_fdr
            test_set.scores = score

        dataset = merge(test_sets)
        return dataset.get_results()


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
