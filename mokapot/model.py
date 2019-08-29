"""
This module defines the model class to used my Molokai.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn.svm as svm

from mokapot.dataset import PsmDataset

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
        best_feat, feat_pass, target = psms.find_best_feature(train_fdr)
