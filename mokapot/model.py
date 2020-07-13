"""
mokapot implements an algorithm for training machine learning models to
distinguish high-scoring target peptide-spectrum matches (PSMs) from
decoy PSMs using an iterative procedure. It is the :py:class:`Model`
class that contains this logic. A :py:class:`Model` instance can be
created from any object with a scikit-learn estimator interface,
allowing a wide variety of models to be used. Once initialized,
the :py:meth:`Model.fit` method trains the underyling classifier
using :doc:`a collection of PSMs <dataset>` with this iterative
approach.
"""
import logging
import pickle

import numpy as np
import pandas as pd
import sklearn.base as base
import sklearn.svm as svm
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
from sklearn.exceptions import NotFittedError

LOGGER = logging.getLogger(__name__)

# Constants -------------------------------------------------------------------
PERC_GRID = {"class_weight": [{0: neg, 1: pos}
                              for neg in (0.1, 1, 10)
                              for pos in (0.1, 1, 10)]}


# Classes ---------------------------------------------------------------------
class Model():
    """
    A machine learning model to re-score PSMs.

    A linear support vector machine (SVM) model is used by default in an
    attempt emulate the SVM models in Percolator. Alternatively, any
    classifier with a `scikit-learn estimator interface
    <https://scikit-learn.org/stable/developers/develop.html#estimators>`_
    can be used. This class also supports hyper parameter optimization
    using classes from the :py:mod:`sklearn.model_selection`
    module, such as the :py:class:`~sklearn.model_selection.GridSearchCV`
    and :py:class:`~sklearn.model_selection.RandomizedSearchCV` classes.

    Parameters
    ----------
    estimator : classifier object, optional
        A classifier that is assumed to implement the scikit-learn
        estimator interface.
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
    is_trained : bool, optional
        Indicates if the model has already been trained.

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
    """
    def __init__(self, estimator=None, scaler=None, is_trained=False):
        """Initialize a Model object"""
        if estimator is None:
            svm_model = svm.LinearSVC(dual=False)
            estimator = ms.GridSearchCV(svm_model, param_grid=PERC_GRID,
                                        refit=False,
                                        cv=3)

        self.estimator = base.clone(estimator)
        self.features = None
        self.is_trained = is_trained

        if scaler == "as-is":
            self.scaler = DummyScaler()
        elif scaler is None:
            self.scaler = pp.StandardScaler()
        else:
            self.scaler = base.clone(scaler)

    def __repr__(self):
        """How to print the class"""
        trained = {True: "A trained", False: "An untrained"}

        return (f"{trained[self.is_trained]} mokapot.model.Model object:\n"
                f"\testimator: {self.estimator}\n"
                f"\tscaler: {self.scaler}\n"
                f"\tfeatures: {self.features}")

    def save(self, out_file):
        """
        Save the model to a file.

        Parameters
        ----------
        out_file : str
            The name of the file for the saved model.

        Returns
        -------
        str
            The output file name.

        Note
        ----
        Because classes may change between mokapot and scikit-learn
        versions, a saved model may not work when either is changed
        from the version that created the model.
        """
        with open(out_file, "wb+") as out:
            pickle.dump(self, out)

        return out_file

    def decision_function(self, psms):
        """
        Score a collection of PSMs

        Parameters
        ----------
        psms : PsmDataset object
            :doc:`A collection of PSMs <dataset>` to score.

        Returns
        -------
        numpy.ndarray
            A vector containing the score for each PSM.
        """
        if not self.is_trained:
            raise NotFittedError("This model is untrained. Run fit() first.")

        feat_names = psms.features.columns.tolist()
        if set(feat_names) != set(self.features):
            raise ValueError("Features of the input data do not match the "
                             "features of this Model.")

        feat = self.scaler.transform(psms.features.loc[:, self.features].values)
        return self.estimator.decision_function(feat)

    def predict(self, psms):
        """Alias for :py:meth:`decision_function`."""
        return self.decision_function(psms)

    def fit(self, psms, train_fdr=0.01, max_iter=10, direction=None):
        """
        Fit the machine learning model using the Percolator algorithm.

        The model if trained by iteratively learning to separate decoy
        PSMs from high-scoring target PSMs. By default, an initial
        direction is chosen as the feature that best separates target
        from decoy PSMs. A false discovery rate threshold is used to
        define how high a target must score to be used as a positive
        example in the next training iteration.

        Parameters
        ----------
        psms : PsmDataset object
            :doc:`A collection of PSMs <dataset>` from which to train
            the model.
        train_fdr : float, optional
            The maximum false discovery rate at which to consider a
            target PSM as a positive example.
        max_iter : int, optional
            The number of iterations to perform.
        direction : str or None, optional
            The name of the feature to use as the initial direction for
            ranking PSMs. The default, :code:`None`, automatically
            selects the feature that finds the most PSMs below the
            `train_fdr`. This
            will be ignored in the case the model is already trained.
        """
        if not sum(psms.targets):
            raise ValueError("No target PSMs were available for training.")
        elif not sum(~psms.targets):
            raise ValueError("No decoy PSMs were available for training.")
        elif len(psms.data) <= 200:
            logging.warning("Few PSMs are available for model training (%i). "
                            "The learned models may be unstable.",
                            len(psms.data))

        # Choose the initial direction
        LOGGER.info("Finding initial direction...")
        best_feat, feat_pass, feat_labels, _ = psms._find_best_feature(train_fdr)
        if direction is None and not self.is_trained:
            LOGGER.info("  - Selected feature %s with %i PSMs at q<=%g.",
                        best_feat, feat_pass, train_fdr)
            start_labels = feat_labels
        elif self.is_trained:
            scores = self.estimator.decision_function(psms.features)
            start_labels = psms._update_labels(scores, fdr_threshold=train_fdr)
            LOGGER.info("  - The pretrained model found %i PSMs at q<=%g.",
                        (start_labels == 1).sum(), train_fdr)
        else:
            desc_labels = psms._update_labels(psms.features[direction].values)
            asc_labels = psms._update_labels(psms.features[direction].values,
                                             desc=False)
            if (desc_labels == 1).sum() >= (asc_labels == 1).sum():
                start_labels = desc_labels
            else:
                start_labels = asc_labels

            LOGGER.info("  - Selected feature %s with %i PSMs at q<=%g.",
                        direction, (start_labels == 1).sum(), train_fdr)

        # Normalize Features
        self.features = psms.features.columns.tolist()
        norm_feat = self.scaler.fit_transform(psms.features)

        # Initialize Model and Training Variables
        if hasattr(self.estimator, "estimator"):
            LOGGER.info("Selecting hyperparameters...")
            cv_samples = norm_feat[feat_labels.astype(bool), :]
            cv_targ = (feat_labels[feat_labels.astype(bool)]+1)/2
            self.estimator.fit(cv_samples, cv_targ)
            best_params = self.estimator.best_params_
            model = self.estimator.estimator
            model.set_params(**best_params)
            LOGGER.info("  - best parameters: %s", best_params)
        else:
            model = self.estimator

        # Begin training loop
        target = start_labels
        num_passed = []
        LOGGER.info("Beginning training loop...")
        for i in range(max_iter):
            # Fit the model
            samples = norm_feat[target.astype(bool), :]
            iter_targ = (target[target.astype(bool)]+1)/2
            model.fit(samples, iter_targ)

            # Update scores
            scores = model.decision_function(norm_feat)

            # Update target
            target = psms._update_labels(scores, fdr_threshold=train_fdr)
            num_passed.append((target == 1).sum())
            LOGGER.info("  - Iteration %i: %i training PSMs passed.",
                        i, num_passed[i])

        # If the model performs worse than what was initialized:
        if (num_passed[-1] < (start_labels == 1).sum()
                or num_passed[-1] < feat_pass):
            raise RuntimeError("Model performs worse after training.")

        self.estimator = model

        weights = _get_weights(self.estimator, self.features)
        if weights is not None:
            LOGGER.info("Normalized feature weights in the learned model:")
            for line in weights:
                LOGGER.info("    %s", line)

        self.is_trained = True
        LOGGER.info("Done training.")


class DummyScaler():
    """
    Implements the interface of scikit-learn scalers, but does
    nothing to the data. This simplifies the training code.

    :meta private:
    """
    def fit(self, x):
        pass

    def fit_transform(self, x):
        return x

    def transform(self, x):
        return x


# Functions -------------------------------------------------------------------
def save_model(model, out_file):
    """
    Save a :py:class:`mokapot.model.Model` object to a file.

    Parameters
    ----------
    out_file : str
        The name of the file for the saved model.

    Returns
    -------
    str
        The output file name.

    Note
    ----
    Because classes may change between mokapot and scikit-learn versions,
    a saved model may not work when either is changed from the version
    that created the model.
    """
    return model.save(out_file)


def load_model(model_file):
    """
    Load a saved model for mokapot.

    The saved model can either be a saved :py:class:`mokapot.model.Model`
    object or the output model weights from Percolator. In Percolator,
    these can be obtained using the :code:`--weights` argument.

    Parameters
    ----------
    model_file : str
        The name of file from which to load the model.

    Returns
    -------
    mokapot.model.Model
        The loaded :py:class:`mokapot.model.Model` object.
    """
    # Try a percolator model first:
    try:
        weights = pd.read_csv(model_file, sep="\t", nrows=2).loc[1, :]
        logging.info("Loading the Percolator model.")

        weight_cols = [c for c in weights.index if c != "m0"]
        model = Model(estimator=svm.LinearSVC(), scaler="as-is",
                      is_trained=True)

        weight_vals = weights.loc[weight_cols]
        weight_vals = weight_vals[np.newaxis, :]
        model.estimator.coef_ = weight_vals
        model.estimator.intercept_ = weights.loc["m0"]
        model.features = weight_cols

    # Then try loading it with pickle:
    except UnicodeDecodeError:
        logging.info("Loading mokapot model.")
        with open(model_file, "rb") as mod_in:
            model = pickle.load(mod_in)

    return model


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
