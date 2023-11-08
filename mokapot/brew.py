"""Defines a function to run the Percolator algorithm."""
import copy
import logging
import warnings
from collections.abc import Iterable, Iterator

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from . import utils
from .confidence import Confidence
from .dataset import PsmDataset, merge_datasets
from .model import Model, PercolatorModel

LOGGER = logging.getLogger(__name__)


class Barista:
    """Train and evaluate mokapot models using cross-validation.

    The provided examples are analyzed using the semi-supervised learning
    algorithm that was introduced by `Percolator <http://percolator.ms>`_.
    Cross-validation is used to ensure that the learned models to not overfit
    to the examples used for model training. If a multiple collections of
    examples are provided, they are aggregated for model training, and the
    confidence estimates are calculated separately for each collection.

    A list of previously trained models can be provided to the ``model``
    argument to rescore the PSMs in each fold. Note that the number of
    models must match ``folds``. Furthermore, it is valid to use the
    learned models on the same dataset from which they were trained,
    but they must be provided in the same order, such that the
    relationship of the cross-validation folds is maintained.

    Parameters
    ----------
    datasets : PsmDataset object or list of PsmDataset objects
        One or more :doc:`collections of PSMs <dataset>` objects.
        PSMs are aggregated across all of the collections for model
        training, but the confidence estimates are calculated and
        returned separately.
    model: Model object or list of Model objects, optional
        The :py:class:`mokapot.Model` object to be fit. The default is
        :code:`None`, which attempts to mimic the same support vector
        machine models used by Percolator. If a list of
        :py:class:`mokapot.Model` objects is provided, they are assumed
        to be previously trained models and will and one will be
        used to rescore each fold.
    folds : int, optional
        The number of cross-validation folds to use. PSMs originating
        from the same mass spectrum are always in the same fold.
    max_workers : int, optional
        The number of processes to use for model training. More workers
        will require more memory, but will typically decrease the total
        run time. An integer exceeding the number of folds will have
        no additional effect. Note that logging messages will be garbled
        if more than one worker is enabled.
    rng : np.random.Generator, int, optional
        A seed or generator used to generate splits, or None to use the
        default random number generator state.
    """

    def __init__(
        self,
        datasets: PsmDataset | Iterable[PsmDataset],
        model: Model | Iterable[Model] | None = None,
        folds: int = 3,
        max_workers: int = 1,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        """Initialize the barista."""
        self.datasets = utils.listify(datasets)
        self.folds = folds
        self.max_workers = max_workers
        self.rng = np.random.default_rng(rng)
        self.model = utils.listify(
            PercolatorModel() if model is None else model
        )

        self.unit = self.datasets[0].unit

        # Validate models:
        if all(m.is_trained for m in self.model):
            if not len(self.model) == self.folds:
                raise ValueError(
                    f"The number of trained models ({len(self.model)}) "
                    f"must match the number of folds ({self.folds})"
                )

            self.model.sort(key=lambda x: x.fold)
        elif any(m.is_trained for m in self.model):
            raise ValueError(
                "Some, but not all, of the models were previously trained."
            )
        elif len(self.model) > 1:
            raise ValueError("Only one untrained model is allowed.")
        else:
            self.model = [
                copy.deepcopy(self.model[0]) for _ in range(self.folds)
            ]

        # If multiple datasets are used, log the aggregate number of examples.
        if len(datasets) > 1:
            LOGGER.info("")
            LOGGER.info(
                "Found %i total %s.",
                sum(len(d) for d in self.datasets),
                self.unit,
            )

    def brew(self) -> Confidence | list[Confidence]:
        """Fit models and calculate confidence estimates.

        Returns
        -------
        Confidence or list of Confidence
            The results for each of the datasets.
        """
        # Create the training splits.
        # Note that Joblib doesn't work with generators.
        LOGGER.info("Splitting %s into %i folds...", self.unit, self.folds)
        train_sets = self.training_splits()
        if self.max_workers != 1:
            train_sets = list(train_sets)

        fitted = Parallel(n_jobs=self.max_workers, require="sharedmem")(
            delayed(_fit_model)(copy.copy(d), m, f)
            for f, (d, m) in enumerate(zip(train_sets, self.model))
        )

        results = []
        try:
            # Models were fit successfully.
            fitted.sort(key=lambda x: x.fold)
            fitted = list(fitted)
            LOGGER.info("Re-scoring held-out %s...", self.unit)
            for test_sets in self.testing_splits():
                scores = pl.Series(
                    "mokapot score",
                    np.concatenate(
                        [m.predict(d) for m, d in zip(fitted, test_sets)]
                    ),
                )
                combined = merge_datasets(_drop_cols(test_sets))
                results.append(combined.assign_confidence(scores, desc=True))

        except AttributeError:
            # Fallback on best feature.
            warnings.warn(
                "Learned model did not improve over the best feature. "
                "Now scoring by the best feature for each collection "
                f"of {self.unit}.",
            )
            for dataset in self.datasets:
                results.append(dataset.assign_confidence())

        if len(results) == 1:
            return results[0]

        return results

    def training_splits(self) -> Iterator[PsmDataset]:
        """Yield the training splits.

        Yields
        ------
        PsmDataset
            The merged training subset across all of the datasets.
        """
        for idx in range(self.folds):
            yield merge_datasets(
                d.fold(idx, train=True, folds=self.folds)
                for d in self.datasets
            )

    def testing_splits(self) -> Iterator[PsmDataset]:
        """Yield the test sets for each dataset.

        Yields
        ------
        PsmDataset
        """
        for dataset in self.datasets:
            yield [
                dataset.fold(i, train=False, folds=self.folds)
                for i in range(self.folds)
            ]


def brew(
    datasets: PsmDataset | Iterable[PsmDataset],
    model: Model | Iterable[Model] | None = None,
    folds: int = 3,
    max_workers: int = 1,
    rng: np.random.Generator | int | None = None,
) -> tuple[Confidence | list[Confidence], list[Model]]:
    """Re-score one or more collections of examples (e.g. PSMs).

    The provided examples are analyzed using the semi-supervised learning
    algorithm that was introduced by `Percolator <http://percolator.ms>`_.
    Cross-validation is used to ensure that the learned models to not overfit
    to the examples used for model training. If a multiple collections of
    examples are provided, they are aggregated for model training, and the
    confidence estimates are calculated separately for each collection.

    A list of previously trained models can be provided to the ``model``
    argument to rescore the PSMs in each fold. Note that the number of
    models must match ``folds``. Furthermore, it is valid to use the
    learned models on the same dataset from which they were trained,
    but they must be provided in the same order, such that the
    relationship of the cross-validation folds is maintained.

    Parameters
    ----------
    datasets : PsmDataset object or list of PsmDataset objects
        One or more :doc:`collections of PSMs <dataset>` objects.
        PSMs are aggregated across all of the collections for model
        training, but the confidence estimates are calculated and
        returned separately.
    model: Model object or list of Model objects, optional
        The :py:class:`mokapot.Model` object to be fit. The default is
        :code:`None`, which attempts to mimic the same support vector
        machine models used by Percolator. If a list of
        :py:class:`mokapot.Model` objects is provided, they are assumed
        to be previously trained models and will and one will be
        used to rescore each fold.
    folds : int, optional
        The number of cross-validation folds to use. PSMs originating
        from the same mass spectrum are always in the same fold.
    max_workers : int, optional
        The number of processes to use for model training. More workers
        will require more memory, but will typically decrease the total
        run time. An integer exceeding the number of folds will have
        no additional effect. Note that logging messages will be garbled
        if more than one worker is enabled.
    rng : np.random.Generator, int, optional
        A seed or generator used to generate splits, or None to use the
        default random number generator state.

    Returns
    -------
    Confidence object or list of Confidence objects
        An object or a list of objects containing the
        :doc:`confidence estimates <confidence>` at various levels
        (i.e. PSMs, peptides) when assessed using the learned score.
        If a list, they will be in the same order as provided in the
        `psms` parameter.
    list of Model objects
        The learned :py:class:`~mokapot.model.Model` objects, one
        for each fold.

    """
    maker = Barista(
        datasets=datasets,
        model=model,
        folds=folds,
        max_workers=max_workers,
        rng=rng,
    )

    results = maker.brew()
    return results, maker.model


def _fit_model(train_set: PsmDataset, model: Model, fold: int) -> Model | None:
    """Fit the estimator using the training data.

    Parameters
    ----------
    train_set : PsmDataset
        A PsmDataset that specifies the training data
    model : tuple of Model
        The mokapot model to train.
    fold : int
        The fold number.

    Returns
    -------
    model : mokapot.model.Model or None
        The trained model or None if training resulted in worse
        performance.
    """
    LOGGER.info("")
    LOGGER.info("=== Analyzing Fold %i ===", fold + 1)
    model.fold = fold
    was_trained = model.is_trained
    try:
        model.fit(train_set)
    except RuntimeError as msg:
        if str(msg) != "Model performs worse after training.":
            raise

        if model.is_trained and not was_trained:
            return None

    return model


def _drop_cols(datasets: Iterable[PsmDataset]) -> Iterable[PsmDataset]:
    """Drop unnecessary columns.

    This function should actually not be necessary, but for some reason
    Polars does not like our test set concatenations. My hope is that
    this function is removed soon.

    Parameters
    ----------
    datasets: Iterable[PsmDataset]
        The datasets

    Yields
    ------
    PsmDataset
        The PsmDataset with missing columns to save memory.
    """
    for dataset in datasets:
        # Matches confidence object init.
        keep_cols = [
            c
            for c in dataset.columns
            if c not in dataset.schema.features or c in dataset.schema.metadata
        ]
        dataset = copy.copy(dataset)
        dataset.schema = copy.deepcopy(dataset.schema)
        dataset.schema.features = []
        dataset._data = dataset.data.select(keep_cols)
        yield dataset
