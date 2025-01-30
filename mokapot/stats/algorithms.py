"""
The idea of this module is to encapsulate all algorithms including their
"standard" parameters into callable objects.

Currently, there is only the QvalueAlgorithm for training here. But we could
also put more algo's here (e.g. for the peps, differentiate between qvalues for
training and for confidence estimation, etc). Either by specific classes or
maybe by some algorithm registry.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from typeguard import typechecked

import mokapot.stats.peps as peps
import mokapot.stats.qvalues as qvalues
import mokapot.stats.qvalues_storey as qvalues_storey

LOGGER = logging.getLogger(__name__)


## Algorithms for pi0 estimation
class Pi0EstAlgorithm(ABC):
    # Derived classes: StoreyPi0Algorithm, TDSlopePi0Algorithm
    pi0_algo = None

    @abstractmethod
    def estimate(self, scores, targets):
        raise NotImplementedError

    @classmethod
    def set_algorithm(cls, pi0_algo: Pi0EstAlgorithm):
        cls.pi0_algo = pi0_algo

    @classmethod
    def long_desc(cls):
        return cls.pi0_algo.long_desc()


@typechecked
class TDCPi0Algorithm(Pi0EstAlgorithm):
    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]) -> float:
        targets_count = targets.sum()
        decoys_count = (~targets).sum()
        if decoys_count == 0:
            LOGGER.warning(
                f"Can't estimate pi0 with zero decoys (targets={targets_count}, "
                f"decoys={decoys_count}, total={len(targets)})"
            )
        decoy_target_ratio = decoys_count / targets_count
        return decoy_target_ratio

    def long_desc(self):
        return "decoy_target_ratio"


@typechecked
class StoreyPi0Algorithm(Pi0EstAlgorithm):
    def __init__(self, method: str, eval_lambda: float):
        self.method = method
        self.eval_lambda = eval_lambda

    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]) -> float:
        pvals = qvalues_storey.empirical_pvalues(
            scores[targets], scores[~targets], mode="conservative"
        )
        pi0est = qvalues_storey.estimate_pi0(
            pvals,
            method=self.method,
            lambdas=np.arange(0.2, 0.8, 0.01),
            eval_lambda=self.eval_lambda,
        )
        return pi0est.pi0

    def long_desc(self):
        return f"storey_pi0(method={self.method}, lambda={self.eval_lambda})"


class SlopePi0Algorithm(Pi0EstAlgorithm):
    def __init__(self, hist_bins="scott", slope_threshold=0.9):
        self.bins = hist_bins
        self.slope_threshold = slope_threshold

    def estimate(self, scores, targets):
        hist_data = peps.hist_data_from_scores(scores, targets, bins=self.bins)
        hist_data.as_densities()
        _, target_density, decoy_density = hist_data.as_densities()
        return peps.estimate_pi0_by_slope(
            target_density, decoy_density, threshold=self.slope_threshold
        )

    def long_desc(self):
        return (
            f"pi0_by_slope(hist_bins={self.bins}, "
            f"slope_threshold={self.slope_threshold})"
        )


## Algorithms for qvalue computation
@typechecked
class QvalueAlgorithm(ABC):
    qvalue_algo = None

    @abstractmethod
    def qvalues(self, scores, targets, desc):
        raise NotImplementedError

    @classmethod
    def set_algorithm(cls, qvalue_algo: QvalueAlgorithm):
        cls.qvalue_algo = qvalue_algo

    @classmethod
    def eval(cls, scores, targets, desc=True):
        return cls.qvalue_algo.qvalues(scores, targets, desc)

    @classmethod
    def long_desc(cls):
        return cls.qvalue_algo.long_desc()


@typechecked
class TDCQvalueAlgorithm(QvalueAlgorithm):
    def qvalues(self, scores, targets, desc):
        return qvalues.tdc(scores, target=targets, desc=desc)

    def long_desc(self):
        return "mokapot tdc algorithm"


@typechecked
class CountsQvalueAlgorithm(QvalueAlgorithm):
    def __init__(self, tdc: bool):
        self.tdc = tdc

    def qvalues(self, scores: np.ndarray[float], targets: np.ndarray[bool], desc: bool):
        if not desc:
            scores = -scores
        qvals = qvalues.qvalues_from_counts(scores, targets, is_tdc=self.tdc)
        return qvals

    def long_desc(self):
        return "qvalue_by_counts"


@typechecked
class StoreyQvalueAlgorithm(QvalueAlgorithm):
    def __init__(self, *, pvalue_method="best", pi0_algo=None):
        super().__init__()
        self.pvalue_method = pvalue_method
        self.pi0_algo = pi0_algo

    def qvalues(self, scores: np.ndarray[float], targets: np.ndarray[bool], desc: bool):
        pi0_algo = self.pi0_algo or Pi0EstAlgorithm.pi0_algo
        pi0 = pi0_algo.estimate(scores, targets)
        LOGGER.debug(f"pi0-estimate: pi0={pi0}, algo={pi0_algo.long_desc()}")
        if not desc:
            scores = -scores
        qvals = qvalues.qvalues_from_storeys_algo(scores, targets, pi0)
        return qvals

    def long_desc(self):
        return "qvalues_by_storeys_algo"


# Algoritms for pep computation
class PEPAlgorithm(ABC):
    # Derived classes: TriqlerPEPAlgorithm, HistNNLSAlgorithm, KDENNLSAlgorithm
    # Not yet: StoreyLFDRAlgorithm (probit, logit)
    pass


# Configuration of algorithms via command line arguments
def configure_algorithms(config):
    tdc = config.tdc

    match (config.pi0_algorithm, config.tdc):
        case ("default", True) | ("ratio", _):
            if not tdc:
                raise ValueError(
                    "Can't use 'ratio' for pi0 estimation, when 'tdc' is false"
                )
            pi0_algorithm = TDCPi0Algorithm()
        case ("default", False) | ("storey_fixed", _):
            pi0_algorithm = StoreyPi0Algorithm("fixed", config.pi0_eval_lambda)
        case ("storey_bootstrap", _):
            pi0_algorithm = StoreyPi0Algorithm("bootstrap", config.pi0_eval_lambda)
        case ("storey_smoother", _):
            pi0_algorithm = StoreyPi0Algorithm("smoother", config.pi0_eval_lambda)
        case ("slope", _):
            pi0_algorithm = SlopePi0Algorithm()
        case _:
            raise NotImplementedError
    Pi0EstAlgorithm.set_algorithm(pi0_algorithm)

    match (config.qvalue_algorithm, config.tdc):
        case ("default", True) | ("tdc", _):
            if not tdc:
                raise ValueError(
                    "Can't use 'tdc' algorithm for q-values, when 'tdc' is false"
                )
            qvalue_algorithm = TDCQvalueAlgorithm()
        case ("default", False) | ("storey", _):
            qvalue_algorithm = StoreyQvalueAlgorithm()
        case ("from_counts", _):
            qvalue_algorithm = CountsQvalueAlgorithm(tdc)
        case _:
            raise NotImplementedError
    QvalueAlgorithm.set_algorithm(qvalue_algorithm)

    LOGGER.debug(f"pi0 algorithm: {pi0_algorithm.long_desc()}")
    LOGGER.debug(f"q-value algorithm: {qvalue_algorithm.long_desc()}")


QvalueAlgorithm.set_algorithm(TDCQvalueAlgorithm())


"""
todo:
 * try slope method for pi0 estimation on pvalues
 * check and improve pvalue computation for discrete distributions
 * check pi0 estimation with ratios for tdc/storey
 * implement storey with bootstrap
"""
