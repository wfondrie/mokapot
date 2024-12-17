"""
The idea of this module is to encapsulate all algorithms including their
"standard" parameters into callable objects.

Currently, there is only the QvalueAlgorithm for training here. But we could
also put more algo's here (e.g. for the peps, differentiate between qvalues for
training and for confidence estimation, etc). Either by specific classes or
maybe by some algorithm registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from typeguard import typechecked

import mokapot.peps as peps
import mokapot.qvalues as qvalues
import mokapot.qvalues_storey as qvalues_storey


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


class TDCPi0Algorithm(Pi0EstAlgorithm):
    def estimate(self, scores, targets):
        target_decoy_ratio = targets.sum() / (~targets).sum()
        return target_decoy_ratio


class StoreyPi0Algorithm(Pi0EstAlgorithm):
    def __init__(self, method: str, eval_lambda):
        self.method = method
        self.eval_lambda = eval_lambda

    def estimate(self, scores, targets):
        pvals = qvalues_storey.empirical_pvalues(
            scores[targets], scores[~targets], mode="best"
        )
        pi0est = qvalues_storey.estimate_pi0(
            pvals,
            method=self.method,
            lambdas=np.arange(0.2, 0.8, 0.01),
            eval_lambda=self.eval_lambda,
        )
        return pi0est.pi0


class SlopePi0Algorithm(Pi0EstAlgorithm):
    def estimate(self, scores, targets):
        hist_data = peps.hist_data_from_scores(scores, targets)
        hist_data.as_densities()
        _, target_density, decoy_density = hist_data.as_densities()
        return peps.estimate_pi0_by_slope(target_density, decoy_density)


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


class StoreyQvalueAlgorithm(QvalueAlgorithm):
    def __init__(self, *, pvalue_method="best"):
        super().__init__()
        self.pvalue_method = pvalue_method

    def qvalues(self, scores, targets, desc):
        pi0 = Pi0EstAlgorithm.pi0_algo.estimate(scores, targets)

        stat1 = scores[targets]
        stat0 = scores[~targets]

        pvals1 = qvalues_storey.empirical_pvalues(stat1, stat0, mode=self.pvalue_method)
        qvals1 = qvalues_storey.qvalues(pvals1, pi0=pi0, small_p_correction=False)

        qvals0 = np.interp(stat0, stat1, qvals1)
        qvals = np.zeros_like(scores)
        qvals[targets] = qvals1
        qvals[~targets] = qvals0
        return qvals

    def long_desc(self):
        return "storey"


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
            pi0_algorithm = StoreyPi0Algorithm("bootstrap", config.pi0_eval_lambda)
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
        case _:
            raise NotImplementedError
    QvalueAlgorithm.set_algorithm(qvalue_algorithm)


QvalueAlgorithm.set_algorithm(TDCQvalueAlgorithm())

# config.peps_algorithm
# config.peps_error
# config.qvalue_algorithm
# config.tdc
# config.stream_confidence

# config.pi0_algorithm
# config.pi0_eval_lambda
