from abc import abstractmethod, ABC

import numpy as np
import scipy as sp
from typeguard import typechecked


# This file (class) is only for checking validity of TD modelling assumptions.


@typechecked
def set_mu_std(dist: sp.stats.rv_continuous, mu: float, std: float):
    """Modifies distribution parameters to have specified mean and standard
    deviation.

    Note: the input distribution needs to have finite mean and standard
    deviation for this method to work.

    Parameters
    ----------
    dist : sp.stats.rv_continuous
        The continuous random variable distribution object.
    mu : float
        The desired mean value for the distribution.
    std : float
        The desired standard deviation value for the distribution.

    Returns
    -------
    dist
        The distribution object with updated mean and standard deviation.
    """
    kwds = dist.kwds
    kwds["loc"] = 0
    kwds["scale"] = 1
    rv0 = dist.dist(**kwds)
    kwds["scale"] = std / rv0.std()
    rv1 = dist.dist(**kwds)
    kwds["loc"] = mu - rv1.mean()
    return dist.dist(**kwds)


@typechecked
def set_support(dist, lower: float, upper: float):
    """Modifies distribution object to have fixed support.

    Note: the input distribution must have finite support already.

    Parameters
    ----------
    dist : sp.stats.rv_continuous
        The continuous random variable distribution object.
    lower : float
        The new lower limit of the support.
    upper : float
        The new upper limit of the support.

    Returns
    -------
    dist
        The distribution object with updated support.
    """
    kwds = dist.kwds
    kwds["loc"] = 0
    kwds["scale"] = 1
    rv0 = dist.dist(**kwds)
    kwds["scale"] = (upper - lower) / (rv0.support()[1] - rv0.support()[0])
    rv1 = dist.dist(**kwds)
    kwds["loc"] = lower - rv1.support()[0]
    return dist.dist(**kwds)


class TDModel(ABC):
    """Abstract base class for target-decoy models.

    Attributes:
        R0 (object): The distribution model for decoy scores.
        R1 (object): The distribution model for true target scores.
        pi0 (float): The fraction of foreign spectra (i.e. spectra for which
          the generating spectrum is not in the database, see [Keich 2015].

    Methods:
        sample_decoys(N):
            Generates N decoy scores from the (initial) decoy score
            distribution.

        sample_true_targets(N):
            Generates N true target scores from the (initial) target score
            distribution.

        sample_targets(N, include_is_fd=True, shuffle_result=True):
            Generates N target scores by sampling from both the target
            and decoy score distributions.

        sample_scores(N):
            Abstract method for generating N target and decoy scores.

        decoy_pdf(x):
            Computes the probability density function of the decoy score
            distribution at x.

        true_target_pdf(x):
            Computes the probability density function of the target score
            distribution at x.

        decoy_cdf(x):
            Computes the cumulative distribution function of the decoy score
            distribution at x.

        true_target_cdf(x):
            Computes the cumulative distribution function of the target score
            distribution at x.

        true_pep(x):
            Computes the posterior error probability for a given score x.

        true_fdr(x):
            Computes the false discovery rate for a given score x.

        get_sampling_pdfs(x):
            Abstract method for getting the sampling PDFs for a given score x.
    """

    def __init__(self, R0, R1, pi0):
        self.R0 = R0
        self.R1 = R1
        self.pi0 = pi0

    def sample_decoys(self, N):
        return self.R0.rvs(N)

    def sample_true_targets(self, N):
        return self.R1.rvs(N)

    def sample_targets(self, N, include_is_fd=True, shuffle_result=True):
        NT = N
        NT0 = int(np.round(self.pi0 * NT))
        NT1 = NT - NT0
        R0 = self.R0
        R1 = self.R1

        nat1 = R1.rvs(NT1)
        nat0 = R0.rvs(NT1)
        target_scores = np.concatenate((np.maximum(nat1, nat0), R0.rvs(NT0)))
        is_fd = np.concatenate((nat1 < nat0, np.full(NT0, True)))

        if shuffle_result:
            indices = np.arange(target_scores.shape[0])
            np.random.shuffle(indices)
            target_scores = target_scores[indices]
            is_fd = is_fd[indices]

        if include_is_fd:
            return target_scores, is_fd
        else:
            return target_scores

    @abstractmethod
    def sample_scores(self, N):
        pass

    def _sample_both(self, NT, ND):
        target_scores, is_fd = self.sample_targets(NT, include_is_fd=True)
        decoy_scores = self.sample_decoys(ND)
        return target_scores, decoy_scores, is_fd

    @staticmethod
    def _sort_and_return(scores, is_target, is_fd):
        sort_idx = np.argsort(-scores)
        sorted_scores = scores[sort_idx]
        is_target = is_target[sort_idx]
        is_fd = is_fd[sort_idx]
        return sorted_scores, is_target, is_fd

    def decoy_pdf(self, x):
        return self.R0.pdf(x)

    def true_target_pdf(self, x):
        return self.R1.pdf(x)

    def decoy_cdf(self, x):
        return self.R0.cdf(x)

    def true_target_cdf(self, x):
        return self.R1.cdf(x)

    def true_pep(self, x):
        T_pdf, TT_pdf, FT_pdf, D_pdf, pi0 = self.get_sampling_pdfs(x)
        return pi0 * FT_pdf / T_pdf

    def true_fdr(self, x):
        if any(np.diff(x) < 0):
            raise ValueError("x must be non-decreasing, but wasn't'")

        # This pi0 is in both cases the Storey pi0, not the Keich pi0
        T_pdf, TT_pdf, FT_pdf, D_pdf, FDR = self.get_sampling_pdfs(x)

        fdr = FDR * np.flip(
            np.cumsum(np.flip(FT_pdf)) / np.cumsum(np.flip(T_pdf))
        )

        return fdr

    @staticmethod
    def _integrate(func_values, x):
        # Note: this is a bit primitive but does it's job here. Nicer would be
        # an adapted higher order integration rule that integrates over the
        # decoy probability density (which is all we need this method for), but
        # this one does the job
        return sp.integrate.trapz(func_values, x)

    def _get_input_pdfs_and_cdfs(self, x):
        pi0 = self.pi0
        X0_pdf = self.true_target_pdf(x)
        X0_cdf = self.true_target_cdf(x)
        X_pdf = (1 - pi0) * X0_pdf
        X_cdf = pi0 + (1 - pi0) * X0_cdf
        Y_pdf = self.decoy_pdf(x)
        Y_cdf = self.decoy_cdf(x)
        return X_pdf, X_cdf, Y_pdf, Y_cdf

    @abstractmethod
    def get_sampling_pdfs(self, x):
        pass


class TDCModel(TDModel):
    """A TDModel class for target decoy competition or concatenated search"""

    def sample_scores(self, N):
        target_scores, decoy_scores, is_fd = self._sample_both(N, N)
        is_target = target_scores >= decoy_scores
        all_scores = np.where(is_target, target_scores, decoy_scores)
        return self._sort_and_return(all_scores, is_target, is_fd)

    def get_sampling_pdfs(self, x):
        X_pdf, X_cdf, Y_pdf, Y_cdf = self._get_input_pdfs_and_cdfs(x)

        DP = TDModel._integrate(X_cdf * Y_cdf * Y_pdf, x)
        FDR = DP / (1 - DP)
        T_pdf = (X_pdf * Y_cdf**2 + X_cdf * Y_cdf * Y_pdf) / (1 - DP)
        TT_pdf = X_pdf * Y_cdf**2 / (1 - 2 * DP)
        FT_pdf = (X_cdf * Y_cdf * Y_pdf) / DP
        D_pdf = FT_pdf
        return T_pdf, TT_pdf, FT_pdf, D_pdf, FDR


class STDSModel(TDModel):
    """A TDModel class for separate search"""

    def sample_scores(self, NT, ND=None):
        ND = NT if ND is None else ND
        target_scores, decoy_scores, is_fd = self._sample_both(NT, ND)
        all_scores = np.concatenate((target_scores, decoy_scores))
        is_target = np.concatenate((np.full(NT, True), np.full(ND, False)))
        is_fd = np.concatenate((
            is_fd,
            np.full(ND, False),
        ))  # is_fd value for decoys is irrelevant
        return self._sort_and_return(all_scores, is_target, is_fd)

    def get_sampling_pdfs(self, x):
        X_pdf, X_cdf, Y_pdf, Y_cdf = self._get_input_pdfs_and_cdfs(x)

        FDR = TDModel._integrate(Y_pdf * X_cdf, x)
        T_pdf = X_pdf * Y_cdf + X_cdf * Y_pdf
        TT_pdf = X_pdf * Y_cdf / (1 - FDR)
        FT_pdf = X_cdf * Y_pdf / FDR
        D_pdf = Y_pdf
        return T_pdf, TT_pdf, FT_pdf, D_pdf, FDR
