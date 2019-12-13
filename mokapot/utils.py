"""
Utility functions
"""
import numpy as np

def unnormalize_weights(weights, intercept, feat_mean, feat_std):
    """Take in normalized weights, return unnormalized weights"""
    new_weights = np.divide(weights, feat_std,
                            out=np.zeros_like(weights),
                            where=(feat_std != 0))

    int_sub = np.divide(feat_mean, feat_std,
                        out=np.zeros_like(feat_mean),
                        where=(feat_std != 0))

    intercept = intercept - (int_sub * weights).sum()

    return new_weights, intercept
