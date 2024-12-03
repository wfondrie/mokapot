import numpy as np
import scipy as sp

def empirical_pvalues(s: np.ndarray[float], s0: np.ndarray[float]) -> np.ndarray[float]:
    """
    Computes the empirical p-values for a set of values.

    Parameters
    ----------
    s : np.ndarray[float]
        Array of data values/test statistics (typically scores) for which
        p-values are to be computed.

    s0 : np.ndarray[float]
        Array of data values (scores) simulated under the null hypothesis
        against which the data values `s` are to be compared.

    Returns
    -------
    np.ndarray[float]
        Array of empirical p-values corresponding to the input data array `s`.
    """
    emp_null = sp.stats.ecdf(s0)
    return emp_null.sf.evaluate(s)



