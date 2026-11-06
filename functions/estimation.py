import numpy as np
import scipy.stats as stat

# deskriptiv statistik -----------------------------------

def mean(x):
    """ 
    middelværdi 
    """
    return np.mean(x)

def median(x):
    """ 
    median
    """
    return np.median(x)

def variance(x):
    """
    varians
    """
    return np.var(x, ddof=1)

def ci_mean(data, confidence=0.95):

    n = len(data)
    xbar = np.mean(data)
    s = np.std(data, ddof=1)

    alpha = 1 - confidence
    t = stat.t.ppf(1 - alpha/2, n - 1)

    margin = t * s / np.sqrt(n)

    return (xbar - margin, xbar + margin)

def std(x):
    """
    
    """
    return np.std(x, ddof=1)

def minimum(x):
    """
    min værdi
    """
    return np.min(x)

def maximum(x):
    """
    max værdi
    """
    return np.max(x)