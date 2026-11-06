import numpy as np
import scipy.stats as stat

# kendt sigma -----------------------------------
def ci_mean_known_sigma(xbar, sigma, n, confidence=0.95):
    """
    
    """
    alpha = 1 - confidence

    z = stat.norm.ppf(1 - alpha/2)

    margin = z*sigma/np.sqrt(n)

    return (xbar-margin, xbar+margin)


# ukendt sigma -----------------------------------
def ci_mean(data, confidence=0.95):
    """

    """
    n = len(data)

    xbar = np.mean(data)

    s = np.std(data, ddof=1)

    alpha = 1-confidence

    t = stat.t.ppf(1-alpha/2, n-1)

    margin = t*s/np.sqrt(n)

    return (xbar-margin, xbar+margin)

# hypothesis testing -----------------------------------

# single sample:
def t_test_1sample(data, mu0):

    """
    One-sample t-test

    H0: μ = mu0
    H1: μ ≠ mu0

    Returns:
        t_stat, p_value
    """

    t_stat, p_value = stat.ttest_1samp(data, mu0)

    return t_stat, p_value

# single sample z-test:
def z_test(xbar, mu0, sigma, n):
    """
    
    """

    z = (xbar-mu0)/(sigma/np.sqrt(n))

    p = 2*(1-stat.norm.cdf(abs(z)))

    return z, p

# two sample test:
def t_test_2sample(x, y):
    """
    
    """

    t_stat, p_value = stat.ttest_ind(x, y)

    return t_stat, p_value

# two sample paried test:
def paired_t_test(x, y):
    """

    """

    t_stat, p_value = stat.ttest_rel(x, y)

    return t_stat, p_value

