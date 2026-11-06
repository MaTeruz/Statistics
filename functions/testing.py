import numpy as np
import scipy.stats as stat

# kendt sigma -----------------------------------
def ci_mean_known_sigma(xbar, sigma, n, confidence=0.95):
    """
    Confidence interval for a population mean when sigma is known.

    Parameters
    ----------
    xbar : float
        Sample mean.
    sigma : float
        Known population standard deviation.
    n : int
        Sample size.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    alpha = 1 - confidence

    z = stat.norm.ppf(1 - alpha/2)

    margin = z*sigma/np.sqrt(n)

    return (xbar-margin, xbar+margin)


# ukendt sigma -----------------------------------
def ci_mean(data, confidence=0.95):
    """
    Confidence interval for a population mean when sigma is unknown.

    Parameters
    ----------
    data : array-like
        Sample observations.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    tuple
        (lower_bound, upper_bound)
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

    prints
    -------
        t_stat, p_value

    Returns:
        t_stat, p_value
    """

    t_stat, p_value = stat.ttest_1samp(data, mu0)

    print("\nSingle-sample t-test (equal variances)")
    print(f"t = {t_stat:.4f}")
    print(f"p = {p_value:.4f}")

    return t_stat, p_value

# single sample z-test:
def z_test(xbar, mu0, sigma, n, alpha):
    """
    One-sample z-test for a population mean.

    H0: μ = mu0
    H1: μ ≠ mu0

    Parameters
    ----------
    xbar : float
        Sample mean.

    mu0 : float
        Hypothesized population mean.

    sigma : float
        Known population standard deviation.

    n : int
        Sample size.

    prints
    -------
        z, p_value, reject

    Returns
    -------
    tuple
        (z_statistic, p_value)
    """

    z = (xbar-mu0)/(sigma/np.sqrt(n))

    p = 2*(1-stat.norm.cdf(abs(z)))

    reject = p < alpha

    print("\nOne-sample z-test")
    print(f"z = {z:.4f}")
    print(f"p = {p:.4f}")
    print(f"Reject H0 = {reject}")

    return z, p, reject

# two sample test:
def t_test_2sample(x, y):
    """
    Two-sample t-test using raw observations.

    H0: μ1 = μ2
    H1: μ1 ≠ μ2

    Parameters
    ----------
    x : array-like
        Sample 1 observations.

    y : array-like
        Sample 2 observations.

    Returns
    -------
    tuple
        (t_statistic, p_value)
    """

    t_stat, p_value = stat.ttest_ind(x, y)

    print("\nTwo-sample t-test (equal variances)")
    print(f"t = {t_stat:.4f}")
    print(f"p = {p_value:.4f}")

    return t_stat, p_value


def t_test_equal_var(
    x1, x2,
    s1_sq, s2_sq,
    n1, n2,
    alpha
    ):
    """
    Two-sample t-test assuming equal variances.

    H0: μ1 = μ2
    H1: μ1 ≠ μ2
    
    prints
    -------
        t_stat, p_value, df, reject

    Returns
    -------
    t_stat, p_value, df
    """

    sp2 = (
        ((n1 - 1)*s1_sq + (n2 - 1)*s2_sq)
        /
        (n1 + n2 - 2)
    )

    t = (x1 - x2) / np.sqrt(
        sp2*(1/n1 + 1/n2)
    )

    df = n1 + n2 - 2

    p = 2*(1 - stat.t.cdf(abs(t), df))
    
    reject = p < alpha

    print("\nTwo-sample t-test (equal variances)")
    print(f"t = {t:.4f}")
    print(f"p = {p:.4f}")
    print(f"df = {df}")
    print(f"Reject H0 = {reject}")

    return t, p, df, reject

# two sample paried test:
def paired_t_test(x, y):
    """
    Paired t-test.

    H0: μd = 0
    H1: μd ≠ 0

    where

        d = x - y

    Assumptions
    -----------
    1. Observations are paired.
    2. Pairs are independent.
    3. The differences are approximately normally distributed.

    Parameters
    ----------
    x : array-like
        First sample observations.

    y : array-like
        Second sample observations.

    Prints
    ------
    t_statistic, p_value

    Returns
    -------
    tuple
        (t_statistic, p_value)
    """

    t_stat, p_value = stat.ttest_rel(x, y)

    print("\nPaired t-test")
    print(f"t = {t_stat:.4f}")
    print(f"p = {p_value:.4f}")

    return t_stat, p_value


def ci_difference_means_equal_var(
    x1, x2,
    s1_sq, s2_sq,
    n1, n2,
    confidence=0.95
    ):
    """
    Confidence interval for μ1 - μ2 assuming equal variances.

    Parameters
    ----------
    x1, x2 : float
        Sample means.

    s1_sq, s2_sq : float
        Sample variances.

    n1, n2 : int
        Sample sizes.

    confidence : float
        Confidence level.
 
    prints
    -------
        (lower_bound, upper_bound)
        
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """

    sp2 = ((n1 - 1)*s1_sq + (n2 - 1)*s2_sq) / (n1 + n2 - 2)

    se = np.sqrt(sp2 * (1/n1 + 1/n2))

    alpha = 1 - confidence

    df = n1 + n2 - 2

    t = stat.t.ppf(1 - alpha/2, df)

    margin = t * se

    diff = x1 - x2

    print("\nConfidence interval for μ1 - μ2 assuming equal variances")
    print(f"95% CI for μ1 - μ2 = ({diff - margin:.4f}, {diff + margin:.4f})")

    return diff - margin, diff + margin



def ci_difference_means_unequal_var(
    x1, x2,
    s1_sq, s2_sq,
    n1, n2,
    confidence=0.95
    ):
    """
    Confidence interval for μ1 - μ2 assuming unequal variances.

    Welch confidence interval.

    Parameters
    ----------
    x1, x2 : float
        Sample means.

    s1_sq, s2_sq : float
        Sample variances.

    n1, n2 : int
        Sample sizes.

    confidence : float
        Confidence level.

    prints
        -------
            (lower_bound, upper_bound)

    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """

    se = np.sqrt(s1_sq/n1 + s2_sq/n2)

    df = (
        (s1_sq/n1 + s2_sq/n2)**2
        /
        (
            ((s1_sq/n1)**2)/(n1-1)
            +
            ((s2_sq/n2)**2)/(n2-1)
        )
    )

    alpha = 1 - confidence

    t = stat.t.ppf(1 - alpha/2, df)

    margin = t * se

    diff = x1 - x2

    print("\nConfidence interval for μ1 - μ2 assuming unequal variances")
    print(f"95% CI for μ1 - μ2 = ({diff - margin:.4f}, {diff + margin:.4f})")

    return diff - margin, diff + margin

