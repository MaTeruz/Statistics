import numpy as np
from math import comb
import scipy.stats as stat

# Binomialfordeling -----------------------------------

# når P(X = x)
def binomial_pmf(n, x, p):
    """
     Beregner P(X = x) for X ~ Bin(n,p), for Real n, 0 <= p <= 1, x = 0,1,...,n
    """
    return stat.binom.pmf(x, n ,p)


# når P(X <= x)
def binomial_cdf(n, x, p):
    """
    Beregner P(X <= x)
    """
    return stat.binom.cdf(x, n, p)


# Possion -----------------------------------

# P(X=x)

def poisson_pmf(x, lam):
    return stat.poisson.pmf(x, lam)


def poisson_cdf(x, lam):
    return stat.poisson.cdf(x, lam)



# normalfordeling -----------------------------------


def normal_prob(x, mu=0, sigma=1):
    """
    Beregner P(X <= x)
    """
    return stat.norm.cdf(x, loc=mu, scale=sigma)


def normal_interval(a, b, mu=0, sigma=1):
    """
    Beregner P(a <= X <= b)
    """
    return stat.norm.cdf(b, mu, sigma) - stat.norm.cdf(a, mu, sigma)


def normal_cdf(x, mu=0, sigma=1):
    """

    """
    return stat.norm.cdf(x, mu, sigma)

def normal_quantile(p, mu=0, sigma=1):
    """

    """
    return stat.norm.ppf(p, mu, sigma)

def normal_interval(a, b, mu=0, sigma=1):
    """

    """
    return stat.norm.cdf(b, mu, sigma) - stat.norm.cdf(a, mu, sigma)

# Sandsynlighed ud fra data
def empirical_prob(data, event):
    """
    Estimer sandsynlighed fra observationer
    """
    count = sum(1 for x in data if event(x))
    return count / len(data)


