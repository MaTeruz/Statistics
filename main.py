from functions import *

data = [12, 15, 13, 16, 14]

print(mean(data))
print(ci_mean(data))

from functions import *

print("=== BINOMIAL ===")

print(binomial_pmf(3, 10, 0.2))
print(binomial_cdf(3, 10, 0.2))

print("\n=== POISSON ===")

print(poisson_pmf(4, 2))
print(poisson_cdf(4, 2))

print("\n=== NORMAL ===")

print(normal_cdf(1.96))
print(normal_quantile(0.975))
print(normal_interval(-1.96, 1.96))

