from functions import *

# -----------------------------------
# --- data 
# -----------------------------------

# skirv din data her
data = [] 


# -----------------------------------
# --- sandsynlighed
# -----------------------------------



# -----------------------------------
# --- fordeling
# -----------------------------------



# -----------------------------------
# --- hypotese testing
# -----------------------------------

# example opg 10.2.2

# Consider a hypothosis test H_0 : mu_1 = mu_2 against H_1 : mu_1 neq mu_2
# suppose the sample sizes are
n1 = 15
n2 = 15

# the means are:
x1 = 4.7
x2 = 7.8

# and that 
s1 = 4
s2 = 6.25

# assume that S sigma_1^2 = sigma_2^2 and the data is drawn from a normal distribution: use
alpha = 0.05

# -----------------------------------
# solution:
# -----------------------------------

result = t_test_equal_var(
    x1, x2,
    s1, s2,
    n1, n2,
    alpha
)

lower, upper = ci_difference_means_equal_var(
    x1, x2,
    s1, s2,
    n1, n2
)


# -----------------------------------

# -----------------------------------