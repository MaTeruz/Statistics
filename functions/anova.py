
import numpy as np
import scipy.stats as stat

# -----------------------------------

def anova(*groups):

    F, p = stat.f_oneway(*groups)
    return F, p
