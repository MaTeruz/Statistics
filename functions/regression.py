import numpy as np
import scipy.stats as stat
import matplotlib as plt

def regression(x, y):

    return stat.linregress(x, y)

def regression_plot(x, y):

    result = stat.linregress(x, y)

    plt.scatter(x, y)

    plt.plot(
        x,
        result.intercept + result.slope*np.array(x),
        color="red"
    )

    plt.show()