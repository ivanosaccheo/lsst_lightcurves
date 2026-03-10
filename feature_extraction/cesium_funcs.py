"""
Cesium custom feature functions
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
## custom cesium feature functions


def mean_variance(t, x, e):
    """STD normalized by median"""
    return np.std(x)/np.median(x)

def weighted_average(t, x, e):
    """Arithmetic mean of observed values, weighted by measurement errors."""
    return np.average(x, weights=1. / (e**2))

def pair_slope_trend(t, x, e):
    """
    Considering the last 30 (time-sorted) measurements of source magnitude,
    the fraction of increasing first differences minus the fraction of
    decreasing first differences.
    """
    data_last = x[-30:]
    return (float(len(np.where(np.diff(data_last) > 0)[0]) -
            len(np.where(np.diff(data_last) <= 0)[0])) / 30)

def standard_deviation(t,x,e):
    return np.std(x)

def small_kurtosis(t, x, e):
    """
    Small Kurtosis
    
    See http://www.xycoon.com/peakedness_small_sample_test_1.html
    """
    N = len(x)
    
    # if N = 1 or 2 or 3
    if N in [1, 2, 3]:
        return np.nan
    
    mean = np.mean(x)
    std = np.std(x)

    S = sum(((x - mean) / std) ** 4)

    c1 = float(N * (N + 1)) / ((N - 1) * (N - 2) * (N - 3))
    c2 = float(3 * (N - 1) ** 2) / ((N - 2) * (N - 3))

    return c1 * S - c2

def chi2_per_dof(t, x, e):
    "Chi squared (relative to the mean) per DOF"
    
    weighted_mean = weighted_average(t, x, e)
    N = len(x)
    
    return (((x-weighted_mean)/e)**2).sum()/(N-1)


def excess_var(t, x, e):
    """Excess variance, see arXiv:1710.10943"""
    
    weights = 1/e**2
    weighted_mean = weighted_average(t, x, e)
    delta_x = (x - weighted_mean)/np.sqrt(1-weights/np.sum(weights))
    
    return np.mean(delta_x**2 - e**2)

def normed_evar(t, x, e):
    """Normalized excess variance"""
    N = len(x)
    weighted_mean = weighted_average(t, x, e)
    evar = np.sum((x-weighted_mean)**2 -(e**2))
    # returned normed evar
    return evar/(N*weighted_mean**2)
    
def rcs(t, x, e):
    """Range of a cumulative sum, see 'Kim et al. 2011'"""
    
    sigma = np.std(x)
    N = len(x)
    m = np.mean(x)
    s = np.cumsum(x - m) * 1.0 / (N * sigma)
    R = np.max(s) - np.min(s)
    
    return R

def von_N_ratio(t, x, e):
    """Compute von Neumann ratio"""
    
    delta_x = x[1:] -x[:-1]
    N = len(x)
    return (delta_x**2).sum()/(N-1)/np.var(x)

def min_dt(t, x, e):
    """Minimum time seperation between two observations"""
    return np.min(t[1:] -t[:-1])

def mean_error(t,x,e):
    return np.nanmean(e)

def Pvar(t, x, e):
    "Pvar "
    weighted_mean = weighted_average(t, x, e)
    N = len(x)
    chi_2 = (((x-weighted_mean)/e)**2).sum()
    return stats.chi2.cdf(chi_2, (N-1))


