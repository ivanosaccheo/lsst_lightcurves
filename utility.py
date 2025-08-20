import numpy as np 
import pandas as pd
import os
import glob 
import catalog_extraction_library as celib 
from astropy.stats import sigma_clipped_stats
from scipy.stats import binned_statistic


def get_truth_label(truth_df):
    label = np.full(len(truth_df), 1)  #galaxies
    label[~np.isfinite(truth_df["Z"])] = 0
    
    is_agn = truth_df["is_agn"].fillna(False).astype(bool)
    is_type2 = truth_df["is_optical_type2"].fillna(False).astype(bool)

    label[is_agn & is_type2] = 2
    label[is_agn & ~is_type2] = 3
    return label


def get_variable_label(truth_df):
    label = get_truth_label(truth_df)
    eclipsing_binary = np.genfromtxt("../input/eclipsing_binary.dat", skip_header = True).astype(int)
    var_label = np.full(len(truth_df), 0)  #galaxies and non variable stars
    var_label[truth_df["ID"].isin(eclipsing_binary)] =1 # variable star
    var_label[label == 2] = 2
    var_label[label == 3] = 3
    return var_label


def get_features_table(path = "/staff1/saccheo/agile/summary_features/diff/2025-08-19"):
    features = []
    for name in os.listdir(path):
        fname = os.path.join(path, name)
        features.append(pd.read_parquet(fname))
    features = pd.concat(features)
    try:
        features.columns = ['_'.join([feature, band]) if feature != 'objectId' else 'objectId' for feature, band in features.columns]
    except ValueError:
        pass
    return features

def clipped_mean(x, sigma = 5):
    mean, _, _ = sigma_clipped_stats(x, sigma= sigma, maxiters=5)
    return mean

def clipped_median(x, sigma = 5):
    _, median, _ = sigma_clipped_stats(x, sigma= sigma, maxiters=5, cenfunc = "median")
    return median


def clipped_std(x, sigma = 5):
    _, _, std = sigma_clipped_stats(x, sigma= sigma, maxiters=5)
    return std


def get_above_Nsigma(mag, std, bins, binned_mean, binned_std, Nsigma = 3):
    is_above = np.zeros(len(mag), dtype = bool)
    bin_idx = np.digitize(mag, bins=bins) - 1 
    for i in range(len(binned_mean)):
        select = bin_idx == i
        if not np.any(select):
            continue
        if np.isnan(binned_mean[i]) or np.isnan(binned_std[i]):
            continue  
        threshold= binned_mean[i] + Nsigma*binned_std[i]
        is_above[select] = std[select] > threshold
    
    return is_above

def get_mean_std_bins(x, y, mean_func = "mean", clipping_sigma = 5,
                      bins = np.arange(14, 27, 0.33)):
    
    if clipping_sigma is None:
        std_statistic = "std"
        mean_statistic = mean_func
    else:
        std_statistic = lambda x : clipped_std(x, sigma = clipping_sigma)
        if mean_func == "median":
            mean_statistic = lambda x : clipped_median(x, sigma = clipping_sigma) 
        else:
            mean_statistic = lambda x : clipped_mean(x, sigma = clipping_sigma) 

    mean, edges, _ = binned_statistic(x, y, statistic = mean_statistic,
                                      bins = bins)
    std, edges, _ = binned_statistic(x,y, statistic = std_statistic,
                                  bins = bins)

    return mean, std, edges

def select_variable_with_std(table, band = "g",  Nsigma = 3, clipping_sigma = 5,
                             bins = np.arange(14, 27, 0.33),
                             mean_func = "mean"):
    
    mag = celib.flux_to_mag(table[f"{band}_psfFlux"])
    std =  table[f"std_{band}"].to_numpy()

    binned_mean, binned_std, _ = get_mean_std_bins(mag, std, mean_func = mean_func, 
                                       clipping_sigma = clipping_sigma, bins = bins)
    
    is_above = get_above_Nsigma(mag, std, bins, binned_mean, binned_std, Nsigma = Nsigma)
    return is_above

def get_completeness(labels, selected):
    true = labels == 3
    den = np.sum(true)
    if den == 0:
        return np.nan
    return np.sum(selected & true)/den

def get_precision(labels, selected):
    true = labels == 3
    den = np.sum(selected)
    if den == 0:
        return np.nan
    return np.sum(selected & true)/den

def get_N(labels):
    true = labels == 3
    return np.sum(true)



def get_completeness_precision_curve(df, band = "g", mag_cuts = np.arange(15, 28, 1),
                                     label_name = "label", selected_name = "std_selected"):
    
    mags = celib.flux_to_mag(df[f"{band}_psfFlux"])
    completeness, precision, N = [], [], []
    for mc in mag_cuts:
        logic = mags <= mc
        labels, selected = df[label_name][logic].to_numpy(), df[selected_name][logic].to_numpy()
        completeness.append(get_completeness(labels, selected))
        precision.append(get_precision(labels, selected))
        N.append(get_N(labels))
    return np.array(completeness), np.array(precision), np.array(N)


def get_mag(df, band):
    return celib.flux_to_mag(df[f"{band}_psfFlux"])