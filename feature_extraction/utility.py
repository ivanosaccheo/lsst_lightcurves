import numpy as np 
import pandas as pd
import os
import glob 
import catalog_extraction_library as celib 
from astropy.stats import sigma_clipped_stats
from scipy.stats import binned_statistic,  median_abs_deviation, binned_statistic_2d
from scipy.interpolate import interp1d
import healsparse
import query_library as qlib
from astropy.coordinates import SkyCoord
from astropy import units as u


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


def get_features_table(path = "/data1/isaccheo/standard_features/2025-11-11",
                       change_nepochs = True):
    features = []
    for name in os.listdir(path):
        fname = os.path.join(path, name)
        features.append(pd.read_parquet(fname))
    if len(features)>1:
        features = pd.concat(features)
        try:
            features.columns = ['_'.join([feature, band]) if feature != 'objectId' else 'objectId' for feature, band in features.columns]
        except ValueError:
            pass
        if change_nepochs:
            for band in "ugrizy":
                select = np.isnan(features[f"Pvar_{band}"])
                features.loc[select, f"n_epochs_{band}"] = 0
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

def clipped_mad_std(x, sigma = 5):
    _, _, std = sigma_clipped_stats(x, stdfunc = "mad_std", sigma = sigma, maxiters=5)
    return std


def get_above_Nsigma(mag, std, bins, edges, binned_mean, binned_std, Nsigma = 3, 
                      interpolate = False, mag_lim = 15, std_min = 0.01):
    
    if interpolate:
        xpos = 0.5*(edges[:-1]+edges[1:])
        f = interp1d(xpos, binned_mean + Nsigma * binned_std, bounds_error=False, fill_value= np.nan)
        is_above = std > f(mag)
    else:
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
    is_above[mag <= mag_lim] = False
    is_above[std <= std_min] = False
    return is_above

def get_mean_std_bins(x, y, mean_func = "mean", clipping_sigma = 5,
                      bins = np.arange(14, 27, 0.33)):
    
    if clipping_sigma is None:
        std_statistic =  lambda x : median_abs_deviation(x, nan_policy ="omit") if mean_func == "median" else "std" 
        mean_statistic = mean_func
    
    else:
        
        if mean_func == "median":
            mean_statistic = lambda x : clipped_median(x, sigma = clipping_sigma) 
            std_statistic = lambda x : clipped_mad_std(x, sigma = clipping_sigma)
        else:
            mean_statistic = lambda x : clipped_mean(x, sigma = clipping_sigma)
            std_statistic = lambda x : clipped_std(x, sigma = clipping_sigma)

    mean, edges, _ = binned_statistic(x, y, statistic = mean_statistic,
                                      bins = bins)
    std, edges, _ = binned_statistic(x,y, statistic = std_statistic,
                                  bins = bins)
    if mean_func == "median":
        std = 1.4826*std ##MAD to sigma

    return mean, std, edges

def select_variable_with_std(table, band = "r",  Nsigma = 3, clipping_sigma = 5,
                             bins = np.arange(14, 27, 0.33),
                             mean_func = "mean", interpolate = False, mag_lim = 16, std_min = 0.01):
    
    mag = celib.flux_to_mag(table[f"{band}_psfFlux"])
    std =  table[f"std_{band}"].to_numpy()

    binned_mean, binned_std, edges = get_mean_std_bins(mag, std, mean_func = mean_func, 
                                       clipping_sigma = clipping_sigma, bins = bins)
    
    is_above = get_above_Nsigma(mag, std, bins, edges, binned_mean, binned_std, Nsigma = Nsigma,
                                interpolate = interpolate, mag_lim = mag_lim, std_min = std_min)
    return is_above



def get_Nsigma_line(table, xarray = np.linspace(14,28),
                    band = "r", 
                    Nsigma = 3, clipping_sigma = 5,
                    bins = np.arange(14, 27, 0.33),
                    mean_func = "mean", interpolate = False):
    mag = celib.flux_to_mag(table[f"{band}_psfFlux"])
    std =  table[f"std_{band}"].to_numpy()
    binned_mean, binned_std, edges = get_mean_std_bins(mag, std, mean_func = mean_func, 
                                       clipping_sigma = clipping_sigma, bins = bins)
    if interpolate:
        xpos = 0.5*(edges[:-1]+edges[1:])
        f = interp1d(xpos, binned_mean + Nsigma * binned_std, bounds_error=False, fill_value= np.nan)
        values = f(xarray)
    else:
        values = np.full(len(xarray), fill_value = np.nan)
        bin_idx = np.digitize(xarray, bins=bins) - 1 
        for i in range(len(binned_mean)):
            select = bin_idx == i
            if not np.any(select):
                continue
            if np.isnan(binned_mean[i]) or np.isnan(binned_std[i]):
                continue  
            values[select] =  binned_mean[i] + Nsigma*binned_std[i]
    return xarray, values
    

def get_completeness(labels, selected, target_label=3):
    tp = get_true_positives(labels, selected, target_label=target_label)
    N = get_N_positives(labels, target_label=target_label)
    return tp/N if N > 0 else np.nan

def get_precision(labels, selected, target_label = 3):
    tp = get_true_positives(labels, selected, target_label = target_label)
    den = np.sum(selected)
    return tp/den if den > 0 else np.nan

def get_N_positives(labels, target_label = 3):
    positives = labels == target_label
    return np.sum(positives)

def get_N_negatives(labels, target_label = 3):
    negatives = labels != target_label
    return np.sum(negatives)

def get_true_positives(labels, selected, target_label=3):
    true = labels == target_label
    return np.sum(selected & true)

def get_false_positives(labels, selected, target_label=3):
    false  = labels != target_label
    return np.sum(selected & false)


def get_completeness_precision_curve(df, parameter_name = "r", bins = np.arange(15, 28, 1),
                                     cumulative = False, target_label = 3,
                                     label_name = "label", selected_name = "std_selected"):

    if parameter_name in ("ugrizy"):
        values = celib.flux_to_mag(df[f"{parameter_name}_psfFlux"])
    else:
        values = df[parameter_name].to_numpy()
    completeness, precision, N, value_mean = [], [], [], []
    for bin_low, bin_high in zip(bins[:-1], bins[1:]):
        if cumulative:
            logic = values < bin_high
        else:
            logic = (values < bin_high) & (values>= bin_low)
        labels, selected = df[label_name][logic].to_numpy(), df[selected_name][logic].to_numpy()
        completeness.append(get_completeness(labels, selected, target_label = target_label))
        precision.append(get_precision(labels, selected, target_label = target_label))
        N.append(get_N_positives(labels, target_label = target_label))
        value_mean.append(np.nanmedian(values[logic]))

    return np.array(completeness), np.array(precision), np.array(N), np.array(value_mean)

def get_metrics_numbers(df, band = "r", mag_cuts = np.arange(15, 28, 1),
                        cumulative = False, target_label = 3,
                        label_name = "label", selected_name = "std_selected"):
    
    mags = celib.flux_to_mag(df[f"{band}_psfFlux"])
    tp, fp, N_positives, N_negatives = [], [], [], []
    for mlow, mhigh in zip(mag_cuts[:-1], mag_cuts[1:]):
        if cumulative:
            logic = mags < mhigh
        else:
            logic = (mags < mhigh) & (mags >= mlow)
        labels, selected = df[label_name][logic].to_numpy(), df[selected_name][logic].to_numpy()
        tp.append(get_true_positives(labels, selected, target_label=target_label))
        fp.append(get_false_positives(labels, selected, target_label=target_label))
        N_positives.append(get_N_positives(labels, target_label=target_label))
        N_negatives.append(get_N_negatives(labels, target_label=target_label))

    return np.array(tp), np.array(fp), np.array(N_positives), np.array(N_negatives)


def get_mag(df, band):
    return celib.flux_to_mag(df[f"{band}_psfFlux"])


def compute_distance(ra, dec):
    ra_array = ra[:, None]
    dec_array = dec[:, None]
    distance = (ra_array - ra_array.T)**2 + (dec_array - dec_array.T)**2
    return np.sqrt(distance)

def get_clustered(distance, N_min = 3, clustered_sep = 60):
    is_neighbour = distance <= clustered_sep/3600
    clustered = np.sum(is_neighbour,  axis = 1)>= N_min
    return clustered


def keep_non_flagged(table, band= "r", nepoch_min=10, mag_max =26,
                     extended_flag = True, psf_flag = True): 
    flux_min, _ = celib.mag_to_flux(mag_max, 0.2) 
    select = (table[f"n_epochs_{band}"] >= nepoch_min)  & (table[f"{band}_psfFlux"]>= flux_min) 
    if psf_flag:
        select = select & (table[f"{band}_psfFlux_flag"]==0)
    if extended_flag:
        select = select & (table[f"{band}_extendedness_flag"]==0)
    return select

def keep_non_edges(table, mag_min = 25.5):
    hsp_map = healsparse.HealSparseMap.read("../input/deepCoadd_psf_maglim_map_weighted_mean.fits", nside_coverage=32)
    ra, dec = table["RA"],table["DEC"]
    hsp_map_value = hsp_map.get_values_pos(ra, dec, lonlat=True)
    return hsp_map_value >= mag_min 
    
def mask_nearby_stars(table, band = "r", star_mag = [9, 11.5], sep_radius = [2,1]):

    query = f"""SELECT RA, DEC, [lsst-{band}_total]
               FROM Truth
               ORDER BY [lsst-{band}_total] DESC
               LIMIT 1000"""
    truth = qlib.query_agile(query)
    coordinates = SkyCoord(table["RA"],table["DEC"], unit = "degree")
    star_magnitudes = celib.flux_to_mag(truth[f"lsst-{band}_total"]*1000)
    if not isinstance(star_mag, list):
        star_mag = [star_mag]
    if not isinstance(sep_radius, list):
        sep_radius = [sep_radius]
    masked = np.zeros(len(table), dtype = "bool")
    for mag, radius in zip(star_mag, sep_radius):
        select = star_magnitudes <= mag 
        star_coordinates = SkyCoord(truth[select]["RA"],truth[select]["DEC"], unit = "degree")
        idx, sep2d, _ = coordinates.match_to_catalog_sky(star_coordinates)
        masked = np.logical_or(masked,sep2d < radius*u.arcmin)
    return masked


def get_binned_metric_2d(real, predicted, xvalues, yvalues, 
                         metric = "completeness",  xbins = 10, ybins = 10):
    
    real = real.astype(bool)
    predicted = predicted.astype(bool)

    tp = np.logical_and(real, predicted).astype(int)
    fn = np.logical_and(real,  ~predicted).astype(int)
    fp = np.logical_and(~real, predicted).astype(int)
    
    if metric != "count":
        tp_bin, xedges, yedges, _ = binned_statistic_2d(xvalues, yvalues, tp, statistic = "sum", bins=(xbins, ybins))
        fn_bin, _, _, _ = binned_statistic_2d(xvalues, yvalues, fn, statistic = "sum", bins=(xbins, ybins))
        fp_bin, _, _, _ = binned_statistic_2d(xvalues, yvalues, fp, statistic = "sum", bins=(xbins, ybins))

    if metric == "completeness":
        denom = tp_bin + fn_bin 
        statistic = np.where(denom > 0, tp_bin/denom, np.nan)

    elif metric == "precision":
        denom = tp_bin + fp_bin 
        statistic = np.where(denom > 0, tp_bin/denom, np.nan)

    elif metric == "count":
        statistic, xedges, yedges, _ = binned_statistic_2d(xvalues, yvalues, np.ones_like(xvalues), 
                                                           statistic = "count", bins=(xbins, ybins))
    
    elif metric == "count_true":
        statistic = tp_bin + fn_bin 
    
    elif metric == "count_predicted":
        statistic = tp_bin + fp_bin 

    else:
        raise ValueError(f"""'metric' must be one of: 'completeness', 'precision', 
                                                       'count', 'count_true', 'count_predicted'""")
    
    return statistic, xedges, yedges



def get_binned_metric(true, predicted, xvalues, metric = "completeness",  bins = 10):
    
    true = true.astype(bool)
    predicted = predicted.astype(bool)

    tp = np.logical_and(true, predicted).astype(int)
    fn = np.logical_and(true,  ~predicted).astype(int)
    fp = np.logical_and(~true, predicted).astype(int)
    
    if metric != "count":
        tp_bin, xedges, _ = binned_statistic(xvalues, tp, statistic = "sum", bins=bins)
        fn_bin, _, _ = binned_statistic(xvalues, fn, statistic = "sum", bins=bins)
        fp_bin, _, _ = binned_statistic(xvalues, fp, statistic = "sum", bins=bins)
    
    if metric == "completeness":
        denom = tp_bin + fn_bin 
        statistic = np.where(denom > 0, tp_bin/denom, np.nan)

    elif metric == "precision":
        denom = tp_bin + fp_bin 
        statistic = np.where(denom > 0, tp_bin/denom, np.nan)
    
    elif metric == "count":
        statistic, xedges, _ = binned_statistic(xvalues, np.ones_like(xvalues), statistic = "count", bins=bins)

    elif metric == "count_true":
        statistic = tp_bin + fn_bin 
    
    elif metric == "count_predicted":
        statistic = tp_bin + fp_bin 

    else:
        raise ValueError(f"""'metric' must be one of: 'completeness', 'precision', 
                                                       'count', 'count_true', 'count_predicted'""")
    
    return statistic, xedges


def update_sigma_df(table, band = "r",  clipping_sigma = 5,
                    bins = np.arange(14, 27, 0.33),
                    mean_func = "mean", interpolate = False):
    
    mag = celib.flux_to_mag(table[f"{band}_psfFlux"])
    std =  table[f"std_{band}"].to_numpy()

    binned_mean, _, edges = get_mean_std_bins(mag, std, mean_func = mean_func, 
                                       clipping_sigma = clipping_sigma, bins = bins)

    select = (table["label"] == 2) |  (table["label"] == 3) 

    phot_sigma2, intrins_sigma2 = get_photometric_and_intrinsic_variability(mag[select], std[select],
                                                                        bins=bins, edges=edges, 
                                                                        binned_mean =binned_mean, 
    
    
    
                                                                       interpolate = interpolate)
    new_sigma = np.sqrt(intrins_sigma2*(table.loc[select, "Z"]+1) + phot_sigma2)
    new_sigma = np.where(intrins_sigma2>0, new_sigma, std[select])
    new_df = table.copy()
    new_df.loc[select, f"std_{band}" ] = new_sigma
    return new_df

def get_photometric_and_intrinsic_variability(mag, std, bins, edges, binned_mean,
                              interpolate = False):
    if interpolate:
        xpos = 0.5*(edges[:-1]+edges[1:])
        f = interp1d(xpos, binned_mean, bounds_error=False, fill_value= np.nan)
        photometric_variability = f(mag)
    else:
        bin_idx = np.digitize(mag, bins=bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < len(binned_mean))
        photometric_variability = np.full(len(mag), np.nan)
        photometric_variability[valid] = binned_mean[bin_idx[valid]]
    
    intrinsic_variability = np.maximum(std**2-photometric_variability**2, 0)
 
    return photometric_variability**2, intrinsic_variability
     
