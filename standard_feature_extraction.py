######This script is used to extract normal variability features, i.e. not involving gaussian processs

import os
import copy
import time
import datetime
import argparse
import warnings

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from numba import njit
import catalog_extraction_library as celib
import query_library as qlib
from joblib import delayed, Parallel


import dask.multiprocessing
import dask.local
from cesium import features, featurize

from cesium_funcs import *

warnings.filterwarnings("ignore", category=RuntimeWarning)


@njit
def sigma_clip(mag, magerr, num_sigma=5):
    """Clip light curve based on deviations from median"""
    med = np.median(mag)
    sigma = 0.6745 * np.median(np.abs(mag - med))
    res = mag - med
    return res - magerr < num_sigma*sigma


def get_band_lightcurve(object_df, band, min_Npoints=5):
    # Drop rows with NaN in flux or flux error
    object_df_temp = object_df.dropna(subset=[f'psfFlux_{band}', f'psfFluxErr_{band}'])
    
    default = np.full(min_Npoints, np.nan)
    
    if len(object_df_temp) >= min_Npoints:
        flux = object_df_temp[f'psfFlux_{band}'].to_numpy()
        flux_err = object_df_temp[f'psfFluxErr_{band}'].to_numpy()
        idx = sigma_clip(flux, flux_err, num_sigma=5)  

        if np.sum(idx) >= min_Npoints:
            mjd = object_df_temp[f'exptime_{band}'].to_numpy()[idx]  
            mag = flux[idx]
            mag_err = flux_err[idx]
            return mjd, mag, mag_err

    return default, default, default


def to_cesium(object_df, min_Npoints = 5):
    """Put lc into the format cesium needed"""
    ts, xs, es = [], [], []
    for band in 'ugrizy':
        mjd, mag, mag_err = get_band_lightcurve(object_df, band, min_Npoints=min_Npoints)
        ts.append(mjd)
        xs.append(mag)
        es.append(mag_err)            
        
    return ts, xs, es

def feature_extract(object_df, features, user_funcs, min_Npoints = 5):
    
    if object_df.shape[0] < min_Npoints:
        return None
    
    # into the form ceisum needs
    mjds, mags, mag_errs = to_cesium(object_df, min_Npoints = min_Npoints)
    # extract feats
    f2use = features+[feat for feat in user_funcs.keys() if feat not in features]
    fset = featurize.featurize_time_series(times=mjds,
                                           values=mags,
                                           errors=mag_errs,
                                           features_to_use=f2use,
                                           custom_functions=user_funcs,
                                           scheduler=dask.local.get_sync,
                                           raise_exceptions=False)

    ### reshape data to make channels under column
    fset.columns = fset.columns.droplevel(1)
    midx = pd.MultiIndex.from_product([fset.columns, ['u', 'g', 'r', 'i', 'z', 'y']])
    fset_flat = fset.values.flatten(order='F')
    new_df = pd.DataFrame(data=fset_flat).T
    new_df.columns = midx
    new_df["objectId"] = object_df["objectId"].iloc[0]

    return new_df

def get_feature_list():
    flux_feats = [f'flux_percentile_ratio_mid{r}' for r in [20, 35, 50, 65, 80]] + \
                ['percent_difference_flux_percentile']
    
    #1. general feasture (computed using mag)
    gen_feats = [feat for feat in features.GENERAL_FEATS 
                 if feat not in flux_feats + ['peroid_fast']]
    
    #2. Cadence features
    cad_feats = ['n_epochs', 'total_time']
    
    #3. LS/periodic features
    ls_feats1 = [f'freq{i}_amplitude{j}' for i in [1,2,3] for j in [1, 2, 3, 4]] + \
                [f'freq{i}_rel_phase{j}' for i in [1,2,3] for j in [2, 3, 4]]
    ls_feats2 = [f'freq{i}_freq' for i in [1,2,3]]
    ls_feats3 = ['freq1_signif', 'freq_signif_ratio_21', 'freq_signif_ratio_31',
                'freq_varrat', 'freq_y_offset', 'linear_trend']
    ls_feats = ls_feats1 + ls_feats2 + ls_feats3
    ls_feats.sort()
    
    # all from cesium built-in
    feats_ls = flux_feats + cad_feats + gen_feats + ls_feats
    #feats_ls = cad_feats 
    ## ------------------------------------------
    # cesium custom funcs to pass in
    user_funcs = {'chi2_per_dof':chi2_per_dof, 'mean_variance':mean_variance,
                "standard_deviation" : standard_deviation,
                "mean_error":mean_error, 
                "pair_slope_trend":pair_slope_trend, 'small_kurtosis':small_kurtosis,
                'excess_var':excess_var, 'normed_evar':normed_evar, 'rcs':rcs,
                'von_N_ratio': von_N_ratio, 'weighted_average': weighted_average,
                'min_dt':min_dt, "Pvar" : Pvar}

    return feats_ls, user_funcs



def add_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s', "--snr_min", help="Minimum SNR required to keep forced photometry point. Default is 1", 
                        nargs='?', type = float, const = 5, default = 5)
    parser.add_argument('-n', "--nobs_min", help="Minimum number of observations in the lightcurve required to extract features. Default is 5", 
                        nargs='?', type = int, const = 5, default = 5)
    parser.add_argument('-j', "--jobs_number", help = "Number of jobs to launch for parallel computation. Default is 20",
                        nargs='?', type = int, const = 20, default = 20)
    parser.add_argument('-d', '--difference', help= "Use force photometry fluxes extracted from difference images",
                    action='store_true')
    parser.add_argument('-v', '--variance', help= "Use errors computed from variance as a function of magnitudes",
                    action='store_true')
    args = parser.parse_args()
    
    return args


def get_metadata(args, patch, time_required):
    today = datetime.date.today().isoformat()
    metadata = {"date": str(today), 
                "time": str(time_required),
                "patch" : str(patch),
                "SNR_min" : str(args.snr_min),
                "N_obs_min" : str(args.nobs_min),
                "Difference_Fluxes" : str(args.difference),
                "Variance_Error" : str(args.variance),
                "N_core" : str(args.jobs_number)}
    return metadata



def extraction_routine(args, patch, filename, savedir):
    forced_photometry_tables = celib.read_forced_photometry(patch = patch, SNR_minimum=args.snr_min, 
                                                            coadd= (args.variance or args.difference), 
                                                                    difference_flux=args.difference)
    
    tic = time.perf_counter()
    forced_photometry_tables = Parallel(n_jobs=args.jobs_number)(delayed(celib.split_bands)(object_df)
                                                                 for object_df in forced_photometry_tables)
    toc = time.perf_counter()
    print(f"Bands separated in individual columns in {toc-tic} seconds")

    if args.difference:
        tic = time.perf_counter()
        forced_photometry_tables = Parallel(n_jobs = args.jobs_number)(delayed(celib.add_coadd_flux_to_difference)(object_df, SNR_minimum=args.snr_min)  
                                                                   for object_df in forced_photometry_tables)
        toc = time.perf_counter()
        print(f"Added Object fluxes to difference fluxes  in {toc-tic} seconds")


    tic = time.perf_counter()
    forced_photometry_tables = Parallel(n_jobs = args.jobs_number)(delayed(celib.convert_to_mag)(object_df)  
                                                                   for object_df in forced_photometry_tables)
    toc = time.perf_counter()
    print(f"Tables transformed to magnitudes and in {toc-tic} seconds")

    if args.variance: 
        tic = time.perf_counter()
        error_functions_dict = celib.get_observational_errors_dict(difference_flux = args.difference)
        forced_photometry_tables = Parallel(n_jobs = args.jobs_number)(delayed(celib.replace_errors_with_variance)(object_df, 
                                                                    error_functions_dict)  
                                                                    for object_df in forced_photometry_tables)
        toc = time.perf_counter()
        print(f"Replaced errors with Variance based ones in {toc-tic} seconds")
       
    feats_ls, user_funcs = get_feature_list()
    
    tic = time.perf_counter()
    feature_tables = Parallel(n_jobs=args.jobs_number)(delayed(feature_extract)(
         object_df, feats_ls, user_funcs, min_Npoints = args.nobs_min) for object_df in forced_photometry_tables)
    toc = time.perf_counter()

    print(f"Computed features in {(toc-tic)/3600} hours")
    #remove none; combine returned dfs into one and round to 8 decimals
    feature_tables_clean = [feature_table for feature_table in feature_tables if feature_table is not None]
    feature_df = pd.concat(feature_tables_clean).round(8)

    metadata = get_metadata(args=args, patch = patch, time_required = (toc-tic)/3600 )
    feature_df = pa.Table.from_pandas(feature_df)
    feature_df = feature_df.replace_schema_metadata({**feature_df.schema.metadata, **metadata})
    pq.write_table(feature_df, os.path.join(savedir, filename))

def main():
    today = datetime.date.today().isoformat()
    main_save_dir = "/data1/isaccheo"
    args = add_parser()
    if args.variance:
        savedir = os.path.join(main_save_dir, "standard_features", f"variance_{today}")
    else:
        savedir = os.path.join(main_save_dir, "standard_features", today)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    
    
    available_patches = qlib.query_available_patches()
    for patch in available_patches:
        filename = f"patch_{patch}"
        tic = time.perf_counter()
        extraction_routine(args, patch, filename, savedir)
        toc = time.perf_counter()
        print(f"Finished patch = {patch} in {(toc-tic)/3600} hours")
        

if __name__== "__main__":
    main()
