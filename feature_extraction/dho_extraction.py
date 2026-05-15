import os
import copy
import time
import datetime
import argparse 

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from numba import njit
from joblib import delayed, Parallel

from eztao.carma import  DHO_term, CARMA_term
from eztao.ts import  dho_fit
from eztao.ts.carma_fit import neg_param_ll
from celerite import GP

import catalog_extraction_library as celib
import shared_functions
import query_library as qlib


@njit
def sigma_clip(mag, magerr, num_sigma=5):
    """Clip light curve based on deviations from median"""
    med = np.median(mag)
    sigma = 0.6745 * np.median(np.abs(mag - med))
    res = mag - med
    return res - magerr < num_sigma*sigma


def neg_lp_dho(params, y, gp):
    """custom negative log probability function for DHO."""
    log10_tau_perturb = (params[-1] - params[-2])/np.log(10)
    if (-3 <= log10_tau_perturb) and (log10_tau_perturb) <= 5:
        prior = 0
    else:
        prior = - (np.abs(log10_tau_perturb - 1) - 4)
    
    return -prior + neg_param_ll(params, y, gp) 

def fit_dho(t, y, yerr, n_iter = 5, n_opt = 25):
    log_ll = -np.inf
    best = np.full(4, -1)
    for i in range(n_iter):
        best_cand = dho_fit(t, y, yerr, n_opt=n_opt, neg_lp_func=neg_lp_dho, 
                                    scipy_opt_kwargs={'tol': 1e-12})
        # get log likelihood for best
        kernel = CARMA_term(np.log(best_cand[:2]), np.log(best_cand[2:]))
        gp = GP(kernel, mean = np.median(y))
        gp.compute(t, yerr)
        cand_ll = gp.log_likelihood(y)
        if cand_ll > log_ll:
            log_ll = cand_ll
            best = best_cand
    return np.hstack([best, log_ll])

    

def dho_extract(object_df, bands, clip=True, min_Npoints = 5, n_iter=5, n_opt =25):
    """
    extract DHO features for all bands for one object
    Applied 3 point median filter on y and remove data with large error"""
    n_features_per_band = 6
    result_array = np.full(len(bands) * n_features_per_band, np.nan)
    #each band has 0=a1_dho, 1=a2_dho, 2=b0_dho, 3=b1_dho
    #4=log_ll, 5 = N_points see eztao.carma.DHO_term (subclass of CARMA_term)
    
    if object_df.shape[0] < min_Npoints:
        return np.full(len(bands)*6, np.nan)

    for j, band in enumerate(bands):
        start = j * n_features_per_band
        mjd, flux, flux_err = shared_functions.get_band_lightcurve(object_df, band, 
                                                                 min_Npoints=min_Npoints, 
                                                                 clip = clip)

        if mjd is None:
            continue
        N_points = len(flux)
        result_array[start+5] = N_points
        result_array[start:start+5] = fit_dho(mjd, flux, flux_err, n_iter=n_iter, n_opt = n_opt)

    
    return result_array




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

    tic = time.perf_counter()
    feature_arrays = Parallel(n_jobs=args.jobs_number)(delayed(dho_extract)(
         object_df, 'ugrizy', clip = True, min_Npoints = args.nobs_min, n_iter=args.iter_number, n_opt=args.opt_number) 
         for object_df in forced_photometry_tables)
    toc = time.perf_counter()
    print(f"Computed DHO features in {(toc-tic)/3600} hours")

    objectId= [object_df["objectId"].iloc[0] for object_df in forced_photometry_tables]

    columns_lists= [[f"DHO_a1_{i}", f"DHO_a2_{i}", f"DHO_b0_{i}", f"DHO_b1_{i}",
                     f"DHO_log_ll_{i}", f"DHO_Npoints_{i}"] for i in ['u','g','r','i','z','y']]
    columns = [i for lista in columns_lists for  i in lista]
    
    feature_df = pd.DataFrame()
    feature_df["objectId"] = objectId
    if len(feature_arrays)>1:
        feature_df[columns] = np.vstack(feature_arrays)
        metadata = shared_functions.get_metadata(args, (toc-tic)/3600)
        feature_df = pa.Table.from_pandas(feature_df)
        feature_df = feature_df.replace_schema_metadata({**feature_df.schema.metadata, **metadata})
        pq.write_table(feature_df, os.path.join(savedir, filename)) 
    return None 

def main():
    today = datetime.date.today().isoformat()
    main_save_dir = "/data1/isaccheo"
    args = shared_functions.add_parser()
    if args.variance:
        savedir = os.path.join(main_save_dir, "dho_features", f"variance_{today}")
    else:
        savedir = os.path.join(main_save_dir, "dho_features", today)
    
    os.makedirs(savedir, exist_ok=True)
    
    available_patches = qlib.query_available_patches()
    for patch in available_patches:
        filename = f"patch_{patch}"
        tic = time.perf_counter()
        extraction_routine(args, patch, filename, savedir)
        toc = time.perf_counter()
        print(f"Finished patch = {patch} in {(toc-tic)/3600} hours")
        
    



if __name__== "__main__":
    main()






