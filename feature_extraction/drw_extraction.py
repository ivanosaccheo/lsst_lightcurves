import os
import time
import datetime
import argparse

import pandas as pd
import numpy as np
from numba import njit
from joblib import delayed, Parallel
import pyarrow as pa
import pyarrow.parquet as pq


from eztao.carma import DRW_term
from eztao.ts import drw_fit
from celerite import GP


import catalog_extraction_library as celib
import shared_functions
import query_library as qlib



def fit_drw(t, y, yerr, n_opt = 25, n_iter = 5):
    log_ll = -np.inf
    best = np.full(2, -1)
    for i in range(n_iter):
        best_cand = drw_fit(t, y, yerr, n_opt=n_opt, scipy_opt_kwargs={'tol': 1e-14}) #best_cand[0] = Amplitude, best_cand[0] = Tau
        # Only used to get log likelihood for best
        kernel = DRW_term(np.log(best_cand[0]), np.log(best_cand[1]))
        gp = GP(kernel, mean = np.median(y))
        gp.compute(t, yerr)
        cand_ll = gp.log_likelihood(y)
        # compare fit against the other
        if cand_ll > log_ll:
            log_ll = cand_ll
            best = best_cand
    return np.hstack([best, log_ll])



def drw_extract(object_df, bands, clip=True, min_Npoints = 5, n_opt = 25, n_iter = 5):
    """
    extract DRW features for all bands for one object
    Applied 3 point median filter on y and remove data with large error"""

    n_features_per_band = 4 #each band has 0=ampl, 1=tau, 2=log_ll, 3 = N_points
    result_array = np.full(len(bands) * n_features_per_band, np.nan)

    if object_df.shape[0] < min_Npoints:
        return result_array

    for j, band in enumerate(bands):
        start = j * n_features_per_band
        mjd, flux, flux_err = shared_functions.get_band_lightcurve(object_df, band, 
                                                                 min_Npoints=min_Npoints, 
                                                                clip = clip)
        if mjd is None:
            continue
        N_points = len(flux)
        result_array[start+3] = N_points
       
        ampl, tau, log_ll = fit_drw(mjd, flux, flux_err, n_iter=n_iter, n_opt = n_opt)
        result_array[start:start+3] = ampl, tau, log_ll

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
    feature_arrays = Parallel(n_jobs=args.jobs_number)(delayed(drw_extract)(
         object_df, 'ugrizy', clip = True, min_Npoints = args.nobs_min, n_iter=args.iter_number, n_opt=args.opt_number) 
         for object_df in forced_photometry_tables)
    toc = time.perf_counter()
    print(f"Computed DRW features in {(toc-tic)/3600} hours")

    objectId= [object_df["objectId"].iloc[0] for object_df in forced_photometry_tables]

    columns_lists= [[f"DRW_amplitude_{i}", f"DRW_tau_{i}", f"DRW_log_ll_{i}", f"DRW_Npoints_{i}"] 
                    for i in ['u','g','r','i','z','y']]
    columns = [i for lista in columns_lists for  i in lista]
    
    feature_df = pd.DataFrame()
    feature_df["objectId"] = objectId
    if len(feature_arrays)>0:
        feature_df[columns] = np.vstack(feature_arrays)
        metadata = shared_functions.get_metadata(args, (toc-tic)/3600)
        feature_df = pa.Table.from_pandas(feature_df)
        feature_df = feature_df.replace_schema_metadata({**feature_df.schema.metadata, **metadata})
        pq.write_table(feature_df, os.path.join(savedir, filename)) 
  

def main():
    today = datetime.date.today().isoformat()
    main_save_dir = "/data1/isaccheo"
    args = shared_functions.add_parser()
    if args.variance:
        savedir = os.path.join(main_save_dir, "drw_features", f"variance_{today}")
    else:
        savedir = os.path.join(main_save_dir, "drw_features", today)
    
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

