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
import query_library as qlib

@njit
def clip_lc(mag, num_sigma=3, max_iter = 1000):

    lc_len = mag.shape[0]
    sigma = 0.6745 * np.median(np.abs(mag - np.median(mag)))
    ## 3 point filtered median
    med = mag.copy()
    med[1:-1] = [np.median(mag[i - 1 : i + 2]) for i in range(1, lc_len - 1)]
    # need to deal with edges of median filter
    med[0] = np.median(np.array([mag[-1], mag[0], mag[1]]))
    med[-1] = np.median(np.array([mag[-2], mag[-1], mag[0]]))
    res = np.abs(mag - med)
    thresh = num_sigma * sigma #set clipping thresh hold
    # if remove too much, raise bar until only remove 10%
    for _ in range(max_iter):
        ratio = np.sum(res > thresh) / lc_len
        if ratio < 0.1:
            break
        thresh += 0.1

    return res < thresh

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

def get_band_lightcurve(object_df, band, min_Npoints=5, clip = True):
    # Drop rows with NaN in flux or flux error
    df = object_df.dropna(subset=[f'psfFlux_{band}', f'psfFluxErr_{band}'])
    if len(df) < min_Npoints:
        return None, None, None
    mjd = df[f'exptime_{band}'].to_numpy()
    mag = df[f'psfFlux_{band}'].to_numpy()
    mag_err = df[f'psfFluxErr_{band}'].to_numpy()
    if clip:
        idx = clip_lc(mag)
        mjd, mag, mag_err = mjd[idx] , mag[idx], mag_err[idx]
    if len(mjd)<min_Npoints:
        return None, None, None
    flux, flux_err = celib.mag_to_flux(mag, mag_err) ##Now in nanoJansky
    flux_err_median = np.median(flux_err)
    clean_mask = (flux_err - flux_err_median) < 5 * flux_err_median
    if np.sum(clean_mask)<min_Npoints:
        return None, None, None
    
    return mjd[clean_mask], flux[clean_mask], flux_err[clean_mask]
     



def dho_extract(object_df, bands, clip=True, min_Npoints = 5, n_iter=5, n_opt =25):
    """
    extract DHO features for all bands for one object
    Applied 3 point median filter on y and remove data with large error"""
    
    if object_df.shape[0] < min_Npoints:
        return np.full(len(bands)*6, np.nan)

    result_array = np.full(len(bands)*6, np.nan)  #each band has 0=a1_dho, 1=a2_dho, 2=b0_dho, 3=b1_dho
                                                   #4=log_ll, 5 = Npoints see eztao.carma.DHO_term (subclass of CARMA_term)

    for j, band in enumerate(bands):
        mjd, mag, mag_err = get_band_lightcurve(object_df, band, min_Npoints=min_Npoints, clip = clip)

        if mjd is None:
            continue
        Npoints = len(mag)
        result_array[j*6+5] = Npoints
       
        try:
            result_array[j*6:j*6+5] = fit_dho(mjd, mag, mag_err, n_iter=n_iter, n_opt = n_opt)
        except Exception:
            pass
    
    return result_array


def add_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s', "--snr_min", help="Minimum SNR required to keep forced photometry point. Default is 1", 
                        nargs='?', type = float, const = 1, default = 1)
    parser.add_argument('-n', "--nobs_min", help="Minimum number of observations in the lightcurve required to extract features. Default is 30", 
                        nargs='?', type = int, const = 30, default = 30)
    parser.add_argument('-j', "--jobs_number", help = "Number of jobs to launch for parallel computation. Default is 20",
                        nargs='?', type = int, const = 20, default = 20)
    parser.add_argument('-o', "--opt_number", help = "Number of optimizers to run. Default is 25",
                        nargs='?', type = int, const = 25, default = 25)
    parser.add_argument('-i', "--iter_number", help = "Number of iterations for which the fit is performed. The one providing the best likelihood is kept. Default is 5",
                        nargs='?', type = int, const = 5, default = 5)
    parser.add_argument('-d', '--difference', help= "Use force photometry fluxes extracted from difference images",
                    action='store_true')
    parser.add_argument('-v', '--variance', help= "Use errors computed from variance as a function of magnitudes",
                    action='store_true')
    args = parser.parse_args()
    
    return args

def get_metadata(args, time_required):
    today = datetime.datetime.today().date()
    metadata = {"date": str(today), 
                "time": str(time_required),
                "SNR_min" : str(args.snr_min),
                "N_obs_min" : str(args.nobs_min),
                "N_opt" : str(args.opt_number),
                "Difference_Fluxes" : str(args.difference),
                "Variance_Error" : str(args.variance),
                "N_iter" : str(args.iter_number),
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
        metadata = get_metadata(args, (toc-tic)/3600)
        feature_df = pa.Table.from_pandas(feature_df)
        feature_df = feature_df.replace_schema_metadata({**feature_df.schema.metadata, **metadata})
        pq.write_table(feature_df, os.path.join(savedir, filename)) 
    return None 

def main():
    today = datetime.date.today().isoformat()
    main_save_dir = "/data1/isaccheo"
    args = add_parser()
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






