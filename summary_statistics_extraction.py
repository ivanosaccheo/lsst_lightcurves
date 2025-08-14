######This script is used to get only summary statistic 
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

import catalog_extraction_library as celib
import query_library as qlib
from joblib import delayed, Parallel

warnings.filterwarnings("ignore", category=RuntimeWarning)



def feature_extract(object_df, bands = ["u","g","r","i","z","y"]):
    """
    object_df = DataFrame with exptime_{band}'psfFlux_{band}', 'psfFluxErr_{band}'
    """
    if object_df.empty:
        return None

    data = {}
    for band in bands:
        mag_col = f"psfFlux_{band}"
        if mag_col in object_df.columns:
            mag_vals = object_df[mag_col].dropna()
            data[f"n_epochs_{band}"] = len(mag_vals)
            data[f"std_{band}"] = mag_vals.std()
            
        else:
            data[f"n_epochs_{band}"] = 0
            data[f"std_{band}"] = np.nan
            
    if 'objectId' in object_df.columns:
        data["objectId"] = object_df["objectId"].iloc[0]
    
    return pd.DataFrame([data])


def add_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s', "--snr_min", help="Minimum SNR required to keep forced photometry point. Default is 1", 
                        nargs='?', type = float, const = 5, default = 5)
    parser.add_argument('-j', "--jobs_number", help = "Number of jobs to launch for parallel computation. Default is 20",
                        nargs='?', type = int, const = 20, default = 20)
    parser.add_argument('-d', '--difference', help= "Use force photometry fluxes extracted from difference images",
                    action='store_true')
    args = parser.parse_args()
    
    return args


def get_metadata(args, patch, time_required):
    today = datetime.date.today().isoformat()
    metadata = {"date": str(today), 
                "time": str(time_required),
                "patch" : str(patch),
                "SNR_min" : str(args.snr_min),
                "Difference_Fluxes" : str(args.difference),
                "N_core" : str(args.jobs_number)}
    return metadata



def extraction_routine(args, patch, filename, savedir):
    forced_photometry_tables = celib.read_forced_photometry(patch = patch, SNR_minimum=args.snr_min, 
                                                            coadd= args.difference, 
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
    feature_tables = Parallel(n_jobs=args.jobs_number)(delayed(feature_extract)(
         object_df) for object_df in forced_photometry_tables)
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
    extra_dir  = "diff" if args.difference else "normal"
    savedir = os.path.join(main_save_dir, "summary_features", extra_dir, today)
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
