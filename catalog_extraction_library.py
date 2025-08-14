import sys
import pandas as pd
import os
import numpy as np
from scipy.interpolate import interp1d
from astropy.table import Table
import copy
import query_library as qlib


def flux_to_mag(f):
    f = np.asarray(f)
    f = np.where(f>0, f, np.nan)
    mag = -2.5 * np.log10(f * 1e-9 / 3631)
    return mag

def err_flux_to_err_mag(err_f, f):
    f, err_f = np.asarray(f), np.asarray(err_f)
    f = np.where(f>0, f, np.nan)
    err_f = np.where(err_f>0, err_f, np.nan)
    return (2.5/np.log(10))*(err_f/f)

def split_bands(df):
    band_data = {}
    for band in "ugrizy":
        band_df = df[df["band"] == band].sort_values(by="expMidptMJD")
        if not band_df.empty:
            band_data[f"exptime_{band}"] = band_df["expMidptMJD"].reset_index(drop=True)
            band_data[f"psfFlux_{band}"] = band_df["psfFlux"].reset_index(drop=True)
            band_data[f"psfFluxErr_{band}"] = band_df["psfFluxErr"].reset_index(drop=True)
        else:
            band_data[f"exptime_{band}"] = pd.Series(dtype="float64")
            band_data[f"psfFlux_{band}"] = pd.Series(dtype="float64")
            band_data[f"psfFluxErr_{band}"] = pd.Series(dtype="float64")

    result_df = pd.concat(band_data, axis=1)
    result_df["objectId"] = df["objectId"].iloc[0]

    for band in "ugrizy":
        coadd_col = f"{band}_coadd"
        if coadd_col in df.columns:
            result_df[f"coadd_{band}"] = df[coadd_col].iloc[0]

    return result_df


def add_coadd_flux_to_difference(df, SNR_minimum = 5):
    new_df = df.copy()
    for band in "ugrizy":
        coadd_col = f"coadd_{band}"
        if coadd_col in new_df.columns:
            new_df[f"psfFlux_{band}"] = new_df[f"psfFlux_{band}"]+new_df[coadd_col]
            select = new_df[f"psfFlux_{band}"] < SNR_minimum * new_df[f"psfFluxErr_{band}"]
            new_df.loc[select, f"psfFlux_{band}"] = np.nan
    return new_df

 
def convert_to_mag(df):
    """converts fluxes to magnitudes for all bands"""
    new_df = df.copy()
    for band in "ugrizy":
        new_df[f"exptime_{band}"] = new_df[f"exptime_{band}"]-new_df[f"exptime_{band}"].iloc[0]
        new_df[f"psfFluxErr_{band}"] = err_flux_to_err_mag(new_df[f"psfFluxErr_{band}"],new_df[f"psfFlux_{band}"])
        new_df[f"psfFlux_{band}"] = flux_to_mag(new_df[f"psfFlux_{band}"])
        coadd_col = f"coadd_{band}"
        if coadd_col in new_df.columns:
            new_df[coadd_col] = flux_to_mag(new_df[coadd_col])

    return new_df


def read_forced_photometry(patch = 23, SNR_minimum = 5, coadd = False, difference_flux = False):
    
    forced_df = qlib.query_force_photometry(patch = patch, snr = SNR_minimum, coadd = coadd, 
                                            difference_flux = difference_flux)
    
    fdfs = [group_df for _, group_df in forced_df.groupby('objectId')] #forced table is splitted in many tables according to objectId
    return fdfs 


def mag_to_flux(mag,mag_err, luptitudes = False, band = None):
    """magnitudes to nJy,
    Lupton magnitudes (luptitudes) are SDSS magnitude system"""
    if luptitudes and (band in "ugriz"):
        b = dict(zip('ugriz', np.array([1.4, 0.9, 1.2, 1.8, 7.4])*1e-10)) # asinh mag softening params
        corr = dict(zip('ugriz', [-0.04, 0, 0, 0, 0.02]))  # sdss to AB zero point correction
        # ratio between AB zeropoint flux and SDSS zeropoint flux in u and z bands
        # f_{0_AB}/f_{0_SDSS}
        ab2sdss_zp = np.ones(5)
        ab2sdss_zp[0] = b['u']*2*np.sinh(-(corr['u']*np.log(10)/2.5)-np.log(b['u']))
        ab2sdss_zp[4] = b['z']*2*np.sinh(-(corr['z']*np.log(10)/2.5)-np.log(b['z']))
        ab2sdss_zp = dict(zip('ugriz', ab2sdss_zp))
        a = 2.5/np.log(10)
        # convert from sdss mag to flux
        flux = np.sinh(mag*np.log(10)/(-2.5) - 
                   np.log(b[f'{band}']))*2*b[f'{band}']*1e9*3631/ab2sdss_zp[f'{band}']
        flux_err = np.abs((flux*mag_err/a)/np.tanh(-mag/a - np.log(b[f'{band}'])))
    
    else:
        flux = 10**(-(mag/2.5))*1e9*3631
        flux_err = mag_err*flux*np.log(10)/2.5
    return flux, flux_err


def read_metadata(filename):
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(filename)
    metadata = parquet_file.schema_arrow.metadata
    metadata = {k.decode('utf-8'): v.decode('utf-8') for k, v in metadata.items()}
    return metadata


def get_latest_file(matching_type, directory = "../output"):
    import re
    import datetime
    pattern = re.compile(rf"{re.escape(matching_type)}.*_(\d{{4}}-\d{{2}}-\d{{2}})\.parq")
    dated_files = {}
    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            date_str = match.group(1)
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                dated_files[filename] = date_obj
            except ValueError:
                continue
    return max(dated_files, key=dated_files.get)



def get_observational_errors_function(band, difference_flux = False, path = "../output"):
    if difference_flux:
        fname = "observed_variance_difference.dat"
    else:
        fname = "observed_variance.dat"
    observed_variance = pd.read_csv(os.path.join(path,fname), sep =" ")
    select = np.isfinite(observed_variance[f"{band}_error"])
    f = interp1d(observed_variance.loc[select, "magnitude"], observed_variance.loc[select,f"{band}_error"], 
                 fill_value = "extrapolate")
    return f


def get_observational_errors_dict(difference_flux = False):
    dizionario= {}
    for band in "ugrizy":
        dizionario[band] = get_observational_errors_function(band, difference_flux=difference_flux)
    return dizionario


def replace_errors_with_variance(object_df, dizionario):
    new_df = object_df.copy()
    for band in "ugrizy":
        try:
            mag = new_df[f"coadd_{band}"].iloc[0]
            new_df[f'psfFluxErr_{band}'] = dizionario[band](mag)
        except KeyError:
            pass
    return new_df
