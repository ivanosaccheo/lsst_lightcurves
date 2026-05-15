from numba import njit
import numpy as np
import datetime
import argparse
import catalog_extraction_library as celib



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


@njit
def clip_lc(mag, num_sigma=3, max_iter=1000):

    lc_len = mag.shape[0]
    if lc_len < 3:
        return np.ones(lc_len, dtype=np.bool_)

    mad = np.median(np.abs(mag - np.median(mag)))
    sigma = mad / 0.6745

    if sigma <= 0:
        return np.ones(lc_len, dtype=np.bool_)

    med = mag.copy()

    for i in range(1, lc_len - 1):
        med[i] = np.median(mag[i - 1:i + 2])

    med[0] = np.median(np.array([mag[-1], mag[0], mag[1]]))
    med[-1] = np.median(np.array([mag[-2], mag[-1], mag[0]]))

    res = np.abs(mag - med)
    thresh = num_sigma * sigma

    for _ in range(max_iter):  #set clipping thresh hold if remove too much, raise bar until only remove 10%
        ratio = np.sum(res > thresh) / lc_len
        if ratio < 0.1:
            break
        thresh += 0.1 * sigma

    return res < thresh



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
    if flux_err_median <= 0 or not np.isfinite(flux_err_median):
        return None, None, None
    
    clean_mask = flux_err <= 5 * flux_err_median
    if np.sum(clean_mask)<min_Npoints:
        return None, None, None
    
    return mjd[clean_mask], flux[clean_mask], flux_err[clean_mask]




