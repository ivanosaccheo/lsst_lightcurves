import pandas as pd
import os
import numpy as np
from scipy.stats import binned_statistic
import query_library as qlib
import catalog_extraction_library as celib 

only_extended = False
feature_dir = "/data1/isaccheo/standard_features/2025-07-20"
savepath = "../output"
savename = "observed_variance_difference.dat"


fnames =  os.listdir(feature_dir)
catalog = []
for fname in fnames:
    df = pd.read_parquet(os.path.join(feature_dir, fname))
    df.columns = ['_'.join([feature, band]) if feature != 'objectId' else 'objectId' for feature, band in df.columns]
    catalog.append(df)
catalog = pd.concat(catalog, axis = 0)


coadd = qlib.query_coadd_photometry(keep_psf = True, keep_cmodel=False,)

catalog = catalog.merge(right=coadd, left_on="objectId", right_on = "objectId")

variance_df = pd.DataFrame()
bins = np.arange(14, 27, 0.33)
for band in "ugrizy":
    if only_extended:
        select = catalog[f"{band}_extendedness"] == 0
    else:
        select = np.ones(len(catalog), dtype = bool)
    stats, edges, _ = binned_statistic(x = celib.flux_to_mag(catalog.loc[select, f"{band}_psfFlux" ]),
                                    values = catalog[f"std_{band}"], 
                                    statistic = lambda y: np.nanmedian(y),
                                    bins = bins)
    variance_df[f"{band}_error"] = stats
variance_df["magnitude"] = 0.5*(edges[:-1]+edges[1:])
variance_df.to_csv(os.path.join(savepath, savename), sep =" ", index = False)