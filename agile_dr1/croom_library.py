
import pandas as pd
import numpy as np
import catalog_extraction_library as celib 
import copy


def quality_cut(data, i_band_limit = 22):
    bright = data["i"] <= i_band_limit
    good = (data["err_i"] < 0.2) & (data["err_u"] < 0.4) & (data["err_g"] < 0.13)
    good = good & (data["err_r"] < 0.13) & (data["err_z"] < 0.6)
    return good & bright
   
def general_cut(data):
    a = (data["u-g"] > -1)   & (data["u-g"] < 0.8)
    b = (data["g-r"] > -0.8) & (data["g-r"] < 0.0)
    c = (data["r-i"] > -0.6) & (data["r-i"] < -0.1)
    d = (data["i-z"] > -1)   & (data["i-z"] < -0.1)
    e = (data["g-i"] > -1.5) & (data["g-i"] < -0.3)
   
    reject = a & ((b & c & d) | e)
    return ~reject

def bright_cut(data):
    a = (data["u-g"]<0.8) & (data["g-r"]<0.6) & (data["r-i"]<0.6)
    b = (data["u-g"] > 0.6) & (data["g-i"]>0.2)
    c = (data["u-g"]>0.45) & (data["g-i"] >0.35) 
    d = (data["r_extendedness"] >= 0.99) & (data["u-g"]>0.2) & (data["g-r"]>0.25) & (data["r-i"]<0.3)
    e = (data["r_extendedness"] >= 0.99) & (data["u-g"]>0.45)
    accept = a & (~b) & (~c) & (~d) & (~e)
    return accept

def faint_cut(data):
    a = (data["u-g"]<0.8) & (data["g-r"]<0.5) & (data["r-i"]<0.6)
    b = (data["u-g"] > 0.5) & (data["g-i"]>0.15)
    c = (data["u-g"]>0.4) & (data["g-i"] >0.3) 
    d = (data["u-g"]>0.2) & (data["g-i"] >0.45) 
    e = (data["r_extendedness"] >= 0.99) & (data["g-r"]>0.3)
    accept = a & (~b) & (~c) & (~d) & (~e)
    return accept

def high_z_cut_1(data):
    a = data["u"] > 20.6
    b = data["u-g"] > 1.5
    c = data["g-r"] < 1.2
    d = data["r-i"] < 0.3 
    e = data["i-z"]> -1
    f = data["g-r"] > 0.44*data["u-g"]-0.76
    return a & b & c & d & e & f


def high_z_cut_2(data):
    b = (data["u-g"]>1.5) | (data["u"]>20.6)
    c=  data["g-r"] > 0.7
    d = (data["g-r"]>2.8) | (data["r-i"] < 0.44*data["g-r"]-0.558)
    e = (data["i-z"] > 0.25) & (data["i-z"] > -1)
    return b & c & d & e


def high_z_cut_3(data):
    a = data["u"]>21.5
    b = data["g"]>21.0
    c=  data["r-i"] > 0.6
    d=  data["i-z"] > -1
    e = data["i-z"] > 0.52 * data["r-i"] -0.762
    return a & b & c & d & e

def get_colors(df, catalog = 'mock', extra_info = ["ID", "Z", "label"]):
    bands = "ugrizy"
    if catalog.casefold() == "mock":
        cols = [f"lsst-{band}_total" for band in bands]
        factor = 1000 #mock is in microJy
    elif catalog.casefold() == "photometric":
        cols = [f"{band}_psfFlux" for band in bands]
        factor = 1
    data = pd.DataFrame()
    for col in extra_info:
        try:
            data[col] = df[col]
        except KeyError:
            pass

    for (col, band) in zip(cols, bands):
        data[band] = celib.flux_to_mag(df[col]*factor)
        data[f"err_{band}"] = celib.err_flux_to_err_mag(df[f"{col}Err"], df[col])
    for band1, band2 in zip(bands[:-1], bands[1:]):
        data[f"{band1}-{band2}"] = data[band1]-data[band2]
    data["g-i"] = data['g']-data['i']
    for band in 'ugrizy':
        data[f"{band}_extendedness"]  = df[f"{band}_extendedness"]
    return data


def get_completeness_mag_bins(mag_bins):
    croom = pd.read_csv("../input/Croom_09_completeness.dat", sep = "\s+")
    croom["bin"]= np.digitize(croom["gmag"], mag_bins)
    return croom.groupby(["bin", "redshift"]).mean().reset_index()


def apply_croom09(df, i_band_limit = 22, g_band_limit = 21.85):
     
    data = copy.deepcopy(df)
    quality_logic = quality_cut(data, i_band_limit=i_band_limit) & (data["g"]<=g_band_limit)
    
    data["is_croom_quality"] = quality_logic
    data["is_croom_agn"] = np.zeros(len(data), dtype=bool)
    
    data.loc[data["g"]<21.15, "is_croom_agn"] = general_cut(data[data["g"]<21.15]) & bright_cut(data[data["g"]<21.15])
    data.loc[data["g"]>=21.15, "is_croom_agn"] = general_cut(data[data["g"]>=21.15]) & faint_cut(data[data["g"]>=21.15])
    data["is_croom_highz"] = high_z_cut_1(data) | high_z_cut_2(data) | high_z_cut_3(data)
    
    data.loc[:, "is_croom_highz"] = data["is_croom_highz"] & (data["i"]<21) & quality_logic
    data.loc[:, "is_croom_agn"] = data.loc[:, "is_croom_agn"] & quality_logic
    
    return data