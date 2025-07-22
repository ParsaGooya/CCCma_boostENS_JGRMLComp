from tqdm import tqdm
import os
import yaml 
import gc
from data_load      import load_toy_data
from util_analysis  import get_variance, power
import numpy as np


filename = '/home/rpg002/boostEns_diag_tas/figures_paper/config.yaml'
with open(filename) as f:
    dict_yaml = yaml.load(f, Loader=yaml.loader.SafeLoader)

var        = dict_yaml['variable'][0]
dir_source = dict_yaml['dir_source']
dict_toy   = dict_yaml['data_toy_info']
list_data  = dict_yaml['list_data_raw'] + dict_yaml['list_data_toy']

dict_data = load_toy_data(list_data,
                          dict_toy)


import random
dict_data['toy_cVAE2'] = dict_data['toy_cVAE2'].isel(realization = random.sample(range(0, 10000), 2000))


time_to_show = 9
year_to_show = 2012
idata = 'population'



print(idata)
ds = dict_data[idata].sel(year=year_to_show,
                            time=time_to_show).load()
data_pwr = []
data_wav = []
for ii in tqdm(ds.realization.values):
    (spectrum,
        wavelengths_km) = power(ds[var].sel(realization=ii).sel(lon=ds.lon.values[:-1]))
    data_pwr.append(np.log2(spectrum))
    data_wav.append(wavelengths_km)
    

np.save(f'/home/rpg002/boostEns_diag_tas/power_data/{idata}_power.npy', np.array(data_pwr))
np.save(f'/home/rpg002/boostEns_diag_tas/power_data/{idata}_wavelength.npy', np.array(data_wav))
