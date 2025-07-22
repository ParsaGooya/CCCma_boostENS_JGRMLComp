import numpy as np
import xarray as xr
import glob

def load_data(dir_in,
              file_in,
              single=True,
              rename=None):
    """
    """
    data = xr.open_mfdataset(f'{dir_in}/{file_in}.nc',
                             combine='nested',
                             concat_dim='year')
    
    if rename is not None:
        data = data.rename(rename)
    
    return data


def load_toy_data(list_data,
                  ddata_info,
                  box=np.array([[89.9,-89.9],  # [lat_min, lat_max] 
                                [0.01,360.]]),  # [lon_min, lon_max]
                  y0_test=2010,
                  y1_test=2014,                  
                  var='tas'):

    dict_data = {}

    for idata in list_data:

        dir_in  = ddata_info[idata]['dir_in']
        file_in = ddata_info[idata]['file_in']
        var_in  = ddata_info[idata]['var_in']
    
        print(idata)
        print("====")
        ds = load_data(dir_in,
                       file_in,
                       rename = {'lead_time'   : 'time',
                                 'ensembles'   : 'realization',
                                 var_in        : var})
        ds = ds.sel(lat=slice(box[0,0],
                              box[0,1])).sel(lon=slice(box[1,0],
                                                       box[1,1])).sel(year=slice(y0_test,
                                                                                 y1_test))
    
        if idata == 'train_sample':
            ds = ds.sel(realization=slice(1,20))
    
        dict_data[idata] = ds.squeeze().astype('float32')

    return dict_data



def load_clm_data(list_data,
                  ddata_info,
                  var='tas'):

    dict_data = {}

    for idata in list_data:

        dir_in  = ddata_info[idata]['dir_in']
        file_in = ddata_info[idata]['file_in']
        var_in  = ddata_info[idata]['var_in']
    
        print(idata)
        print("====")
        ds = load_data(dir_in,
                       file_in,
                       rename = {'lead_time'   : 'time',
                                 'ensembles'   : 'realization',
                                 var_in        : var})
    
        dict_data[idata] = ds.squeeze().astype('float32')

    if 'population' in list_data:
        dict_data['train_sample'] = dict_data['population'].sel(realization=[4,7,11,12,20]).astype('float32')


    return dict_data



