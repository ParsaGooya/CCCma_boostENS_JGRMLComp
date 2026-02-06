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
                             concat_dim='year',
                             coords='minimal')
    
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




def load_hist_data(list_data,
                  ddata_info,
                  var='tas',
                  train_ensemble_size = 10):

    dict_data = {}

    if 'population' in list_data:
        list_data.append( 'population_extention')

    for idata in list_data:

        dir_in  = ddata_info[idata]['dir_in']
        file_in = ddata_info[idata]['file_in']
        var_in  = ddata_info[idata]['var_in']
        unit_change = ddata_info[idata]['unit_change']
    
        print(idata)
        print("====")
        ds = load_data(dir_in,
                       file_in,
                       rename = {'lead_time'   : 'time',
                                 'ensembles'   : 'realization',
                                 var_in        : var})
    
        dict_data[idata] = ds.squeeze().astype('float32') * unit_change

    

    if 'population' in list_data:
        dict_data['population'] = xr.concat([dict_data['population'], dict_data['population_extention']], dim = 'year')
        dict_data['train_sample'] = dict_data['population'].isel(realization= np.arange(0,train_ensemble_size)).astype('float32')
        del dict_data['population_extention'] 

    return dict_data

def load_obs_data(list_data,
                  ddata_info,
                  var='tas',
                  train_ensemble_size = 1):

    dict_data = {}



    for idata in list_data:

        dir_in  = ddata_info[idata]['dir_in']
        file_in = ddata_info[idata]['file_in']
        var_in  = ddata_info[idata]['var_in']
        unit_change = ddata_info[idata]['unit_change']
    
        print(idata)
        print("====")
        ds = load_data(dir_in,
                       file_in,
                       rename = {'lead_time'   : 'time',
                                 'ensembles'   : 'realization',
                                 var_in        : var})

        dict_data[idata] = ds.astype('float32') * unit_change

        try:
            dict_data[idata] = dict_data[idata].squeeze('channels')
        except:
            pass

    dict_data['train_sample'] = dict_data['population'].isel(realization= np.arange(0,train_ensemble_size)).astype('float32')

    return dict_data
