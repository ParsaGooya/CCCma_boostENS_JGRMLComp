import numpy as np
import xarray as xr
import pyshtools as pysh


def enso_extremes(ds,
                  ds_enso,
                  ds_enso_ref,
                  quantile=None):
    
    if quantile is None:
        ds_nino_extreme = ds.where(ds_enso > ds_enso_ref.max())
        ds_nina_extreme = ds.where(ds_enso < ds_enso_ref.min())
    else: 
        ds_nino_extreme = ds.where(ds_enso >  np.quantile(ds_enso,quantile))
        ds_nina_extreme = ds.where(ds_enso < np.quantile(ds_enso,1-quantile))

    return ds_nino_extreme, ds_nina_extreme


def get_variance(ds,
                 dim_var='realization', uncentered = False):
    # return (ds - ds.mean(dim_var)).var(dim=dim_var)
    if uncentered:
        return (ds**2).mean(dim_var) 
    else:
        return ds.var(dim=dim_var)


import pyshtools as pysh
def power(ds,
          anomalies=False,
          Lmax = None):

    R_earth_km = 6371
    
    if anomalies:
        ds_anoms = ds - ds.mean(['lat','lon'])
    else:
        ds_anoms = ds.copy()

    grid = pysh.SHGrid.from_array(ds_anoms)
    clm  = grid.expand()
    power = clm .spectrum()

    if Lmax is None:
        Lmax = power.size - 1
    elif Lmax > power.size - 1:
        grid = grid.expand().expand_grid(lmax=Lmax)
        clm  = grid.expand()
        power = clm .spectrum()
    else:
        power = power[:Lmax]


    degrees = np.arange(Lmax + 1)


    wavelength_km = np.full_like(degrees, np.nan, dtype=float)
    wavelength_km[1:] = 2 * np.pi * R_earth_km / degrees[1:]


    return power, wavelength_km



from sklearn.decomposition import PCA
def doPCA(ds, n_components=2,  fitted_pca = None, return_explained_variance = True):
    assert n_components <= len(ds.mu)
    ds_ =ds.isel(mu = np.arange(n_components))
    if fitted_pca is None:
        pca = PCA(n_components=n_components)
    else:
        pca = fitted_pca
    if 'ensembles' in ds.dims:
        ds_out = xr.zeros_like(ds_)
        ds_out_flat = xr.zeros_like(ds_.reset_index(('lead_time')).assign_coords(time = np.arange(0,len(ds_.reset_index('lead_time').time))).stack(d = ('time','ensembles')).transpose(...,'mu'))
        if fitted_pca is None:
            ds_out_flat[:] = pca.fit_transform(ds.reset_index(('lead_time')).assign_coords(time = np.arange(0,len(ds_.reset_index('lead_time').time))).stack(d = ('time','ensembles')).transpose(...,'mu').values)
        else:
            ds_out_flat[:] = pca.transform(ds.reset_index(('lead_time')).assign_coords(time = np.arange(0,len(ds_.reset_index('lead_time').time))).stack(d = ('time','ensembles')).transpose(...,'mu').values)
        ds_out[:] = ds_out_flat.unstack('d').transpose(...,'ensembles','mu').values
    else:
        ds_out = xr.zeros_like(ds_)
        if fitted_pca is None:
            ds_out[:]  = pca.fit_transform(ds.values)
        else:
            ds_out[:]  = pca.transform(ds.values)
    if return_explained_variance:
        return ds_out,pca, pca.explained_variance_ratio_
    else:
        return ds_out, pca



def corr_patt(ds1, ds2 , mask=None,): # centered
    '''
    pattern correlation --
    '''
    dim_xy = ['lat','lon']
    covariance = ((ds1 - ds1.mean(dim_xy))*(ds2 - ds2.mean(dim_xy))).mean(dim_xy)
    result = (covariance/(ds1.std(dim_xy)*ds2.std(dim_xy)))

    return result   