
import gc
import sys
import os
sys.path.append(os.path.abspath("/home/rpg002/Decadal_BVAE/"))
import torch
from torch.distributions import Normal
from models.autoencoder import Autoencoder, MAF, RealNVP
from preprocessing import AnomaliesScaler_v1, AnomaliesScaler_v2, Standardizer, PreprocessingPipeline, Spatialnanremove, create_train_mask
import glob
import numpy as np
import xarray as xr
import pyshtools as pysh


def extract_params(model_dir):
    params = {}
    path = glob.glob(model_dir + '/*.txt')[0]
    file = open(path)
    content=file.readlines()
    for line in content:
        key = line.split('\t')[0]
        try:
            value = line.split('\t')[1].split('\n')[0]
        except:
            value = line.split('\t')[1]
        try:    
            params[key] = eval(value)
        except:
            if key == 'ensemble_list':
                ls = []
                for item in value.split('[')[1].split(']')[0].split(' '):
                    try:
                        ls.append(eval(item))
                    except:
                        pass
                params[key] = ls
            else:
                params[key] = value
    return params


def prepare_data_for_AE_toy(ds_in, model_dir, model_year = 2012):


    params = extract_params(model_dir)
    params['BVAE'] = int(model_dir.split('-')[-2][-2:])
    hyperparam = params["hyperparam"]
    reg_scale = params["reg_scale"]
    model = params["model"]
    hidden_dims = params["hidden_dims"]
    time_features = params["time_features"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    dropout_rate = params["dropout_rate"]
    condition_embedding_size = params['condition_embedding_size']

    extra_predictors = params['extra_predictors']
    batch_normalization = params['batch_normalization']

    params["version"] = eval(model_dir.split('/')[-1].split('_')[1][1])  


    if params['version'] == 1:

        params['forecast_preprocessing_steps'] = [
        ('standardize', Standardizer())]
        params['forecast_ensemble_mean_preprocessing_steps'] = [
        ('standardize', Standardizer())]
        params['observations_preprocessing_steps'] = []


    elif params['version'] == 2:

        params['forecast_preprocessing_steps'] = []
        params['forecast_ensemble_mean_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []
        params['remove_ensemble_mean'] = True


    else:
            params['forecast_preprocessing_steps'] = [  ('standardize', Standardizer(axis = (0,1,2)))]
            params['observations_preprocessing_steps'] = []
            params['forecast_ensemble_mean_preprocessing_steps'] = []

    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    forecast_ensemble_mean_preprocessing_steps = params["forecast_ensemble_mean_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]

    if 'CrMmbrTr' in  model_dir:
        params['cross_member_training'] = True 
    else:
        params['cross_member_training'] = False 

    if 'RmEnsMn' in  model_dir:
        params['remove_ensemble_mean'] = True 
    else:
        if params['version'] != 2:
            params['remove_ensemble_mean'] = False 

    if 'cEFullBVAE' in  model_dir:
        params['full_conditioning'] = True 
    else:
        params['full_conditioning'] = False 

    if 'Correction' in  model_dir:
        params['correction'] =  True
    else:
        params['correction'] = False 

    if 'latentdependant' in  model_dir:
        params['condition_dependant_latent'] = True
        assert params['condition_embedding_size'] is not None
        params['non_random_decoder_initialization'] = True
    else: 
        params['condition_dependant_latent'] = False

    if 'LY' in model_dir:
        lead_time = int(model_dir.split('LY')[1][0])
    else:
        lead_time = None
    if 'pR' not in  model_dir:
        params['min_posterior_variance'] =  None
    else:
        params['min_posterior_variance'] =  np.array(params['min_posterior_variance'])

    if 'condition_type' not in params.keys():
        params['condition_type'] = 'ensemble_mean'

    condition_embedding_size = params['condition_embedding_size']
    conditional_embedding = True if condition_embedding_size is not None else False
    params['conditional_embedding'] =conditional_embedding
 
    ds_raw_ensemble = ds_in.transpose('year','lead_time','ensembles',...)
    del ds_in
    gc.collect()

    train_years = ds_raw_ensemble.year[ds_raw_ensemble.year < model_year + 1].to_numpy()
    n_train = len(train_years)
    ds_baseline = ds_raw_ensemble[:n_train,...]
    train_mask = create_train_mask(ds_raw_ensemble[:n_train,...])

    if 'ensembles' in ds_raw_ensemble.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline.shape)
    else:
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline.shape)

    ds_em_before = ds_raw_ensemble.sel(ensembles = params['ensemble_list']).mean('ensembles')
    if params['remove_ensemble_mean']:  
        ds_before = ds_raw_ensemble - ds_raw_ensemble.mean('ensembles')
        if params['version'] == 2 : 
            ds_before = ds_before/ ds_raw_ensemble.std('ensembles')
    else:
        ds_before =  ds_raw_ensemble.copy()

    ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_before[:n_train,...], mask=preprocessing_mask_fct)
    ds = ds_pipeline.transform(ds_before)

    ds_em_pipeline = PreprocessingPipeline(forecast_ensemble_mean_preprocessing_steps).fit(ds_em_before[:n_train,...], mask=preprocessing_mask_fct[:,:,0,:,:,:])
    ds_em = ds_em_pipeline.transform(ds_em_before)

    year_max = ds[:n_train + 1].year[-1].values 

    del ds_baseline, preprocessing_mask_fct, 
    gc.collect()

    ds_train_ = ds[:n_train,...]


    if conditional_embedding:
        if params['condition_type'] == 'climatology':
                        ds_em = xr.concat([ds_em.mean(['year']).expand_dims('year', axis = 0) for _ in range(len(ds_em.year))], dim = 'year').assign_coords(year = ds_em.year.values)
        ds_train_conds_ = ds_em.sel(year = ds_train_.year).stack(time=('year','lead_time')).transpose('time',...)[~train_mask.flatten()]
        if lead_time is not None:
            ds_train_conds_ = ds_train_conds_.where((ds_train_conds_.lead_time >=  (lead_time - 1) * 12 + 1) & (ds_train_conds_.lead_time < (lead_time *12 )+1), drop = True)
    else:
        ds_train_conds_ = None
    try:       
        if params['prior_flow'] is not None:
            dics = {}
            if len((params['prior_flow'].split('args'))) > 0:
                pass
            
            dics['num_layers'] = eval((params['prior_flow'].split('num_layers'))[-1].split(',')[0].split(':')[-1])  
            try:
                dics['base_distribution'] = eval((params['prior_flow'].split('base_distribution'))[-1].split('}')[0].split(':')[-1])  
            except:
                pass
            dics['type'] = eval(model_dir.split('prior')[0].split('_')[-1])
            params['prior_flow'] = dics
    except:
        params['prior_flow'] = None  

    params['model_dir'] = model_dir
    params['model_year'] = model_year

    return ds_train_, ds_train_conds_, params


def extract_latent_space_toy(ds_train_, ds_train_conds_, params):
    # model_dir = dict_toy[idata]['dir_in'].split('/tests')[0]
    model_dir = params['model_dir']
    model_year = params['model_year']
    hyperparam = params["hyperparam"]
    reg_scale = params["reg_scale"]
    model = params["model"]
    hidden_dims = params["hidden_dims"]
    time_features = params["time_features"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    dropout_rate = params["dropout_rate"]
    condition_embedding_size = params['condition_embedding_size']

    extra_predictors = params['extra_predictors']
    batch_normalization = params['batch_normalization']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if params['conditional_embedding']:
        if 'condemb_to_decoder' not in params.keys():
            params['condemb_to_decoder'] = True
        if params['condition_dependant_latent']:
            params['full_conditioning'] = True
    
    ds_train = ds_train_.copy()
    if params['conditional_embedding']:
        ds_train_conds = ds_train_conds_.copy()


    img_dim = ds_train.shape[-2] * ds_train.shape[-1] 

    if time_features is None:
            add_feature_dim = 0
    else:
            add_feature_dim = len(time_features)
    if params['extra_predictors'] is not None:
        add_feature_dim = add_feature_dim + len(params['extra_predictors'])

    hidden_dims = params['hidden_dims']
    model = params['model']


    latent_dim = hidden_dims[0][-1]
    print('loading model ...')
    net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, VAE = params['BVAE'], condition_embedding_dims = params['condition_embedding_size'], full_conditioning = params['full_conditioning'], condition_dependant_latent = params["condition_dependant_latent"], min_posterior_variance = (params['min_posterior_variance']), prior_flow = params['prior_flow'], condemb_to_decoder = params['condemb_to_decoder'] ,device = device)
    net.load_state_dict(torch.load(glob.glob(model_dir+ '/Saved_models' + f'/*-{model_year}*.pth')[0], map_location=torch.device('cpu')))               
    net.to(device)
    net.eval()
    print(net)


    ds_train = ds_train.stack(time = ('year','lead_time')).stack(ref = ('lat','lon')).transpose('time','ensembles',...,'ref')
    if params['conditional_embedding']:
        ds_train_conds = ds_train_conds.stack(ref = ('lat','lon')).transpose('time',...,'ref')
    if time_features is not None:
        time_features_list = np.array([time_features]).flatten()
        feature_indices = {'year': 0, 'lead_time': 1, 'month_sin': 2, 'month_cos': 3, 'sin_t' : 4, 'cos_t' : 5}
        y = (ds_train.year.to_numpy() + np.floor(ds_train.lead_time.to_numpy()/12)) / year_max
        t =  np.arange(len(ds_train.year))
        lt = ds_train.lead_time.to_numpy() / np.max(ds_train.lead_time.to_numpy())
        tsin = np.sin(t)
        tcos = np.cos(t)
        msin = np.sin(2 * np.pi * ds_train.lead_time/12.0)
        mcos = np.cos(2 * np.pi * ds_train.lead_time/12.0)
        time_features = np.stack([y, lt, msin, mcos, tsin, tcos], axis=1)
        time_features = time_features[..., [feature_indices[k] for k in time_features_list]]

    ds_mu = xr.concat([xr.full_like(ds_train[...,-1].squeeze().drop(['ref','lat','lon']), np.NAN) for _ in range(latent_dim)],dim = 'mu').transpose('time','ensembles',...,'mu').rename('latent')
    ds_var = ds_mu.copy()
    ds_samples = ds_mu.copy()
    ds_mu_cond = None
    ds_var_cond = None
    ds_emb = None

    if params['condition_dependant_latent']:
        if params['prior_flow'] is None:
            ds_mu_cond = xr.concat([xr.full_like(ds_train_conds[...,-1].squeeze().drop(['ref','lat','lon']), np.NAN) for _ in range(latent_dim)],dim = 'mu').transpose('time',...,'mu').rename('latent')
            ds_var_cond = ds_mu_cond.copy()
        if any([params['condemb_to_decoder'], params['prior_flow'] is not None]):
            ds_emb = xr.concat([xr.full_like(ds_train_conds[...,-1].squeeze().drop(['ref','lat','lon']), np.NAN) for _ in range(net.embedding_size)],dim = 'mu').transpose('time',...,'mu').rename('latent')
    else:
        if params['condemb_to_decoder']:
            ds_emb = xr.concat([xr.full_like(ds_train_conds[...,-1].squeeze().drop(['ref','lat','lon']), np.NAN) for _ in range(net.embedding_size)],dim = 'mu').transpose('time',...,'mu').rename('latent')


    np.random.seed(1)
    torch.manual_seed(1)

    for t in range(len(ds_train.time)):
        x_in = torch.from_numpy(ds_train[t].data).to(torch.float32)
        if params['conditional_embedding']:
            cond = torch.from_numpy(ds_train_conds.isel(time = t).data).to(torch.float32)

            if params['condition_dependant_latent']:
                if all([params['prior_flow'] is None,params['condemb_to_decoder']])  :
                    ds_emb[t,:] = net.condition_mu(net.embedding(cond.flatten(start_dim=1))).detach().numpy().squeeze()
                elif params['prior_flow'] is not None:
                    ds_emb[t,:] = (net.embedding(cond.flatten(start_dim=1))).detach().numpy().squeeze()

            elif params['condemb_to_decoder']:
                ds_emb[t,:] = net.embedding(cond.flatten(start_dim=1)).detach().numpy().squeeze()
                
            cond = cond.unsqueeze(0).expand_as(x_in)
        else:
            cond = None
        if all([time_features is not None, params['append_mode'] in [1,3]]):
            tf = torch.from_numpy(time_features[t][None,]).to(torch.float32)
            x_in = (x_in, tf.expand(x_in.shape[0], add_feature_dim))
        
        with torch.no_grad():
            mu , log_var = net(x_in, condition = cond, sample_size = 1)[1:3]
            ds_samples[t,:,:] = net.sample(mu, log_var, 1 )[0]
            ds_mu[t,:,:] = mu
            # ds_var[t,:,:] = torch.exp(net(x_in, condition = cond, sample_size = 1)[2] + 1e-4)
            

            if params['condition_dependant_latent']:
                if params['prior_flow'] is None:
                    ds_mu_cond[t,:] = net.condition_mu(net.embedding(torch.from_numpy(ds_train_conds.isel(time = t).data).to(torch.float32))).squeeze()
                    ds_var_cond[t,:] = torch.exp(net.condition_log_var(net.embedding(torch.from_numpy(ds_train_conds.isel(time = t).data).to(torch.float32))).squeeze()) + 1e-4

    return ds_mu, ds_samples, ds_emb, ds_mu_cond, ds_var_cond, net, params

def normal_samples_within_bounds(mean, cov, num_samples, bounds=(-1, 1)):
    samples = []
    while len(samples) < num_samples:
        # Generate a sample from a multivariate normal distribution
        sample = np.random.multivariate_normal(mean, cov)
        # Check if the sample lies within the specified bounds
        if np.all(sample >= bounds[0]) and np.all(sample <= bounds[1]):
            samples.append(sample)
    return np.array(samples)


def generate_sin_phase_space_and_latent_encoding(ds_in, params, net, num_members = 50, pi_normal = True ):
    time = np.arange(len(ds_in.year) * len(ds_in.lead_time))

    harmonic_wavenumbers = ds_in.attrs['Spherical_harmonic_wavenumbers']
    harmonic_m = ds_in.attrs['Spherical_harmonic_order']
    freq_coeffs = ds_in.attrs['time_scales']


    s = normal_samples_within_bounds(np.zeros(len(freq_coeffs)), np.identity(len(freq_coeffs)), 50, bounds=(-np.inf, np.inf)).squeeze()
    target_ens = []
    ds_ens = []
    for e in range(num_members):
        ds_ls = []

        for t in time:
            if pi_normal:
                sin_phase =  np.sin((t * 2 * np.pi / freq_coeffs) +  np.pi * s[e])
            else:
                sin_phase =  np.sin((t * 2 * np.pi / freq_coeffs) +   s[e])
            combined_coeffs = pysh.SHCoeffs.from_zeros(90)
            for ind, hv in enumerate(harmonic_wavenumbers):
                
                coeffs = pysh.SHCoeffs.from_zeros(90)
                coeffs.coeffs[0, hv, harmonic_m[ind]] = sin_phase[ind] 
                combined_coeffs += coeffs

            combined_grid = combined_coeffs.expand(grid='DH')
            ds_ls.append(xr.DataArray(
                                    combined_grid.data,
                                    dims=["lat", "lon"],  # Define the dimensions
                                    coords={"lat": combined_grid.lats() , "lon": combined_grid.lons()},
                                    name = 'fgco2'))
                                    

        target_ens.append(xr.concat(ds_ls, dim = 'time'))
        if pi_normal:
            ds_ens.append(np.sin((time[:,None] * 2 * np.pi / freq_coeffs[None,]) +  np.pi * s[None,e])) 
        else:
            ds_ens.append(np.sin((time[:,None] * 2 * np.pi / freq_coeffs[None,]) +   s[None,e])) 

    maps_LE = xr.concat(target_ens, dim = 'ensembles')
    sin_phase_LE = np.concatenate([item[None,] for item in ds_ens])


    if params['remove_ensemble_mean']:
        # ds = ds_LE - ds_LE.mean(0)
        input_ds = maps_LE - maps_LE.mean('ensembles')
        if params['version'] == 2:
            # ds  =ds/ds_LE.std(0)
            input_ds = input_ds/maps_LE.std('ensembles')
    else:
        
        input_ds = maps_LE

    if params['conditional_embedding']:
        input_conds = maps_LE.mean('ensembles')

    x_out =  xr.full_like(input_ds[:,:,0,-2:].rename({'lon': 'mu'}), np.nan)
    x_in = torch.from_numpy(input_ds.stack(ref = ('lat','lon')).data).to(torch.float32)

    if params['time_features'] is not None:
        time_features_list = np.array([params['time_features']]).flatten()
        feature_indices = {'sin_t' : 0, 'cos_t' : 1}
        t =  np.arange(len(input_ds[0].time))
        tsin = np.sin(t)
        tcos = np.cos(t)
        time_features = np.stack([tsin, tcos], axis=1)
        time_features = time_features[..., [feature_indices[k] for k in time_features_list]]

        
    if params['time_features'] is not None:
        time_features_list = np.array([params['time_features']]).flatten()
        feature_indices = {'sin_t' : 0, 'cos_t' : 1}
        t =  np.arange(len(input_ds[0].time))
        tsin = np.sin(t)
        tcos = np.cos(t)
        time_features = np.stack([tsin, tcos], axis=1)
        time_features = time_features[..., [feature_indices[k] for k in time_features_list]]

    
    with torch.no_grad(): 
        if params['conditional_embedding']:
            cond = torch.from_numpy(input_conds.stack(ref = ('lat','lon')).data).to(torch.float32).unsqueeze(-2)
        else:
            cond = None
        for ens in range(input_ds.shape[0]):
            if all([params['time_features'] is not None, params['append_mode'] in [1,3]]):
                tf = torch.from_numpy(time_features).to(torch.float32)
                x_in_ = (x_in[ens], tf)
            else:
                x_in_ = x_in[ens]

            x_out[ens, :] =  net(x_in_.squeeze(), condition = cond, sample_size = 1)[1]

    return x_out, sin_phase_LE 




def prepare_data_for_AE_clim(ds_in, model_dir, model_year = 2012):

    params = extract_params(model_dir)
    params['BVAE'] = int(model_dir.split('-')[-2][-2:])
    hyperparam = params["hyperparam"]
    reg_scale = params["reg_scale"]
    model = params["model"]
    hidden_dims = params["hidden_dims"]
    time_features = params["time_features"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    dropout_rate = params["dropout_rate"]
    condition_embedding_size = params['condition_embedding_size']

    extra_predictors = params['extra_predictors']
    batch_normalization = params['batch_normalization']

    params["version"] = eval(model_dir.split('/')[-1].split('_')[1][1])  
        

    if params['version'] == 1:
            
            params['forecast_preprocessing_steps'] = [
            ('standardize', Standardizer())]
            params['forecast_ensemble_mean_preprocessing_steps'] = [
            ('standardize', Standardizer())]
            params['observations_preprocessing_steps'] = []

    elif params['version'] == 2:

            params['forecast_preprocessing_steps'] = [
            ('standardize', Standardizer(axis = (0,2)))]
            params['forecast_ensemble_mean_preprocessing_steps'] = [
            ('standardize', Standardizer(axis = (0,)))]
            params['observations_preprocessing_steps'] = []
    elif params['version'] == 3:

            params['forecast_preprocessing_steps'] = [
            ('standardize', Standardizer(axis = (0,1,2)))]
            params['forecast_ensemble_mean_preprocessing_steps'] = [
            ('standardize', Standardizer(axis = (0,1)))]
            params['observations_preprocessing_steps'] = []

    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    forecast_ensemble_mean_preprocessing_steps = params["forecast_ensemble_mean_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]

    if 'CrMmbrTr' in  model_dir:
        params['cross_member_training'] = True 
    else:
        params['cross_member_training'] = False 

    if 'RmEnsMn' in  model_dir:
        params['remove_ensemble_mean'] = True 
    else:
        params['remove_ensemble_mean'] = False 

    if 'cEFullBVAE' in  model_dir:
        params['full_conditioning'] = True 
    else:
        params['full_conditioning'] = False 

    if 'Correction' in  model_dir:
        params['correction'] =  True
    else:
        params['correction'] = False 

    if 'latentdependant' in  model_dir:
        params['condition_dependant_latent'] = True
        assert params['condition_embedding_size'] is not None
        params['non_random_decoder_initialization'] = True
    else: 
        params['condition_dependant_latent'] = False

    if 'LY' in model_dir:
        lead_time = int(model_dir.split('LY')[1][0])
    else:
        lead_time = None
    if 'pR' not in  model_dir:
        params['min_posterior_variance'] =  None
    else:
        params['min_posterior_variance'] =  np.array(params['min_posterior_variance'])

    condition_embedding_size = params['condition_embedding_size']
    conditional_embedding = True if condition_embedding_size is not None else False
    params['conditional_embedding'] =conditional_embedding

    ds_raw_ensemble = ds_in.transpose('year','lead_time','ensembles',...)
    del ds_in
    gc.collect()


    train_years = ds_raw_ensemble.year[ds_raw_ensemble.year < model_year + 1].to_numpy()
    n_train = len(train_years)
    ds_baseline = ds_raw_ensemble[:n_train,...]
    train_mask = create_train_mask(ds_raw_ensemble[:n_train,...])

    if 'ensembles' in ds_raw_ensemble.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline.shape)
    else:
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline.shape)



    ds_em = ds_raw_ensemble.sel(ensembles = params['ensemble_list']).mean('ensembles')
    if params['remove_ensemble_mean']:  
        ds = ds_raw_ensemble - ds_raw_ensemble.mean('ensembles')
    else:
        ds =  ds_raw_ensemble.copy()

    ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds[:n_train,...], mask=preprocessing_mask_fct)
    ds = ds_pipeline.transform(ds)

    ds_em_pipeline = PreprocessingPipeline(forecast_ensemble_mean_preprocessing_steps).fit(ds_em[:n_train,...], mask=preprocessing_mask_fct[:,:,0,...])
    ds_em = ds_em_pipeline.transform(ds_em)
    
    ###
    year_max = ds[:n_train + 1].year[-1].values 

    del ds_baseline, preprocessing_mask_fct, 
    gc.collect()

    ds_train_ = ds[:n_train,...]

    if conditional_embedding:
        if params['condition_type'] == 'climatology':
                        ds_em = xr.concat([ds_em.mean(['year']).expand_dims('year', axis = 0) for _ in range(len(ds_em.year))], dim = 'year').assign_coords(year = ds_em.year.values)
        ds_train_conds_ = ds_em.sel(year = ds_train_.year).stack(time=('year','lead_time')).transpose('time',...)[~train_mask.flatten()]
        if lead_time is not None:
            ds_train_conds_ = ds_train_conds_.where((ds_train_conds_.lead_time >=  (lead_time - 1) * 12 + 1) & (ds_train_conds_.lead_time < (lead_time *12 )+1), drop = True)
    else:
        ds_train_conds_ = None

    try:       
        if params['prior_flow'] is not None:
            dics = {}
            if len((params['prior_flow'].split('args'))) > 0:
                pass
            
            dics['num_layers'] = eval((params['prior_flow'].split('num_layers'))[-1].split(',')[0].split(':')[-1]) 
            try: 
                dics['base_distribution'] = eval((params['prior_flow'].split('base_distribution'))[-1].split('}')[0].split(':')[-1])  
            except:
                pass
            dics['type'] = eval(model_dir.split('prior')[0].split('_')[-1])
            params['prior_flow'] = dics
    except:
        params['prior_flow'] = None

    params['model_dir'] = model_dir
    params['model_year'] = model_year

    return ds_train_, ds_train_conds_, params


def extract_latent_space_clim(ds_train_, ds_train_conds_, params):
    model_dir = params['model_dir']
    model_year = params['model_year']
    hyperparam = params["hyperparam"]
    reg_scale = params["reg_scale"]
    model = params["model"]
    hidden_dims = params["hidden_dims"]
    time_features = params["time_features"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    dropout_rate = params["dropout_rate"]
    condition_embedding_size = params['condition_embedding_size']

    extra_predictors = params['extra_predictors']
    batch_normalization = params['batch_normalization']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if params['conditional_embedding']:
        if 'condemb_to_decoder' not in params.keys():
            params['condemb_to_decoder'] = True
        if all(['full_conditioning' not in params.keys(), params['condition_dependant_latent']]):
            params['full_conditioning'] = True
    
    ds_train = ds_train_.copy()
    if params['conditional_embedding']:
        ds_train_conds = ds_train_conds_.copy()

    img_dim = ds_train.shape[-2] * ds_train.shape[-1] 

    if time_features is None:
            add_feature_dim = 0
    else:
            add_feature_dim = len(time_features)
    if extra_predictors is not None:
        add_feature_dim = add_feature_dim + len(params['extra_predictors'])

    hidden_dims = params['hidden_dims']
    model = params['model']

    
    latent_dim = hidden_dims[0][-1]
    net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, VAE = params['BVAE'], condition_embedding_dims = params['condition_embedding_size'], full_conditioning = params['full_conditioning'], condition_dependant_latent = params["condition_dependant_latent"], min_posterior_variance = (params['min_posterior_variance']), prior_flow = params['prior_flow'], condemb_to_decoder = params['condemb_to_decoder'] ,device = device)
    net.load_state_dict(torch.load(glob.glob(model_dir+ '/Saved_models' + f'/*-{model_year}*.pth')[0], map_location=torch.device('cpu')))               
    net.to(device)
    net.eval()
    print(net)

    ds_train = ds_train.stack(time = ('year','lead_time')).stack(ref = ('lat','lon')).transpose('time','ensembles',...,'ref')
    if params['conditional_embedding']:
        ds_train_conds = ds_train_conds.stack(ref = ('lat','lon')).transpose('time',...,'ref')
    if time_features is not None:
        time_features_list = np.array([time_features]).flatten()
        feature_indices = {'year': 0, 'lead_time': 1, 'month_sin': 2, 'month_cos': 3, 'sin_t' : 4, 'cos_t' : 5}
        y = (ds_train.year.to_numpy() + np.floor(ds_train.lead_time.to_numpy()/12)) / year_max
        t =  np.arange(len(ds_train.year))
        lt = ds_train.lead_time.to_numpy() / np.max(ds_train.lead_time.to_numpy())
        tsin = np.sin(t)
        tcos = np.cos(t)
        msin = np.sin(2 * np.pi * ds_train.lead_time/12.0)
        mcos = np.cos(2 * np.pi * ds_train.lead_time/12.0)
        time_features = np.stack([y, lt, msin, mcos, tsin, tcos], axis=1)
        time_features = time_features[..., [feature_indices[k] for k in time_features_list]]

    ds_mu = xr.concat([xr.full_like(ds_train[...,-1].squeeze().drop(['ref','lat','lon']), np.NAN) for _ in range(latent_dim)],dim = 'mu').transpose('time','ensembles',...,'mu').rename('latent')
    ds_var = ds_mu.copy()
    ds_samples = ds_mu.copy()
    ds_mu_cond = None
    ds_var_cond = None
    ds_emb = None

    if params['condition_dependant_latent']:
        if params['prior_flow'] is None:
            ds_mu_cond = xr.concat([xr.full_like(ds_train_conds[...,-1].squeeze().drop(['ref','lat','lon']), np.NAN) for _ in range(latent_dim)],dim = 'mu').transpose('time',...,'mu').rename('latent')
            ds_var_cond = ds_mu_cond.copy()
        if any([params['condemb_to_decoder'], params['prior_flow'] is not None]):
            ds_emb = xr.concat([xr.full_like(ds_train_conds[...,-1].squeeze().drop(['ref','lat','lon']), np.NAN) for _ in range(net.embedding_size)],dim = 'mu').transpose('time',...,'mu').rename('latent')
    else:
        if params['condemb_to_decoder']:
            ds_emb = xr.concat([xr.full_like(ds_train_conds[...,-1].squeeze().drop(['ref','lat','lon']), np.NAN) for _ in range(net.embedding_size)],dim = 'mu').transpose('time',...,'mu').rename('latent')

    np.random.seed(1)
    torch.manual_seed(1)

    for t in range(len(ds_train.time)):
        x_in = torch.from_numpy(ds_train[t].data).to(torch.float32)
        if params['conditional_embedding']:
            cond = torch.from_numpy(ds_train_conds.isel(time = t).data).to(torch.float32)

            if params['condition_dependant_latent']:
                if all([params['prior_flow'] is None,params['condemb_to_decoder']])  :
                    ds_emb[t,:] = net.condition_mu(net.embedding(cond.flatten(start_dim=1))).detach().numpy().squeeze()
                elif params['prior_flow'] is not None:
                    ds_emb[t,:] = (net.embedding(cond.flatten(start_dim=1))).detach().numpy().squeeze()

            elif params['condemb_to_decoder']:
                ds_emb[t,:] = net.embedding(cond.flatten(start_dim=1)).detach().numpy().squeeze()
                
            cond = cond.unsqueeze(0).expand_as(x_in)
        else:
            cond = None

        if all([time_features is not None, params['append_mode'] in [1,3]]):
            tf = torch.from_numpy(time_features[t][None,]).to(torch.float32)
            x_in = (x_in, tf.expand(x_in.shape[0], add_feature_dim))
        
        with torch.no_grad():
            mu , log_var = net(x_in, condition = cond, sample_size = 1)[1:3]
            ds_samples[t,:,:] = net.sample(mu, log_var, 1 )[0]
            ds_mu[t,:,:] = mu
            # ds_var[t,:,:] = torch.exp(net(x_in, condition = cond, sample_size = 1)[2] + 1e-4)
            

            if params['condition_dependant_latent']:
                if params['prior_flow'] is None:
                    ds_mu_cond[t,:] = net.condition_mu(net.embedding(torch.from_numpy(ds_train_conds.isel(time = t).data).to(torch.float32))).squeeze()
                    ds_var_cond[t,:] = torch.exp(net.condition_log_var(net.embedding(torch.from_numpy(ds_train_conds.isel(time = t).data).to(torch.float32))).squeeze()) + 1e-4

    return ds_mu, ds_samples, ds_emb, ds_mu_cond, ds_var_cond, net, params
