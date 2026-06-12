import numpy as np
import matplotlib.pyplot as plt
import tqdm
import xarray as xr
from pathlib import Path
from torch.distributions import Normal
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from models.autoencoder import Autoencoder, MAF, RealNVP
import gc
import os
import glob


from losses import WeightedMSE, WeightedMSESignLossKLD, Frobenius_norm
from preprocessing import align_data_and_targets, create_train_mask, reshape_obs_to_data
from preprocessing import  Standardizer, PreprocessingPipeline, Spatialnanremove
from torch_datasets import XArrayDataset
from data_locations import *




def beta_finder(step, num_batches, beta ):
    if type(beta) == dict:
        if beta['num_epochs_hold'] is None:
            range_epochs = (beta['num_epoch_warmup'])*num_batches
            return beta['start'] + (beta['end'] - beta['start']) * min((step /range_epochs), 1)
        else:

            range_epochs =   (beta['num_epoch_warmup'] + beta['num_epochs_hold'])*num_batches                              
            cycle_pos = step % range_epochs
            return  beta['start'] + (beta['end'] - beta['start']) * min ((cycle_pos / (beta['num_epoch_warmup']*num_batches)),1)
    else:
        return beta


def resolve_output_address(out_dir_x : str, params : dict, n_validation_years = int, start_test_years = None):
        
        if type(params['beta']) == dict:
            if params['beta']['num_epochs_hold'] is not None:
                beta_arg = 'CycBanealing'
            else:
                beta_arg = 'anealing'
        else:
            beta_arg = params["beta"]


        out_dir = f'{out_dir_x}/ST{min(start_test_years)}_VAL{n_validation_years}_V{params['version']}_B{beta_arg}_batch{params["batch_size"]}_e{params["epochs"]}' 


        if params['lr_scheduler']:
            out_dir = out_dir + '_cosine_lr_scheduler'


        if any([all([params['time_features'] is not None, params['append_mode'] != 1]), params['condition_embedding_size'] is not None]):
            
            if params['condition_embedding_size'] is not None:
                    if params['condition_dependant_latent']:
                        if params['condemb_to_decoder'] is False:
                            out_dir = out_dir + f'_cBVAElatentdependant_cR_{params["boosted_ensemble_size"]}-{params["training_sample_size"]}'
                        else:
                            out_dir = out_dir + f'_cBVAElatentdependant_{params["boosted_ensemble_size"]}-{params["training_sample_size"]}'
                    elif params['full_conditioning']:
                        out_dir = out_dir + f'_cEFullBVAE_{params["boosted_ensemble_size"]}-{params["training_sample_size"]}'
                    else:
                        out_dir = out_dir + f'_cEBVAE_{params["boosted_ensemble_size"]}-{params["training_sample_size"]}'
                    if params['condition_type'] == 'cross_ensemble':
                        out_dir = out_dir + '_XEnsCond'
            else:
                params["full_conditioning"] = False
                out_dir = out_dir + f'_cBVAE_{params["boosted_ensemble_size"]}-{params["training_sample_size"]}'

        else:
            out_dir = out_dir + f'_BVAE_{params["boosted_ensemble_size"]}-{params["training_sample_size"]}'

        
        if params['prior_flow'] is not None:
            out_dir = out_dir + f'_{params["prior_flow"]["type"].__name__}prior'
        
        if params['non_random_decoder_initialization']:
                out_dir = out_dir + '_NnRandDecodInit'


        if params['loss_reduction'] == 'sum':
            out_dir = out_dir + f'_MSESUM'


        if params['min_posterior_variance'] is not None:
            out_dir = out_dir + f'_pR'

        if params['Frobenius_norm_weight'] is not None:
            out_dir = out_dir + f"_CC{params['Frobenius_norm_weight']}"
            
        out_dir = out_dir + f"_TSE{len(params['ensemble_list'])}_LS{params['hidden_dims'][0][-1]}" 

        if params['condition_embedding_size'] is not None:
            out_dir = out_dir + f'_condembsize{params["condition_embedding_size"][-1]}'
        if params['equal_weights']:
            out_dir = out_dir + "_EQW"

        return out_dir

def run_training(params,var,  lead_years = 1, n_validation_years = 0, lead_time = None, n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None, start_test_years = None):
    if var == 'tas':
        data_dir_forecast = LOC_historical_tas
        data_dir_ssp = LOC_ssp245_tas
        data_dir_obs = LOC_historical_tas
        unit_change = 1  ## Change units for ESM data to mol m-2 yr-1

    elif var == 'pr':
        data_dir_forecast = LOC_historical_pr
        data_dir_ssp = LOC_ssp245_pr
        data_dir_obs = LOC_historical_pr
        unit_change = 1000  ## Change units for ESM data to mol m-2 yr-1

    if params['boosted_ensemble_size'] is not None:
        assert type(params['boosted_ensemble_size']) == int, 'Input the size of output ensemble as boosted_ensemble_size ...'
    else:
        params['boosted_ensemble_size'] = 1
    
    if params['boosted_ensemble_size'] is not None:
        params['ensemble_mode'] = 'LE'
        assert params['ensemble_list'] is not None, 'for the cVAE model you need to specify the ensemble size ...'
    
    if not params['non_random_decoder_initialization']:
       if all([ any([params['time_features'] is None, params['append_mode'] != 3]),params['condition_embedding_size'] is None]):
            print(' -------- \n Warning: random decoder initializaiton is True without any conditions provided to the decoder \n --------' ) # else:



    if params['version'] == 1:

        params['forecast_preprocessing_steps'] = [
        ('standardize', Standardizer())]
        params['forecast_ensemble_mean_preprocessing_steps'] = [
        ('standardize', Standardizer())]
        params['observations_preprocessing_steps'] = []

    elif params['version'] == 2:

        params['forecast_preprocessing_steps'] = [
        ('standardize', Standardizer(axis = (0,2)))]
        params['forecast_ensemble_mean_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []

    else:
        params['forecast_preprocessing_steps'] = []
        params['forecast_ensemble_mean_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []
    
    if params['lr_scheduler']:
        max_learning_rate = params['lr']
        min_lr = params['min_lr']
        num_warmup_epchs = params['num_warmup_epchs']
    else:
        min_lr = num_warmup_epchs = max_learning_rate = None

    print("Start training")


    ensemble_list = params['ensemble_list']
    ensemble_mode = params['ensemble_mode'] 


    if all([params['condition_embedding_size'] is not None, params['condition_dependant_latent'] is False]):
        print('Warning: condemb_to_decoder turned True for prior is not condition dependant for cVAE ...')
        params['condemb_to_decoder'] = True

    if params['condition_dependant_latent']:
        assert params['condition_embedding_size'] is not None, 'Specify condition embedding network size for condition dependant latent ...'
        if params['prior_flow'] is None:
            params['non_random_decoder_initialization'] = True
            print('Warning: non_random_decoder_initialization turned on for condition dependant latent in cVAE to be sampled (flow is off) ...')
                  
        else:
            assert params['loss_reduction'] == 'sum', 'loss_reduction has to be sum for normalized flow priors'
            assert params['non_random_decoder_initialization'] is False, 'non_random_decoder_initialization should be False for condition dependant flow based prior ...'
        
        params['full_conditioning'] = True
        print('Warning: full_conditioning turned True for condition dependant latent in cVAE ...')

    if params['condition_embedding_size'] == 'encoder':
        params['condition_embedding_size'] = params["hidden_dims"][0]

    if params['condition_type'] == 'cross_ensemble':
        assert params['condition_embedding_size'] is not None
        assert len(params['ensemble_list']) > 1

    if ensemble_list is not None: ## PG: calculate the mean if ensemble mean is none
        print("Load forecasts")
        ds_in = xr.open_dataset(data_dir_forecast).isel(ensembles = ensemble_list).sel(year = slice(None, None)).load()[var] * unit_change
        if max(start_test_years) > 2015:
            ds_ssp = xr.open_dataset(data_dir_ssp).isel(ensembles = ensemble_list).sel(year = slice(None, max(start_test_years))).load()[var] * unit_change
            ds_in = xr.concat([ds_in, ds_ssp], dim = 'year')
            del ds_ssp
        if ensemble_mode == 'Mean': ##
            ds_in = ds_in.mean('ensembles') ##
        else:
            print(f'Warning: ensemble_mode is {ensemble_mode}. Training for large ensemble ...')

    else:    ## Load specified members
        print("Load forecasts") 
        ds_in = xr.open_dataset(data_dir_forecast).mean('ensembles').sel(year = slice(None, None)).load()[var] * unit_change
        if max(start_test_years) > 2015:
            ds_ssp = xr.open_dataset(data_dir_ssp).mean('ensembles').sel(year = slice(None, max(start_test_years))).load()[var] * unit_change
            ds_in = xr.concat([ds_in, ds_ssp], dim = 'year')
            del ds_ssp
        

    print("Load observations")

    obs_in = ds_in.mean('ensembles')[:,:12].rename({'lead_time' : 'month'})

    obs_in = obs_in.expand_dims('channels', axis=2)
    
    if 'ensembles' in ds_in.dims: ### PG: add channels dimention to the correct axis based on whether we have ensembles or not
        ds_in = ds_in.expand_dims('channels', axis=3).sortby('ensembles')
    else:
        ds_in = ds_in.expand_dims('channels', axis=2) 

    ds_raw, obs_raw = align_data_and_targets(ds_in, obs_in, lead_years)  # extract valid lead times and usable years

    if 'ensembles' in ds_raw.dims: ## PG: reorder dimensions in you have ensembles
        ds_raw_ensemble_mean = ds_raw.transpose('year','lead_time','ensembles',...)
    else:
        ds_raw_ensemble_mean = ds_raw.transpose('year','lead_time',...)
    
    obs_raw = reshape_obs_to_data(obs_raw, ds_raw_ensemble_mean, return_xarray=True)

    if not ds_raw_ensemble_mean.year.equals(obs_raw.year):
            
            ds_raw_ensemble_mean = ds_raw_ensemble_mean.sel(year = obs_raw.year)
    
    del ds_in, obs_in
    gc.collect()

    ##### PG: The ocean has NaN values over land in both forecast and obs data and these are not necessarily in the excat same grid points. ###
    ##### We need to extract the common grid points where both obs and model data exist. That said, we need to flatten both the training and target data
    ##### I defined a Nanremover class. See preprocessing.py.
    
    nanremover = Spatialnanremove()## PG: Get an instance of the class
    nanremover.fit(ds_raw_ensemble_mean[:,:12,...], ds_raw_ensemble_mean[:,:12,...]) ## PG:extract the commong grid points between training and obs data
    ds_raw_ensemble_mean = nanremover.to_map(nanremover.sample(ds_raw_ensemble_mean)) ## PG: flatten and sample training data at those locations
    obs_raw = nanremover.to_map(nanremover.sample(obs_raw)) ## PG: flatten and sample obs data at those locations    
    #######################################################################################################################################



    model = params["model"]
    hidden_dims = params["hidden_dims"]
    time_features = params["time_features"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    batch_normalization = params["batch_normalization"]
    dropout_rate = params["dropout_rate"]
    condition_embedding_size = params['condition_embedding_size']
    optimizer = params["optimizer"]
    lr = params["lr"]
    l2_reg = params["L2_reg"]
    
    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    forecast_ensemble_mean_preprocessing_steps = params["forecast_ensemble_mean_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]

    assert np.all((start_test_years) <= ds_raw_ensemble_mean.year[-1].values + 1)
    test_years = start_test_years
    

    conditional_embedding = True if condition_embedding_size is not None else False

    if n_runs > 1:
        numpy_seed = None
        torch_seed = None

    with open(Path(results_dir, "training_parameters.txt"), 'w') as f:
        f.write(
            f"model\t{model.__name__}\n" +
            f"beta\t{params['beta']}\n" +
            f"hidden_dims\t{hidden_dims}\n" +
            f"time_features\t{time_features}\n" +
            f"append_mode\t{params['append_mode']}\n" +
            f"ensemble_list\t{ensemble_list}\n" + ## PG: Ensemble list
            f"epochs\t{epochs}\n" +
            f"batch_size\t{batch_size}\n" +
            f"batch_normalization\t{batch_normalization}\n" +
            f"dropout_rate\t{dropout_rate}\n" +
            f"L2_reg\t{l2_reg}\n" + 
            f"optimizer\t{optimizer.__name__}\n" +
            f"lr\t{params['lr']}\n" +
            f"lr_scheduler\t{params['lr_scheduler']}: {max_learning_rate} --> {min_lr} cosine annealing with {num_warmup_epchs} warm up epochs\n" + 
            f"forecast_preprocessing_steps\t{[s[0] if forecast_preprocessing_steps is not None else None for s in forecast_preprocessing_steps]}\n" +
            f"observations_preprocessing_steps\t{[s[0] if observations_preprocessing_steps is not None else None for s in observations_preprocessing_steps]}\n" +
            f"condition_embedding_size\t{params['condition_embedding_size']}\n" +
            f"condition_type\t{params['condition_type']}\n" +
            f"condemb_to_decoder\t{params['condemb_to_decoder']}\n" +
            f"prior_flow\t{params['prior_flow']}\n" +
            f"min_posterior_variance\t{params['min_posterior_variance']}\n" + 
            f"non_random_decoder_initialization\t{params['non_random_decoder_initialization']}\n" + 
            f"loss_reduction\t{params['loss_reduction']}\n" 
        )
    
    del ds_raw
    gc.collect()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for run in range(n_runs):
        print(f"Start run {run + 1} of {n_runs}...")
        for y_idx, test_year in enumerate(test_years):
            print(f"Start run for test year {test_year}...")

            train_years = ds_raw_ensemble_mean.year[ds_raw_ensemble_mean.year < test_year - n_validation_years].to_numpy()

            n_train = len(train_years)


            ds_baseline = ds_raw_ensemble_mean[:n_train,...] 
            obs_baseline = obs_raw[:n_train,...] 
            train_mask = create_train_mask(ds_raw_ensemble_mean[:n_train,...])

            if 'ensembles' in ds_raw_ensemble_mean.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
                preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline.shape)
            else:
                preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline.shape)

            preprocessing_mask_obs = np.broadcast_to(train_mask[...,None,None,None], obs_baseline.shape)


            if numpy_seed is not None:
                np.random.seed(numpy_seed)
            if torch_seed is not None:
                torch.manual_seed(torch_seed)
      

            ds_em_before = ds_raw_ensemble_mean.mean('ensembles')

            ds_before =  ds_raw_ensemble_mean.copy()

            ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_before[:n_train,...], mask=preprocessing_mask_fct)
            ds = ds_pipeline.transform(ds_before)

            if params['version'] == 2:
                if params['condition_type'] == 'cross_ensemble':
                    ds_em = ds_pipeline.transform(ds_raw_ensemble_mean).squeeze().rename({'ensembles' : 'channels'})

                else:
                    ds_em = ds_pipeline.transform(ds_em_before.expand_dims('ensembles', axis  =2)).squeeze()
            
            else:

                ds_em_pipeline = PreprocessingPipeline(forecast_ensemble_mean_preprocessing_steps).fit(ds_em_before[:n_train,...], mask=preprocessing_mask_fct)
                if params['condition_type'] == 'cross_ensemble':
                    ds_em = ds_em_pipeline.transform(ds_raw_ensemble_mean.squeeze().rename({'ensembles' : 'channels'}))
                else:
                    ds_em = ds_em_pipeline.transform(ds_em_before)

            obs_pipeline = PreprocessingPipeline(observations_preprocessing_steps).fit(obs_baseline, mask=preprocessing_mask_obs)
            obs = obs_pipeline.transform(obs_raw)
            ####################################################################################
            year_max = ds.year[-1].values 


            del ds_baseline, obs_baseline, preprocessing_mask_obs, preprocessing_mask_fct, ds_before,ds_em_before
            gc.collect()
            # TRAIN MODEL
            ####### time inclusion
            
            ds_train = ds[:n_train,...]
            obs_train = obs[:n_train,...]

            ds_validation = ds[n_train:n_train + n_validation_years,...]
            obs_validation = obs[n_train:n_train + n_validation_years,...]   


            if test_year < ds.year[-1] + 1 :
                ds_test = ds[n_train + n_validation_years :,...]
                obs_test = obs[n_train + n_validation_years :,...]


            weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
            weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove() 
            if params['equal_weights']:
                weights = xr.ones_like(weights)

            weights_ = weights.copy()
        
    
            ########################################################################

            ds_train = nanremover.sample(ds_train) ## PG: flatten and sample training data at those locations
            obs_train = nanremover.sample(obs_train) ## PG: flatten and sample obs data at those locations   
            ds_validation = nanremover.sample(ds_validation) ## PG: flatten and sample training data at those locations
            obs_validation = nanremover.sample(obs_validation) ## PG: flatten and sample obs data at those locations   
            weights = nanremover.sample(weights) ## PG: flatten and sample weighs at those locations
            weights_ = nanremover.sample(weights_)

            img_dim = ds_train.shape[-1] ## PG: The input dim is now the length of the flattened dimention.


            del ds, obs
            gc.collect()
            torch.cuda.empty_cache() 
            torch.cuda.synchronize() 
            weights = weights.values
            weights_ = weights_.values

            if time_features is None:
                    add_feature_dim = 0
            else:
                    add_feature_dim = len(time_features)


            net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, VAE = params['boosted_ensemble_size'], condition_embedding_dims = params['condition_embedding_size'], full_conditioning = params["full_conditioning"] , condition_dependant_latent = params["condition_dependant_latent"], 
                            min_posterior_variance = params['min_posterior_variance'], prior_flow = params['prior_flow'], condemb_to_decoder = params['condemb_to_decoder'],  device = device)

            net.to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = l2_reg)

            ## PG: XArrayDataset now needs to know if we are adding ensemble features. The outputs are datasets that are maps or flattened in space depending on the model.
            val_mask = create_train_mask(ds_validation)
            
            train_set = XArrayDataset(ds_train, obs_train, mask=train_mask, lead_time = lead_time, in_memory=False, time_features=time_features, aligned = True, year_max = year_max, conditional_embedding = conditional_embedding) 
            dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

            validation_set = XArrayDataset(ds_validation, obs_validation, mask=val_mask, lead_time = lead_time,  in_memory=False, time_features=time_features, aligned = True, year_max = year_max, conditional_embedding = conditional_embedding) 
            dataloader_val = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

            if params['lr_scheduler']:
                if num_warmup_epchs > 0:
                    # scheduler = get_cosine_schedule_with_warmup(optimizer, len(dataloader) * num_warmup_epchs, len(dataloader) * params['epochs'],  min_lr)
                    warmup_scheduler = LinearLR(optimizer, start_factor=0.0001, end_factor=1.0, total_iters=len(dataloader) * num_warmup_epchs)
                    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * (params['epochs']  - num_warmup_epchs), eta_min=min_lr)
                    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],milestones=[len(dataloader) * num_warmup_epchs])
                else:   
                    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * params['epochs'], eta_min=min_lr)       
            

            if conditional_embedding:
                if params['condition_type'] == 'climatology':
                    ds_em = xr.concat([ds_em.mean('year').expand_dims('year', axis = 0) for _ in range(len(ds_em.year))], dim = 'year').assign_coords(year = ds_em.year.values)

                ds_train_conds = nanremover.sample(ds_em.sel(year = ds_train.year)).stack(flattened=('year','lead_time')).transpose('flattened',...)[~train_mask.flatten()]
                ds_val_conds = nanremover.sample(ds_em.sel(year = ds_validation.year)).stack(flattened=('year','lead_time')).transpose('flattened',...)[~val_mask.flatten()]

                if lead_time is not None:
                    ds_train_conds = ds_train_conds.where((ds_train_conds.lead_time >=  (lead_time - 1) * 12 + 1) & (ds_train_conds.lead_time < (lead_time *12 )+1), drop = True)
                    ds_val_conds = ds_val_conds.where((ds_val_conds.lead_time >=  (lead_time - 1) * 12 + 1) & (ds_val_conds.lead_time < (lead_time *12 )+1), drop = True)

                ds_train_conds = torch.from_numpy(ds_train_conds.to_numpy())
                ds_val_conds = torch.from_numpy(ds_val_conds.to_numpy())

                if test_year < year_max + 1:
                    ds_test_conds = nanremover.sample(ds_em.sel(year = ds_test.year)).stack(flattened=('year','lead_time')).transpose('flattened',...)
                    if params['condition_type'] == 'cross_ensemble':
                        ds_test_conds = ds_test_conds[:,0,:]
                    if lead_time is not None:
                        ds_test_conds = ds_test_conds.where((ds_test_conds.lead_time >=  (lead_time - 1) * 12 + 1) & (ds_test_conds.lead_time < (lead_time *12 )+1), drop = True)
                    ds_test_conds = torch.from_numpy(ds_test_conds.to_numpy())

            criterion = WeightedMSESignLossKLD(weights=weights, device=device, reduction=params['loss_reduction'])
            if params['Frobenius_norm_weight'] is not None:
                Frobenius_loss = Frobenius_norm( img_dim = img_dim , weight = params['Frobenius_norm_weight'])

            epoch_loss = []
            epoch_MSE = []
            epoch_FLD = []
            epoch_loss_validation = []
            epoch_MSE_validation = []
            epoch_FLD_validation = []
            
            num_batches = len(dataloader)
            num_batches_val = len(dataloader_val)
            step = 0
            for epoch in tqdm.tqdm(range(epochs)):
                

                net.train()
                batch_loss = 0
                batch_loss_MSE = 0
                batch_loss_FLD = 0
                optimizer.zero_grad()
                for batch, (x, y) in enumerate(dataloader):

                    beta = beta_finder(step, num_batches, params['beta'] )
                    step = step +1

                    if conditional_embedding:
                        cond_idx = x[-1]
                        x = [x[i] for i in range(len(x) - 1)] if len(x) >2 else x[0]
                        cond = ds_train_conds[cond_idx].float().to(device)
                    else:
                        cond = None
                    if params['condition_type'] == 'cross_ensemble':
                        batch_idx = torch.arange(cond.size(0), device=cond.device)
                        channel_idx = torch.randint(0, cond.size(1), (cond.size(0),), device=cond.device)
                        cond = cond[batch_idx, channel_idx, :]
                    if (type(x) == list) or (type(x) == tuple):
                        x = (x[0].to(device), x[1].to(device))
                    else:
                        x = x.to(device)
                    if (type(y) == list) or (type(y) == tuple):
                        y, m = (y[0].to(device), y[1].to(device))
                    else:
                        y = y.to(device)
                        m  = None

                    
                    y = x[0] if (type(x) == list) or (type(x) == tuple) else x
                    if params['condition_dependant_latent']:
                        if net.flow is None:
                            adjusted_forecast, mu, log_var, cond_mu, cond_log_var = net(x, condition = cond,  sample_size = params['training_sample_size'])
                            loss, MSE, KLD = criterion(adjusted_forecast, y.unsqueeze(0).expand_as(adjusted_forecast) ,mu, log_var, cond_mu = cond_mu, cond_log_var = cond_log_var ,beta = beta, mask = m, return_ind_loss=True )
                        else:
                            adjusted_forecast, mu, log_var, cond_emb = net(x, condition = cond,  sample_size = params['training_sample_size'])
                            loss, MSE, KLD = criterion(adjusted_forecast, y.unsqueeze(0).expand_as(adjusted_forecast) ,mu, log_var, cond_mu = cond_emb, cond_log_var = None ,beta = beta, mask = m, return_ind_loss=True, normalized_flow = net.flow )
                    else:
                        adjusted_forecast, mu, log_var = net(x, condition = cond,  sample_size = params['training_sample_size'])
                        loss, MSE, KLD = criterion(adjusted_forecast, y.unsqueeze(0).expand_as(adjusted_forecast) ,mu, log_var,beta = beta, mask = m, return_ind_loss=True, normalized_flow = net.flow  )

                    if params['Frobenius_norm_weight'] is not None:
                        FL = Frobenius_loss(adjusted_forecast.mean(0), y)
                        loss = loss + FL
                        batch_loss_FLD += FL.item()

                    batch_loss += loss.item()
                    batch_loss_MSE += MSE.item()
                    loss.backward()
                    optimizer.step()
                    if params['lr_scheduler']:
                        scheduler.step()
                    optimizer.zero_grad()
                epoch_loss.append(batch_loss / num_batches)
                epoch_MSE.append(batch_loss_MSE / num_batches)
                epoch_FLD.append(batch_loss_FLD / num_batches)

                del x, y , m, loss, MSE, KLD 
                gc.collect()
                torch.cuda.empty_cache() 
                ################################# Validation ###############################
                net.eval()
                batch_loss_validation = 0
                batch_loss_MSE_validation = 0
                batch_loss_FLD_validation = 0
                for batch, (x, y) in enumerate(dataloader_val):
                    with torch.no_grad(): 

                        if conditional_embedding:
                            cond_idx = x[-1]
                            x = [x[i] for i in range(len(x) - 1)] if len(x) >2 else x[0]
                            cond = ds_val_conds[cond_idx].float().to(device)
                        else:
                            cond = None

                        if params['condition_type'] == 'cross_ensemble':
                            batch_idx = torch.arange(cond.size(0), device=cond.device)
                            channel_idx = torch.randint(0, cond.size(1), (cond.size(0),), device=cond.device)
                            cond = cond[batch_idx, channel_idx, :]

                        if (type(x) == list) or (type(x) == tuple):
                            x = (x[0].to(device), x[1].to(device))
                        else:
                            x = x.to(device)
                        if (type(y) == list) or (type(y) == tuple):
                            y, m = (y[0].to(device), y[1].to(device))
                        else:
                            y = y.to(device)
                            m  = None

                        y = x[0] if (type(x) == list) or (type(x) == tuple) else x
                        if params['condition_dependant_latent']:
                            if net.flow is None:
                                adjusted_forecast, mu, log_var, cond_mu, cond_log_var = net(x, condition = cond,  sample_size = params['training_sample_size'])
                                loss, MSE, KLD = criterion(adjusted_forecast, y.unsqueeze(0).expand_as(adjusted_forecast) ,mu, log_var, cond_mu = cond_mu, cond_log_var = cond_log_var ,beta = beta, mask = m, return_ind_loss=True )
                            else:
                                adjusted_forecast, mu, log_var, cond_emb = net(x, condition = cond,  sample_size = params['training_sample_size'])
                                loss, MSE, KLD = criterion(adjusted_forecast, y.unsqueeze(0).expand_as(adjusted_forecast) ,mu, log_var, cond_mu = cond_emb, cond_log_var = None ,beta = beta, mask = m, return_ind_loss=True, normalized_flow = net.flow )
                            
                        else:
                            adjusted_forecast, mu, log_var = net(x, condition = cond,  sample_size = params['training_sample_size'])
                            
                            loss, MSE, KLD = criterion(adjusted_forecast, y.unsqueeze(0).expand_as(adjusted_forecast) ,mu, log_var,beta = beta, mask = m, return_ind_loss=True, normalized_flow = net.flow  )
                            

                        if params['Frobenius_norm_weight'] is not None:
                            FL = Frobenius_loss(adjusted_forecast.mean(0), y)
                            loss = loss + FL
                            batch_loss_FLD_validation += FL.item()

                        batch_loss_validation += loss.item()
                        batch_loss_MSE_validation += MSE.item()
                
                epoch_loss_validation.append(batch_loss_validation / num_batches_val)
                epoch_MSE_validation.append(batch_loss_MSE_validation / num_batches_val)
                epoch_FLD_validation.append(batch_loss_FLD_validation / num_batches_val)


                if test_year < year_max + 1:
                    nameSave = f"MODEL_1960-{test_year - n_validation_years -1}"
                else:
                    nameSave = f"MODEL_final_1960-{year_max  - n_validation_years}"

                if epoch == 0:
                    best_valScore = epoch_MSE_validation[-1]
                    earlystopping_counter = 0

                    torch.save( net.state_dict(), results_dir + '/Checkpoints/' + nameSave + f"_epoch_{epoch + 1 }.pth")

                else:
                    if epoch_MSE_validation[-1] < best_valScore - ( 0.02 * best_valScore):  
                        best_valScore = epoch_MSE_validation[-1]
                        earlystopping_counter = 0

                        saved_model = glob.glob(results_dir + '/Checkpoints/' + nameSave + "*.pth")
                        if len(saved_model) > 0:
                            for link in saved_model:
                                os.remove(link)
                        torch.save( net.state_dict(), results_dir + '/Checkpoints/' + nameSave + f"_epoch_{epoch + 1 }.pth")
                        Early_stop = False
                    else:
                        if params['earlystoppingbuffer'] is not None:
                            earlystopping_counter += 1
                            if (earlystopping_counter >= params['earlystoppingbuffer']) and (epoch >= 15 ):  # want to train for at least 20 epochs
                                print(
                                    f"Stopping early --> epoch val MSE score {epoch_MSE_validation[-1]} has not decreased over {params['earlystoppingbuffer']} epochs compared to best {best_valScore} ")
                                with open(Path(results_dir, "training_parameters.txt"), 'a') as f:
                                    f.write(f"\n Test year {test_year}, stopping early at {epoch + 1} --> epoch val MSE score {epoch_MSE_validation[-1]} has not decreased over {params['earlystoppingbuffer']} epochs compared to best {best_valScore}\n")
                                Early_stop = True
                                break

            del train_set, validation_set, dataloader, dataloader_val, adjusted_forecast, x, y , m, criterion, loss
            gc.collect()

            fig, ax = plt.subplots(1,1, figsize=(8,5))
            ax.plot(np.arange(1,len(epoch_loss)+1), epoch_loss, label = 'Epoch loss train', color = 'tab:blue')
            ax.plot(np.arange(1,len(epoch_MSE)+1), epoch_MSE, linestyle = 'dashed', label = 'Epoch MSE train', color = 'tab:blue')
            ax.plot(np.arange(1,len(epoch_loss_validation)+1), epoch_loss_validation, label = 'Epoch loss validation', color = 'tab:orange')
            ax.plot(np.arange(1,len(epoch_MSE_validation)+1), epoch_MSE_validation, linestyle = 'dashed', label = 'Epoch MSE validation', color = 'tab:orange')
            ax.set_title(f'Train/Val Loss \n best val MSE : {best_valScore} ') ###
            if params['Frobenius_norm_weight'] is not None:
                ax.plot(np.arange(1,len(epoch_FLD)+1), epoch_FLD, linestyle = 'dotted', alpha = 0.5, label = 'Epoch FLD train')
                ax.plot(np.arange(1,len(epoch_FLD_validation)+1), epoch_FLD_validation, linestyle = 'dotted', alpha = 0.5, label = 'Epoch FLD validation')
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            plt.show()
            plt.savefig(results_dir+f'/Figures/train_loss_1960-{test_year - n_validation_years -1}.png')
            plt.close()

            # EVALUATE MODEL
            ##################################################################################################################################
            ####### time inclusion
            if test_year < year_max + 1:

                if Early_stop:
                    net.load_state_dict(torch.load(glob.glob(results_dir + '/Checkpoints/' + f"MODEL_1960-{test_year - 1 - n_validation_years}*.pth")[0], map_location=torch.device('cpu')))
                    print('Loading the best checkpoint model ...')
                    net.to(device)

                ds_test = nanremover.sample(ds_test, mode = 'Eval')  ## PG: Sample the test data at the common locations
                obs_test = nanremover.sample(obs_test)

                ##################################################################################################################################

                test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)
                test_years_list = np.arange(1, ds_test.shape[0] + 1)  ## PG: Extract the number of years as well 
                test_set = XArrayDataset(ds_test, obs_test, lead_time = lead_time,  time_features=time_features,  in_memory=False, aligned = True, year_max = year_max, boosted_ensemble_size = params['boosted_ensemble_size'])
                # dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False)
                criterion_test =  WeightedMSE(weights=weights_, device=device, reduction='mean')
                test_results = np.zeros_like(xr.concat([ds_test for _ in range(params['boosted_ensemble_size'])], dim = 'ensembles').values)
                ds_test = ds_test.rename({'ensembles' : 'batch'})

                if 'ensembles' in ds_test.dims:
                    test_loss = np.zeros(shape=(ds_test.shape[0], ds_test.shape[1], ds_test.shape[2]))
                else:
                    test_loss = np.zeros(shape=(ds_test.shape[0], ds_test.shape[1]))
                
                for i, (x, target) in enumerate(test_set):          
                        year_idx, lead_time_list_idx = np.divmod(i, len(test_lead_time_list))
                        lead_time_idx = test_lead_time_list[lead_time_list_idx] - 1
                        net.eval()
                        with torch.no_grad():
                            if (type(x) == list) or (type(x) == tuple):
                                test_raw = (x[0].to(device), x[1].to(device))
                                test_raw = (test_raw[0],test_raw[1].unsqueeze(-2).expand(test_raw[0].shape[0], test_raw[1].shape[-1]))
                            else:
                                test_raw = x.to(device)
                                # test_raw = torch.flatten(test_raw, start_dim= 0 , end_dim=1)
                            if (type(target) == list) or (type(target) == tuple):
                                test_obs, m = (target[0].to(device).unsqueeze(0), target[1].to(device).unsqueeze(0))
                            else:
                                test_obs = target.to(device).unsqueeze(0)
                                m = None
                            if conditional_embedding:
                                    cond = ds_test_conds[i].type_as(test_obs).to(device)
                            else:
                                    cond = None
                            
                            sample_size = test_raw[0].shape[0] if (type(test_raw) == list) or (type(test_raw) == tuple) else test_raw.shape[0]
                            if params['non_random_decoder_initialization'] is False:
                                z =  Normal(torch.zeros(net.latent_size), torch.ones(( net.latent_size))).rsample(sample_shape=(params['boosted_ensemble_size']* sample_size,)).to(device)

                            else:
                                if params['condition_dependant_latent']:
                                    _, _, _, cond_mu, cond_log_var = net(test_raw, condition = cond, sample_size = 1)
                                    cond_var = torch.exp(cond_log_var) + 1e-4
                                    z =  Normal(cond_mu, torch.sqrt(cond_var)).rsample(sample_shape=(params['boosted_ensemble_size'] * sample_size,)).squeeze().to(device)
                                    
                                else:
                                    _, mu, log_var = net(test_raw, condition = cond, sample_size = 1)
                                    samples = net.sample(mu, log_var, 100 )
                                    # var = torch.exp(log_var) + 1e-4
                                    z =  Normal(torch.mean(samples, (0,1)), torch.std(samples, (0,1))).rsample(sample_shape=(params['boosted_ensemble_size'] * sample_size,)).to(device)
                                    # z = torch.unflatten(z, dim = 0, sizes = (-1,len(ensemble_list)))
                                    
                            ### cut from above
                            
                            if params['prior_flow'] is not None:

                                    if params['condition_dependant_latent']:
                                        cond_embedded = net.embedding(cond.to(device))
                                        cond_embedded = cond_embedded.expand((z.shape[0], net.embedding_size))
                                    else:
                                        cond_embedded = None
                                    z,_ = net.flow.inverse(z, condition = cond_embedded)
                                    
                            if all([params['time_features'] is not None, params['append_mode'] != 1]):
                                z = torch.unflatten(z, dim = 0, sizes = (-1,len(ensemble_list)))
                                z = torch.cat([z, test_raw[1].unsqueeze(0).expand((params['boosted_ensemble_size'], *test_raw[1].shape))], dim=-1)
                                z = torch.flatten(z, start_dim = 0, end_dim = 1)

                            if all([ conditional_embedding is True,  params['condemb_to_decoder']]) :
                                cond_embedded = net.embedding(cond.to(device))
                                if all([params['condition_dependant_latent'], params['prior_flow'] is None]):
                                    cond_embedded = net.condition_mu(cond_embedded.flatten(start_dim=1))

                                cond_embedded = cond_embedded.expand((z.shape[0], net.embedding_size))
                                z = torch.cat([z, cond_embedded], dim=-1)
                                                                  
                            target = torch.mean(test_raw[0], 0)  if (type(x) == list) or (type(x) == tuple) else  torch.mean(test_raw, 0)
                            
                            
                            out = net.decoder(z)
                            test_adjusted =   out.unsqueeze(-2)     
                            loss = criterion_test(torch.mean(test_adjusted, 0), target)

                            test_results[year_idx,lead_time_idx, ] = test_adjusted.to(torch.device('cpu')).numpy()
                            test_loss[year_idx, lead_time_idx] = loss.item() 
                del  test_set , test_raw, test_obs, x, target, m,  test_adjusted , ds_test, obs_test,
                gc.collect()

                reverse_preprocessing_pipeline =  ds_pipeline
                ##################################################################################################################################
                test_results_upsampled = nanremover.to_map(test_results)  ## PG: If the output is spatially flat, write back to maps
                test_results_untransformed = reverse_preprocessing_pipeline.inverse_transform(test_results_upsampled.values) ## PG: Check preprocessing.AnomaliesScaler for changes
                result = xr.DataArray(test_results_untransformed, test_results_upsampled.coords, test_results_upsampled.dims, name='nn_adjusted')
                ##################################################################################################################################
                # Store results as NetCDF            
                result.to_netcdf(path=Path(results_dir, f'nn_adjusted_{test_year}-{year_max}_{run+1}.nc', mode='w'))
                del   test_results, test_results_untransformed
                gc.collect()



                del result, net, optimizer
                gc.collect()
                torch.cuda.empty_cache() 
                torch.cuda.synchronize()   





if __name__ == "__main__":

    var = 'tas'
    start_test_years = [2026]
    n_runs = 1  # number of training runs
    n_validation_years = 5
    out_dir_x = '/XXXX/output'

    params = {
        "model": Autoencoder,
        "hidden_dims": [[3000,3000, 3000, 3000, 3000, 500], [3000, 3000, 3000, 3000,3000]],
        'condition_embedding_size' : [3000,3000, 3000, 3000, 3000, 2], 
        'condition_type' : 'cross_ensemble', # 'ensemble_mean' or 'climatology' or 'cross_ensemble'
        'condemb_to_decoder' : True, 
        'min_posterior_variance' :  None, #np.array([0.25]),
        'condition_dependant_latent' : False,
        'prior_flow' : None, #{'type' : RealNVP, 'num_layers' : 5},
        'full_conditioning' : False,
        'boosted_ensemble_size' : 50,
        'training_sample_size' : 1, 
        'non_random_decoder_initialization' : False,
        'beta' : dict(start = 0, end =0.1, num_epoch_warmup = 10,  num_epochs_hold = None) ,
        'time_features': None, 
        'ensemble_list' : np.arange(0,1), #np.arange(1,21)#[f'r{e}i1p2f1' for e in range(1,21,1)] ## PG
        'ensemble_mode' : 'LE',
        "epochs": 100,
        "batch_size": 100,
        "batch_normalization": False,
        "dropout_rate": 0,
        "L2_reg": 0,
        "append_mode": 3,
        "optimizer": torch.optim.Adam,
        "lr": 0.0001 ,
        'lr_scheduler' : True,
        'num_warmup_epchs' : 5,
        'min_lr' : 0,
        'loss_reduction' : 'mean' , # mean or sum
        'Frobenius_norm_weight' : None,
        'earlystoppingbuffer' : None, ## buffer number
        'equal_weights' : True,
        'version' : 2 ### 0, 1 , 2 
    }


    
    out_dir_x  = f'{out_dir_x}/{var}/SOM-FFN/results/{params["model"].__name__}/run_set_final_historical_long'
    out_dir = resolve_output_address(out_dir_x, params, n_validation_years, start_test_years )


    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir + '/Figures').mkdir(parents=True, exist_ok=True)
    Path(out_dir + '/Checkpoints').mkdir(parents=True, exist_ok=True)

    try:
        run_training(params,var,  n_validation_years = n_validation_years,n_runs=n_runs, results_dir=out_dir, numpy_seed=1, torch_seed=1, start_test_years = start_test_years)
        print(f'Output dir: {out_dir}')
        print('Training done.')
    except Exception as e:
        import shutil
        Path(out_dir_x + '/failed_cases').mkdir(parents=True, exist_ok=True)
        shutil.move(out_dir, out_dir_x + '/failed_cases')
        print("Terminated due to the follwoing error:\n", e)
        raise  # 