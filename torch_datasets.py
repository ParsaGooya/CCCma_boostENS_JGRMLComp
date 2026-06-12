import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
import random


class XArrayDataset(Dataset):

    def __init__(self, data: xr.DataArray, 
                    target: xr.DataArray, 
                    mask=None,
                    lead_time = None,  
                    time_features=None,  
                    in_memory=True, 
                    to_device=None, 
                    aligned = False, 
                    year_max = None, 
                    max_month = None,  
                    model = 'Autoencoder', 
                    boosted_ensemble_size = None, 
                    conditional_embedding = False, 
                    cross_member_training = False):

        self.mask = mask
        self.atm_co2_features = None
        self.conditional_embedding = conditional_embedding
        self.cross_member_training = cross_member_training
        self.data = data
        self.data = self.data.stack(flattened=('year','lead_time')).transpose('flattened',...)
        self.target = target.stack(flattened=('year','month')).transpose('flattened',...)

        if self.mask is not None:
            self.data = self.data[~self.mask.flatten()]
            if aligned:
                self.target = self.target[~self.mask.flatten()]
        if len(self.data.year) > 1:
            years_in_months = (self.data.year - self.data.year[0]) * 12
        else:
            years_in_months = 0
        if aligned:
            target_idx = np.arange(len(self.data.flattened))
        else:
            target_idx = (years_in_months + self.data.lead_time - 1).to_numpy()
        self.target = self.target[target_idx,...]

        
        if lead_time is not None:
            self.data = self.data.where((self.data.lead_time >=  (lead_time - 1) * 12 + 1) & (self.data.lead_time < (lead_time *12 )+1), drop = True)
            self.target = self.target.where((self.target.month >=  (lead_time - 1) * 12 + 1) & (self.target.month < (lead_time *12 )+1), drop = True)


        if time_features is not None:

            self.use_time_features = True
            
            self.time_features_list = np.array([time_features]).flatten()
            feature_indices = {'year': 0, 'lead_time': 1, 'month_sin': 2, 'month_cos': 3, 'sin_t' : 4, 'cos_t' : 5}
            # y = self.data.year.to_numpy() / np.max(self.data.year.to_numpy())
            t =  np.arange(len(self.data.year))
            y = (self.data.year.to_numpy() + np.floor(self.data.lead_time.to_numpy()/12)) / year_max
            if max_month is not None:
                t = t + max_month
            else:
                max_month = len(self.data.year)
            lt = self.data.lead_time.to_numpy() / np.max(self.data.lead_time.to_numpy())
            # msin = np.sin(2 * np.pi * lt/12.0)
            # mcos = np.cos(2 * np.pi * lt/12.0)
            tsin = np.sin(t)
            tcos = np.cos(t)
            msin = np.sin(2 * np.pi * self.data.lead_time/12.0)
            mcos = np.cos(2 * np.pi * self.data.lead_time/12.0)
            self.time_features = np.stack([t / max_month, lt, msin, mcos, tsin, tcos], axis=1) ## remmeber I replaced y with t/max_month at the first element of the list.
            self.time_features = self.time_features[..., [feature_indices[k] for k in self.time_features_list]]




        if boosted_ensemble_size is not None:
            self.data_std = self.data.std('ensembles').rename({'lead_time' : 'month'})
            self.target = xr.concat([self.target,self.data_std  ], dim = 'channels')
            self.data = self.data.rename({'ensembles' : 'batch'})
 
        if cross_member_training:
            try:
                self.ensembles_idx = np.arange(len(self.data.batch))
            except:
                self.ensembles_idx = np.arange(len(self.data.ensembles))
            self.shuffled_ens_idx = self.ensembles_idx.copy() ### made this change to train for a different target each
            self.target = self.data[:,self.shuffled_ens_idx,...]
            self.data_unwrapped = self.data.copy()


        if 'ensembles' in self.data.dims: ## PG: if not ensemble mean:
            if model == 'cVAE':
                self.target = self.data.mean('ensembles')

            self.data = self.data.reset_index('lead_time','year').rename({'flattened':'flat'})  ## PG: Change the flattened multi-index coord with a simple coord
            self.data['flat'] = np.arange(0,len(self.data.flat))  ## PG: create new coords for the ('year','lead_time') multi-index that shows indices
            self.data = self.data.stack(flattened=('ensembles','flat')).transpose('flattened',...) ## PG: Unwrap the ensemble dim
            target_idx = self.data.flat.values 
            
            if conditional_embedding:
                self.condition_target_ids = target_idx

            if cross_member_training:
                self.target = self.target.reset_index('lead_time','year').rename({'flattened':'flat'})  ## PG: Change the flattened multi-index coord with a simple coord
                self.target['flat'] = np.arange(0,len(self.target.flat))  ## PG: create new coords for the ('year','lead_time') multi-index that shows indices
                self.target = self.target.stack(flattened=('ensembles','flat')).transpose('flattened',...) ## PG: Unwrap the ensemble dim
            else:
                self.target = self.target[target_idx,...] ## PG: Sample the target at the new unwrapped indices
            
            if self.use_time_features:
                self.time_features = self.time_features[target_idx,...] ## PG: sample time features with the same indices due to the unwrapping the ensemble dim



        if self.use_time_features:

            if  model in ['SCNN' ,'UNet2', 'UNet2_decoupled','CNN', 'CNN_mean']:
                self.time_features = np.concatenate([np.broadcast_to(self.time_features[:, ind,None, None, None],  self.data.isel(channels = 0).expand_dims('channels', axis=1).shape) for ind in range(len(time_features))] , axis = 1)
            

        

        if in_memory:
            
            self.data = torch.from_numpy(self.data.to_numpy()).float()
            self.target = torch.from_numpy(self.target.to_numpy()).float()
            if conditional_embedding: 
                self.condition_target_ids = torch.from_numpy(self.condition_target_ids) 
 

            if self.use_time_features:
                self.time_features = torch.from_numpy(self.time_features).float()


            if to_device:
                self.data.to(to_device)
                self.target.to(to_device)
                if conditional_embedding: 
                    self.condition_target_ids.to(to_device)

                if self.use_time_features:
                    self.time_features = self.time_features.to(to_device)
            
    def __getitem__(self, index):
        x = self.data[index,...]
        y = self.target[index,...]
        if self.conditional_embedding:
            idx = self.condition_target_ids[index,...]


        if torch.is_tensor(x):


            y_ = y

            if self.use_time_features: 
                t = self.time_features[index,...]
                x_ = (x,t)
            else: 
                x_ = x

            if self.conditional_embedding:
                if (type(x) == list) or (type(x) == tuple):
                    x_ = (*x_, idx)
                else:
                    x_ = (x_, idx)

            return x_, y_
        
        else:
            if self.conditional_embedding:
                idx = torch.from_numpy(idx)
            x = torch.from_numpy(x.to_numpy()).float()
            y = torch.from_numpy(y.to_numpy()).float()

            y_ = y
            
            if self.use_time_features:
                t = self.time_features[index,...]
                t = torch.from_numpy(t).float()
                x_ = (x,t)
            else:  
                x_ = x
            
            if self.conditional_embedding:
                if (type(x) == list) or (type(x) == tuple):
                    x_ = (*x_, idx)
                else:
                    x_ = (x_, idx)
                  
            return x_, y_

    def __len__(self):
        return len(self.data)
    
    def shuffle_target_ensembles(self, shuffle_idx = None, return_shuffled_idx = False):

        assert self.cross_member_training, 'cross_member_training should be on for shuffling the target ensemble members ...'
        if shuffle_idx is None:
            self.shuffled_ens_idx = self.shuffle_no_direct_swap(self.ensembles_idx, check_array =  self.shuffled_ens_idx)
        else:
            self.shuffled_ens_idx = shuffle_idx
        self.target = self.data_unwrapped[:,self.shuffled_ens_idx,...]
        if 'ensembles' in self.target.dims:
            self.target = self.target.reset_index('lead_time','year').rename({'flattened':'flat'})  ## PG: Change the flattened multi-index coord with a simple coord
            self.target['flat'] = np.arange(0,len(self.target.flat))  ## PG: create new coords for the ('year','lead_time') multi-index that shows indices
            self.target = self.target.stack(flattened=('ensembles','flat')).transpose('flattened',...) ## PG: Unwrap the ensemble dim

        if return_shuffled_idx:
            return self, self.shuffled_ens_idx
        else:
            return self

    def shuffle_no_direct_swap(self, arr, check_array = None):
        while True:
            shuffled_arr = np.copy(arr)
            np.random.shuffle(shuffled_arr)
            # Check for direct swaps
            direct_swap = False
            for i in range(len(arr)):
                if arr[i] != shuffled_arr[i]:
                    original_index = np.where(arr == shuffled_arr[i])[0][0]
                    if shuffled_arr[original_index] == arr[i]:
                        direct_swap = True
                        break
            
            if not direct_swap:
                if check_array is not None:
                    if not np.array_equal(shuffled_arr,check_array):
                        return shuffled_arr
                else:
                    return shuffled_arr

