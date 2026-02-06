import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature    
from pathlib import Path
from scipy.stats import skew

##
def plot_ts(list_ds,
            dict_ds,
            dict_plot,
            xdim='time',
            var='tas',
            xlabel='',
            ylabel='',
            bbox=(.87,.5,.5,.5),
            text_dict = None, 
            figsize=(6,3),
            legend = True,
            title='',
            ylim = None,
            dir_name=None,
            file_name=None,
            show_model_label = True,
            show=False,
            save=False):
    '''
    plot time series
    '''
    plt.close()
    
    plt.figure(figsize=figsize)

    plt.title(title)

    for ids in list_ds:
        
        if all(['VAE' in ids,show_model_label is False]):
            label = None
        else:
            label = dict_plot[ids]['label']
        da = dict_ds[ids][var]

        if xdim == 'year':
            xx = da.year.values + (da.time.values - 0.5)/12        
        else:
            xx = da[xdim]
            plt.xticks(xx)
        
        plt.plot(xx,
                 da,
                 color=dict_plot[ids]['color'],
                 linestyle = dict_plot[ids]['linestyle'],
                 linewidth = dict_plot[ids]['linewidth'],
                 label=label,)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    if legend:
        plt.legend(loc='best',
               bbox_to_anchor=bbox,
               handlelength=1,
               ncol=1,
               frameon=False) 
    if ylim is not None:
        plt.ylim(ylim )

    if text_dict is not None:

        plt.text(    text_dict['x'], text_dict['y'],               # relative position
        text_dict['text'], 
        fontsize = text_dict['fontsize'],
        transform=plt.gca().transAxes)  # use axes coordinates
        
    if save:
        Path(dir_name).mkdir(parents=True,
                                 exist_ok=True)
        plt.savefig(f'{dir_name}/{file_name}.png',
                    bbox_inches='tight',
                    dpi=300)
    if show:
        plt.show()
    else:
        plt.close()    

##
def plot_qq(list_data,
            dict_data,
            idata_ref='train_sample',
            dict_plt=None,
            figsize=(6,3),
            box=None,
            xlabel='Reference Temperature Anomalies ($^\circ$C)',
            ylabel='Temperature Anomalies ($^\circ$C)',
            title='Quantile-Quantile Plot',
            bbox=(.87,.5,.5,.5),
            legend = True,
            fontsize = 15, 
            text_dict = None,
            add_error = False,
            dir_name=None,
            file_name=None,
            scatter = True,
       show=False,
       save=False):
    '''
    scatterplot quantiles
    '''
    plt.figure(figsize=figsize)
    if box == None:
        box = np.array([[-2,-2],
                        [2,2]])
    else:
        box = np.array(box)

    pnt_min = box[0,:]
    pnt_max = box[1,:]

    ax = plt.gca()
    ax.axline(pnt_min,
              pnt_max,
              color='tab:blue' if idata_ref == 'train_sample'  else 'k',
              linestyle = 'dotted')
    ax.set_xlim(pnt_min[0],pnt_max[0])
    ax.set_ylim(pnt_min[1],pnt_max[1])
    
    ax.set_title(title, fontsize = fontsize)
    ax.set_xlabel(xlabel, fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)

    # var = list(dict_data[idata_ref].data_vars)[0]
    
    da_x = dict_data[idata_ref]

    for idata in [key for key in list_data if key != idata_ref]:

        da_y = dict_data[idata]
        try:
            alpha = dict_plt[idata]['alpha']
        except:
            alpha = 1
        try:
            s = dict_plt[idata]['s']
        except:
            s = None
        if add_error:
            err = f' ({str(np.round(np.sqrt(((da_x- da_y)**2).mean()),2))})'
        else:
            err = ''
        if scatter:
            ax.scatter(da_x,
                   da_y,
                   color=dict_plt[idata]['color'],
                   facecolor=dict_plt[idata]['facecolor'],
                   marker=dict_plt[idata]['marker'],
                   label=dict_plt[idata]['label'] + err,
                    linestyle = dict_plt[idata]['linestyle'],
                   alpha = alpha,
                   s = s) 
        else:
            ax.plot(da_x,
                   da_y,
                   color=dict_plt[idata]['color'],
                   linestyle = dict_plt[idata]['linestyle'],
                #    facecolor=dict_plt[idata]['facecolor'],
                #    marker=dict_plt[idata]['marker'],
                   label=dict_plt[idata]['label'] + err,
                   alpha = alpha)             
    if legend:
        plt.legend(loc='best',
               bbox_to_anchor=bbox,
               handlelength=1,
               ncol=1,
               fontsize = fontsize,
               frameon=False) 
    if text_dict is not None:

        plt.text(    text_dict['x'], text_dict['y'],               # relative position
        text_dict['text'], 
                fontsize = text_dict['fontsize'],
        transform=plt.gca().transAxes)  # use axes coordinates

    plt.rc('xtick',labelsize=fontsize)
    plt.rc('ytick',labelsize=fontsize)
        
    if save:
        Path(dir_name).mkdir(parents=True,
                                 exist_ok=True)
        plt.savefig(f'{dir_name}/{file_name}.png',
                    bbox_inches='tight',
                    dpi=300)
    if show:
        plt.show()
    else:
        plt.close()    


# def plot_qq(list_data,
#             dict_data,
#             idata_ref='train_sample',
#             dict_plt=None,
#             box=None,
#             xlabel='Reference Temperature Anomalies ($^\circ$C)',
#             ylabel='Temperature Anomalies ($^\circ$C)',
#             title='Quantile-Quantile Plot',
#             bbox=(.87,.5,.5,.5),
#             dir_name=None,
#             file_name=None,
#        show=False,
#        save=False):
#     '''
#     scatterplot quantiles
#     '''
    
#     if box == None:
#         box = np.array([[-2,-2],
#                         [2,2]])
#     else:
#         box = np.array(box)

#     pnt_min = box[0,:]
#     pnt_max = box[1,:]
    
#     ax = plt.gca()
#     ax.axline(pnt_min,
#               pnt_max,
#               color='k')
#     ax.set_xlim(pnt_min[0],pnt_max[0])
#     ax.set_ylim(pnt_min[1],pnt_max[1])
    
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     var = list(dict_data[idata_ref].data_vars)[0]
    
#     da_x = dict_data[idata_ref][var].values

#     for idata in [key for key in list_data if key != idata_ref]:

#         da_y = dict_data[idata][var].values
        
#         ax.scatter(da_x,
#                    da_y,
#                    color=dict_plt[idata]['color'],
#                    facecolor=dict_plt[idata]['facecolor'],
#                    marker=dict_plt[idata]['marker'],
#                    label=dict_plt[idata]['label'])  

#     plt.legend(loc='best',
#                bbox_to_anchor=bbox,
#                handlelength=1,
#                ncol=1,
#                frameon=False) 

        
#     if save:
#         Path(dir_name).mkdir(parents=True,
#                                  exist_ok=True)
#         plt.savefig(f'{dir_name}/{file_name}.png',
#                     bbox_inches='tight',
#                     dpi=300)
#     if show:
#         plt.show()
#     else:
#         plt.close()    
        
# def plot_qq(da_x,
#             da_y,
#             da_z=None,
#             box=None,
#             dict_y={'color':'b',
#                     'facecolor':'w',
#                     'marker':'o',
#                     'label':'train'},
#             dict_z={'color':'r',
#                     'facecolor':'w',
#                     'marker':'o',
#                     'label':'toy_VAE1'},
#             dir_name=None,
#             file_name=None,
#        show=False,
#        save=False):
#     '''
#     scatterplot quantiles
#     '''
    
#     if box == None:
#         box = np.array([[-2,-2],
#                         [2,2]])
#     else:
#         box = np.array(box)

#     pnt_min = box[0,:]
#     pnt_max = box[1,:]
    
#     ax = plt.gca()
#     ax.axline(pnt_min,
#               pnt_max,
#               color='k')
#     ax.set_xlim(pnt_min[0],pnt_max[0])
#     ax.set_ylim(pnt_min[1],pnt_max[1])
#     ax.scatter(da_x,
#                da_y,
#                color=dict_y['color'],
#                facecolor=dict_y['facecolor'],
#                marker=dict_y['marker'],
#                label=dict_y['label'])  
#     if da_z is not None:
#         ax.scatter(da_x,
#                    da_z,
#                    color=dict_z['color'],
#                    facecolor=dict_z['facecolor'],
#                    marker=dict_z['marker'],
#                    label=dict_z['label'])   
        
#     if save:
#         Path(dir_name).mkdir(parents=True,
#                                  exist_ok=True)
#         plt.savefig(f'{dir_name}/{file_name}.png',
#                     bbox_inches='tight',
#                     dpi=300)
#     if show:
#         plt.show()
#     else:
#         plt.close()    


##

##

def plot_hist(list_data,
              dict_data,
              dict_plt,
              figsize = (8,6),
              var='tas',
              seasons_dict = {'DJF' : [12,1,2], 'JJA' : [6,7,8], 'D' : [12],  'A' : [8], 'Full' : np.arange(1,13)}, 
              season='Full',
              mean_season = False,
              ylog=True,
              nbins=20,
              density=True,
              fontsize = 10,
              xmin=-3,
              xmax=3,
              xlabel='',
              ylabel='Density',
              skewness = False,
              title='Histogram',
              text_dict = None, 
              bbox=(.86,.5,.5,.5),
              legend_bool = None,
              fig_dir=None,
              fig_name=None,
              show=False,
              save=False):
    '''
    plot histogram
    '''
    plt.figure(figsize = figsize)
    if type(legend_bool) in (list, tuple):
        assert len(legend_bool) == len(list_data)

    for ind, idata in enumerate(list_data):

        data = dict_data[idata][var].sel(time=seasons_dict[season])
        if mean_season:
            data = data.mean('time')
        data = data.to_numpy().flatten()            
        try:
            alpha = dict_plt[idata]['alpha_hist']
        except:
            alpha = 1
        if type(legend_bool) in (list, tuple):
            if legend_bool[ind]:
                label = dict_plt[idata]['label']
            else:
                label = None
        else:
            label = dict_plt[idata]['label']
        
        if skewness:
            skn = skew(data, bias=False)
            skn = f' ({str(np.round(skn,2))})'
        else:
            skn = ''
        
        if label is not None:
            label = label  + skn

        (counts,
         bins,
         patches) = plt.hist(data,
                             bins=nbins,
                             edgecolor=dict_plt[idata]['color'],
                             color=dict_plt[idata]['color'],
                             histtype=dict_plt[idata]['histtype'],
                             linestyle=dict_plt[idata]['linestyle'],
                             log=ylog,
                             density=True,
                             linewidth=2,
                             label=label,
                             alpha = alpha)
    if legend_bool is not None:
        plt.legend(loc='best',
                bbox_to_anchor=bbox,
                fontsize = 12,
                handlelength=1,
                ncol=1,
                frameon=False)         

    plt.xlim((xmin,xmax))
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize= fontsize)
    plt.title(title)
    plt.rc('xtick',labelsize=fontsize)
    plt.rc('ytick',labelsize=fontsize)
    if season != 'Full':
        if text_dict is None:
            plt.text(    0.05, 0.95,               # relative position
            season,    fontsize = 12,
            transform=plt.gca().transAxes)
        else:
            plt.text(    text_dict['x'], text_dict['y'],               # relative position
            season + f' {text_dict["text"]}', fontsize = text_dict['fontsize'],
            transform=plt.gca().transAxes)    
    elif text_dict is not None:
            plt.text(    text_dict['x'], text_dict['y'],               # relative position
            text_dict["text"],   fontsize = text_dict['fontsize'],
            transform=plt.gca().transAxes)   
  # use axes coordinates


    if save:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{fig_dir}/{fig_name}',
                    bbox_inches='tight',
                    dpi=300)
    if show:
        plt.show()   
    else:
        plt.ioff()

##

def plot_spectra(dict_spectra,
                 list_data,
                 dict_plt_data = None,
                 show_ensemble=True,
                 figsize=(12,6),
                 dict_plt={'title'   : 'Power Spectra',
                           'xlabel'  : 'Wavelength (km)',
                           'ylabel'  : 'Log$_2$ Power',                       
                           'legend' :
                           {
                            'ncol': 1,
                            'pos' : (.95,1.),
                            'loc' : 'best',
                           }
                           },
                 fig_dir=None,
                 fig_name=None,
                 show=False,
                 save=False):
    '''
    plot spectra 
    '''
    
    fig, ax = plt.subplots(1,1,
                           figsize=figsize)
    
    
    for idata in list_data:

        data_pwr = dict_spectra[idata]['power']
        data_wav = dict_spectra[idata]['wavelength']
    
        pwr_min = np.min(data_pwr,axis=0)
        pwr_max = np.max(data_pwr,axis=0)

        if show_ensemble:
            ax.fill_between(data_wav[0,:],
                            pwr_min,
                            pwr_max,
                            alpha=.1)
        if dict_plt_data is not None:
            label =  dict_plt_data[idata]["label"]
        else:
            label =  idata
        plt.plot(data_wav[0,:],
                 np.mean(data_pwr,axis=0),
                 label=f'{label} ({len(data_pwr[:,0])} members)')

    plt.gca().invert_xaxis()        

    plt.legend(loc=dict_plt['legend']['loc'],
               bbox_to_anchor=dict_plt['legend']['pos'],
               ncol=dict_plt['legend']['ncol'],
               frameon=False)
    
    ax.set_title(dict_plt['title'])
    ax.set_xlabel(dict_plt['xlabel'])
    ax.set_ylabel(dict_plt['ylabel'])

    if save:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{fig_dir}/{fig_name}',
                    bbox_inches='tight',
                    dpi=300)
    if show:
        plt.show()   
    else:
        plt.ioff()
    

    

##
def plot_map(ds,
             lat=None,
             lon=None,
             polar_stereo=False, 
             central_longitude=180,
             lat_lims=[50,90],
             gridlines=False,
             cmap='RdBu_r', #mpl.cm.RdBu_r,
             vmin=-1.5,
             vmax=1.5,
             vals=None,
             nvals=10,
             cbar=False,
             cbar_label='',
             cbar_integer=False,
             cbar_extend='both',
             ticks_rotation=0,
             ticks_centered=False,
             title=None,
             fnt_size=12,
             coastlines = True,
             figsize=None,#(8,3),
             fig_dir=None,
             fig_name=None,
             show=False,
             save=False,
             **kwargs): 
    '''
    plot geographic map
    '''
    
    mpl.rcParams.update({'font.size': fnt_size})

    if central_longitude == 0: # remove white line at central longitude
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)
            
    if lat is None:
        lat = ds.lat
    if lon is None:
        lon = ds.lon

    crs = ccrs.PlateCarree(central_longitude=central_longitude)    
    if polar_stereo:
        crs = ccrs.NorthPolarStereo()    
        if max(lat_lims) < 0:
            crs = ccrs.SouthPolarStereo()    
        
    plt.close()
    
    fig, ax = plt.subplots(nrows=1,
                           ncols=1, 
                           figsize=figsize, 
                           subplot_kw={'projection' : crs})                           
    
    if vals is not None:
        nvals = len(vals) - 1
        
    if vals is None:
        scale = (vmax-vmin)/float(nvals)
        vals = vmin + (vmax-vmin)*np.arange(nvals+1)/float(nvals)

            
    norm = mpl.colors.BoundaryNorm(vals,
                                   plt.cm.get_cmap(cmap).N)
    
    axis = ax
    if gridlines:
        axis.gridlines(draw_labels=False)
    if polar_stereo:
        polarCentral_set_latlim(lat_lims, axis)
        if coastlines:
            axis.add_feature(cfeature.NaturalEarthFeature('physical',
                                                      'land', 
                                                      '50m',
                                                      # edgecolor='face',
                                                      facecolor='grey'))
        
    clevs = np.linspace(vmin,
                        vmax,
                        nvals+1)
    
    fill = ds.plot.contourf(ax=ax,
                             levels=clevs,
                             cmap=cmap,
                             add_colorbar=False,
                             transform=ccrs.PlateCarree())
    if coastlines:
        axis.coastlines()
    axis.set_title(title,
                   fontsize=fnt_size)

    if cbar:
        clb_x = 0.055 #0.095 
        clb_y = 0.1
        clb_w = 0.9 #0.8
        clb_h = 0.04
        if polar_stereo:
            clb_y = 0.0
        cax = plt.axes([clb_x, # left
                        clb_y, # bottom
                        clb_w, # width
                        clb_h])# height

        cb = mpl.colorbar.ColorbarBase(ax=cax,
                                       cmap=plt.cm.get_cmap(cmap),
                                       # cmap=cmap,
                                       norm=norm,
                                       spacing='uniform',
                                       orientation='horizontal',
                                       extend=cbar_extend,
                                       ticks=vals)

        ticks_labels = np.round(vals,3)
        if cbar_integer:
            ticks_labels = [int(np.ceil(x)) for x in vals]
            
        cax.tick_params(labelsize=fnt_size-2)
        cb.set_ticks(ticks=vals, 
                     rotation=ticks_rotation,
                     labels=ticks_labels)
            
        if ticks_centered:
            cb.set_ticks([x+((vals[1]-vals[0])/2.) for x in vals[:-1]])
            cb.set_ticklabels(ticks_labels[:-1])

        cb.set_label(label=cbar_label,
                     size=fnt_size-2) 
       
    fig.tight_layout()
    if save:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{fig_dir}/{fig_name}',
                    bbox_inches='tight',
                    dpi=300)
    if show:
        plt.show()   
    else:
        plt.ioff()

