class plt_module(object):

    def __init__(self):
        
        self.dict_event_plt = {
                             'nino' : {
                                       'mean_vmin' : -1.,
                                       'mean_vmax' : 1.,
                                       'std_vmin'  : 0,
                                       'std_vmax'  : 7,
                                      },    
                             'nina' : {
                                       'mean_vmin' : -1.,
                                       'mean_vmax' : 1.,
                                       'std_vmin'  : 0,
                                       'std_vmax'  : 7,
                                      },    
                             '0.99' : {
                                       'mean_vmin' : -15,
                                       'mean_vmax' : 15,
                                      },    
                              }

        self.dict_hist_plt = {
                                 'population' : {
                                                  'color'     : 'k',   
                                                  'label'     : 'population',
                                                  'histtype'  : 'stepfilled',  
                                                  'alpha_hist':  0.5,
                                                  'linestyle' : '-',
                                                  'linewidth' : 2, 
                                                  'facecolor' : 'k',
                                                  'marker'    : 'x',
                                                  'alpha'     : 1,
                                                  's'         : 80,
                                                },
                                 'train_sample' : {
                                                  'color'     : 'blue',
                                                  'linestyle' : 'dashed',
                                                  'linewidth' : 2, 
                                                  'label'     : 'train_sample',
                                                  'histtype'  : 'step',  
                                                  'facecolor' : 'blue',
                                                  'marker'    : 'x',
                                                  'alpha'     : 1,
                                                  's'         : 80,
                                                  },

                                 'hist_VAE1'     : {
                                                  'color'     : 'green',
                                                  'linestyle' : '-',
                                                  'linewidth' : 1, 
                                                  'label'     : 'VAE 2.5$\sigma$ + DS',
                                                  'histtype'  : 'step',  
                                                  'facecolor' : 'w',
                                                  'marker'    : 'o',
                                                  'alpha'     : 1,
                                                  },
                                 'hist_VAE2'     : {
                                                  'color'     : 'green',
                                                  'linestyle' : '-',
                                                  'linewidth' : 1, 
                                                  'label'     : 'VAE normal based on train + DS',
                                                  'histtype'  : 'step',  
                                                  'facecolor' : 'w',
                                                  'marker'    : 'o',
                                                  'alpha'     : 1,
                                                  },
                                 'hist_VAE3'     : {
                                                  'color'     : 'green',
                                                  'linestyle' : 'dashed',
                                                  'linewidth' : 1, 
                                                  'label'     : 'VAE 1$\sigma$  + DS',
                                                  'histtype'  : 'step',  
                                                  'facecolor' : 'green',
                                                  'marker'    : 'o',
                                                  'alpha'     : 0.5,
                                                  },
                                 'hist_VAE4'     : {
                                                  'color'     : 'green',
                                                  'linestyle' : 'dotted',
                                                  'linewidth' : 1, 
                                                  'label'     : 'VAE 1$\sigma$',
                                                  'histtype'  : 'step',  
                                                  'facecolor' : 'w',
                                                  'marker'    : 'o',
                                                  'alpha'     : 1,
                                                  },
                                 'hist_benchmark'     : {
                                                  'color'     : 'tab:red',
                                                  'linestyle' : 'dotted',
                                                  'linewidth' : 1, 
                                                  'label'     : 'Gaussian benchmark',
                                                  'histtype'  : 'step',  
                                                  'facecolor' : 'w',
                                                  'marker'    : 'o',
                                                  'alpha'     : 1,
                                                  }
                            }
