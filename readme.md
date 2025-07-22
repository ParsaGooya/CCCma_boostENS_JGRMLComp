# Toward Generative Machine Learning for Boosting Ensembles of Climate Simulations

**Parsa Gooya<sup>1</sup>, Reinel Sospedra-Alfonso<sup>1</sup>, and Johannes Exenberger<sup>2,3</sup>**

<sup>1</sup> Canadian Centre for Climate Modeling and Analysis, Environment and Climate Change Canada, Victoria, British Columbia, Canada  
<sup>2</sup> Vienna University of Technology  
<sup>3</sup> Institute of Software Technology and Artificial Intelligence, Graz University of Technology  

*Manuscript submitted to JGR: Machine Learning and Computation journal.*

---

## Content

This repository contains scripts for designing, running, and tuning hyperparameters of simple MLP-based Variational Autoencoder (VAE) models, as well as code for plotting the results as presented in the manuscript above.

- **`run_training_BVAE.py`** and **`run_training_BVAE_picontrol.py`**  
  Contain the main code for training the VAE models on toy and climate data, respectively.  
  The code supports both standard and conditional VAEs, with simple or condition-dependent prior distributions.  
  The implementation also allows the integration of normalizing flows for the VAE prior.

- **`predict_BVAE.py`** and **`predict_BVAE_picontrol.py`**  
  Generate large ensembles using the trained model in inference mode.  
  Different sampling strategies for the latent space, as well as various formulation/sampling of the decoder noise are implemented.

- **`data_gen_spherical_harmonic.py`**  
  Provides code for generating samples of arbitrary size using the toy dataset.

- **`figures_paper/` directory**  
  Includes notebooks and scripts used for analysis of results and generation of the manuscript figures.

---

## Copyright

© Environment and Climate Change Canada and the contributors, 2025. All rights reserved.  
For inquiries, contact [parsa.gooya@ec.gc.ca](mailto:parsa.gooya@ec.gc.ca).  
Do not copy or reproduce without proper citation.