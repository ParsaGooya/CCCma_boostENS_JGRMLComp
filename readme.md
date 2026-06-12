# Toward Generative Machine Learning for Boosting Ensembles of Climate Simulations

**Parsa Gooya<sup>1</sup>, Reinel Sospedra-Alfonso<sup>1</sup>, and Johannes Exenberger<sup>2,3</sup>**

<sup>1</sup> Canadian Centre for Climate Modeling and Analysis, Environment and Climate Change Canada, Victoria, British Columbia, Canada  
<sup>2</sup> Vienna University of Technology  
<sup>3</sup> Institute of Software Technology and Artificial Intelligence, Graz University of Technology  

*Manuscript submitted to JGR: Machine Learning and Computation journal.*

---

## Content

This repository contains scripts for training and running a simple MLP-based Variational Autoencoder (VAE) models, as well as code for plotting the results as presented in the manuscript above.

- **`run_training_BVAE_historical.py`**
  Contain the main code for training the cVAE models on historical and ssp245 data.  
  The code supports both standard and conditional VAEs, with simple or condition-dependent prior distributions.  
  The implementation also allows the integration of normalizing flows for the VAE prior.

- **`predict_BVAE_historical.py`** 
  Generate large ensembles using the trained model in inference mode.  
  Different sampling strategies for the latent space, as well as various formulation/sampling of the decoder noise are implemented.

- **`figures_paper/` directory**  
  Includes notebooks and scripts used for analysis of results and generation of the manuscript figures.

Note

This repository contains research-grade code developed to support the experiments and figures presented in the accompanying manuscript. The code is provided as-is and is not actively maintained.

A dedicated post-processing software package is currently under development and will be released separately upon completion. To improve readability and reproducibility, the code included here was cleaned and simplified at the time of publication, with unused variables, experimental options, and project-specific utilities removed where possible.

If you encounter issues while running the code or have questions regarding the implementation, please contact Parsa Gooya at parsa.gooya@ec.gc.ca.

---

## Copyright

© Environment and Climate Change Canada and the contributors, 2025. All rights reserved.  
For inquiries, contact [parsa.gooya@ec.gc.ca](mailto:parsa.gooya@ec.gc.ca).  
Do not copy or reproduce without proper citation.