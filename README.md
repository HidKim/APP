# Python Code for Augmented Permanental Process 
This library provides augmented permanental process (APP) implemented in Tensorflow. APP provides a scalable Bayesian framework for estimating point process intensity *as a function of covariates*, with the assumption that covariates are given at every point in the observation domain. For details, see our NeurIPS2022 paper [1].

The code was tested on Python 3.7.2, Tensorflow 2.2.0, and qmcpy 1.3.

# Installation
To install latest version:
```
pip install git+https://github.com/HidKim/APP
```

# Basic Usage
Import APP class:
```
from HidKim_APP import augmented_permanental_process as APP
```
Initialize APP:
```
model = APP(kernel='Gaussian', eq_kernel='RFM',  
            eq_kernel_options={'cov_sampler':'Sobol','n_cov':2**11,'n_dp':500,'n_rfm':500})
```
- `kernel`: *string, default='Gaussian'*. <br> 
&emsp; The kernel function for Gaussian process. Only 'Gaussian' is available now.
- `eq_kernel`:
Fit APP with data:
```
_ = model.fit(d_spk, obs_region, cov_fun, set_par=[], display=True)
```
Predict point process intensity as function of covariates:
```
z = model.predict(t, conf_int=[0.025,0.5,0.975])
```

# Reference
1. Hideaki Kim, Taichi Asami, and Hiroyuki Toda. "Fast Bayesian Estimation of Point Process Intensity as Function of Covariates", *Advances in Neural Information Processing Systems 35*, 2022.
```
@inproceedings{kim2022fastbayesian,
  title={Fast {B}ayesian Estimation of Point Process Intensity as Function of Covariates},
  author={Kim, Hideaki and Asami, Taichi and Toda, Hiroyuki},
  booktitle={Advances in Neural Information Processing Systems 35},
  year={2022}
}
``` 

# License
Released under "SOFTWARE LICENSE AGREEMENT FOR EVALUATION". Be sure to read it.

# Contact
Hideaki Kim (dedeccokim at gmail.com)