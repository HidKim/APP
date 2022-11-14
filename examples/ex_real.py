import dill
from HidKim_APP import augmented_permanental_process as APP
from HidKim_APP import parameter_initializer as p_init
from scipy.interpolate import LinearNDInterpolator
from pylab import *


if __name__ == "__main__":

    # Real-world data used in our NeurIPS2022 paper
    # Dim of covariate is two, dim of observation region is two
    data = dill.load(open('../data/real/Bei.dill','rb'))

    # Construct covariate map with sample covariate data: data['t'] & data['cov']
    lin = LinearNDInterpolator(data['t'], data['cov'])
    fun_cov = lambda x: lin(x)
    
    # Initialize hyper-parameter of Gaussian kernel function:
    # (a,b1,b2) for k(t,t') = a * exp( -(b1*(t-t'))^2 - (b2*(t-t'))^2 )
    init_par = p_init(data['spk'], data['obs'], fun_cov(data['spk']))
    
    # Make a set of hyper-parameters to examine -> set_par
    w = array([1./3, 1./2, 1.0, 2.0, 3.0])
    a, b = meshgrid(w,w)
    z = array([ravel(a),ravel(b),ravel(b)]).T
    set_par = z * init_par
    
    # Perform intensity estimation in APP
    model = APP(kernel='Gaussian', eq_kernel='RFM',
                eq_kernel_options={'cov_sampler':'Sobol', 'n_cov':1500, 'n_rfm':100})
    _ = model.fit(data['spk'], data['obs'], fun_cov ,set_par)
    
    #
    grid_size = 100
    mesh_elev = linspace(120,160,grid_size)
    mesh_slop = linspace(0,0.32,grid_size)
    mesh_cov  = meshgrid(mesh_elev,mesh_slop)
    cov = array([ravel(mesh_cov[0]),ravel(mesh_cov[1])]).T
    
    # Calculate quantiles of estimated intensity function
    r_med = model.predict(cov, conf_int=[0.5])
    imshow(reshape(r_med,(grid_size,grid_size))[::-1,:],vmin=0.002,vmax=0.015)
    xticks([]);yticks([])
    colorbar()
    xlabel('Elevation')
    ylabel('Slope')
    tight_layout()
    show()
