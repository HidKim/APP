import dill
from HidKim_APP import augmented_permanental_process as APP
from HidKim_APP import parameter_initializer as p_init
from scipy.interpolate import LinearNDInterpolator
from pylab import *


if __name__ == "__main__":

    # Real-world data used in our NeurIPS2022 paper
    # Dim of covariate is two, dim of observation region is two
    data = dill.load(open('../data/real/Bei.dill','rb'))
    data_spk = array(data['spk'])

    # Construct covariate map with sample covariate data: data['t'] & data['cov']
    lin = LinearNDInterpolator(data['t'], data['cov'])
    fun_cov = lambda x: lin(x)

    # Initialize hyper-parameter of Gaussian kernel function:
    # (a,b1,b2) for k(t,t') = a * exp( -(b1*(t-t'))^2 - (b2*(t-t'))^2 )
    print(data['spk'])
    init_par = p_init(data['spk'], data['obs'], fun_cov(data['spk']))
    
    # Make a set of hyper-parameters to examine -> set_par
    w = array([1./3, 1./2, 1.0, 2.0, 3.0])
    a, b = meshgrid(w,w)
    z = array([ravel(a),ravel(b),ravel(b)]).T
    set_par = z * init_par
    
    print(set_par)
    sys.exit()
        

    # Plot the estiamtion results with three sample event data
    for i, datum in enumerate(data['spk'][:3]):
        
        # Initialize hyper-parameter of Gaussian kernel function:
        # (a,b) for k(t,t') = a * exp( -(b*(t-t'))^2 )
        init_par = p_init(datum, data['obs'], fun_cov(datum))
        
        # Make a set of hyper-parameters to examine -> set_par
        w = array([1./3, 1./2, 1.0, 2.0, 3.0])
        par0, par1 = init_par[0]*w, init_par[1]*w
        a, b = meshgrid(par0,par1)
        set_par = c_[reshape(a,(a.size,1)),reshape(b,(b.size,1))]
        
        # Perform intensity estimation in APP
        model = APP(kernel='Gaussian', eq_kernel='RFM',
                    eq_kernel_options={'cov_sampler':'Sobol', 'n_cov':1500, 'n_rfm':100})
        _ = model.fit(datum, data['obs'], fun_cov ,set_par)

        # Calculate quantiles of estimated intensity function
        r_low, r_med, r_upp = model.predict(cov, conf_int=[.025,.5,.975])
        
        subplot(3,1,i+1)
        fill_between(cov[:,0], r_low,r_upp, facecolor='r', alpha=0.3, lw=0)
        plot(cov[:,0], r_med, 'r', lw=1.2, label='APP')
        plot(cov[:,0], r_true,'k--', lw=1.2, label='True')
        legend()
        ylabel('Intensity')
        ylim(0,80)
        xlim(0,1)
    
    xlabel('Distance to R')
    tight_layout()
    show()
