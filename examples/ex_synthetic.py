from HidKim_APP import augmented_permanental_process as APP
from scipy.interpolate import LinearNDInterpolator
import dill
from pylab import *



if __name__ == "__main__":

    data = dill.load(open('./data/synthetic/g1_a05.dill','rb'))
    lin = LinearNDInterpolator(data['t'], data['cov'])
    fun_cov = lambda x: lin(x)
    set_par = []
    
    cov = linspace(0,1.5,200)[:,newaxis]
    r_true = data['f'](cov)

    for i, datum in enumerate(data['spk'][:3]):
        
        model = APP(eq_kernel='Gaussian', eq_kernel='RFM',
                    eq_kernel_options={'cov_sampler':'Sobol', 'n_cov':1500, 'n_rfm':100})
        _ = model.fit(datum, data['obs'], fun_cov ,set_par)
        r_low, r_med, r_upp = model.predict(cov, conf_int=[.025,.5,.975])
        
        subplot(3,1,i)
        fill_between(cov[:,0], r_low,r_upp, facecolor='r', alpha=0.3, lw=0)
        plot(cov[:,0], r_med, 'r', lw=1.2)
        plot(cov[:,0], r_true,' k--', lw=1.2)
    
    show()
