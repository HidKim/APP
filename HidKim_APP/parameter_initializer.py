import numpy as np

def parameter_initializer(datum,obs,cov_datum):
    prec = 1./np.std(cov_datum,0)
    var_rate = 1.
    for i in range(len(obs)):
        bins      = np.linspace(obs[i][0],obs[i][1],10)
        rate, _   = np.histogram(datum[:,i],bins=bins)
        rate      = rate / (bins[1]-bins[0])
        var_rate *= pow(np.var(np.sqrt(rate)),1./len(obs))
    set_par = [var_rate]+list(prec)
    return set_par
