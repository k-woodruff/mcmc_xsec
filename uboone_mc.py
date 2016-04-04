import numpy as np
import CCNCxsec as xs
import scipy.stats as st


''' microboone monte-carlo data points '''

# Q2 bin centers
Q2   = np.linspace(0.05,1.85,19)
enu  = 1.

# ratio of NCEp to CCQEn in larsoft sim
data = np.array([6.65,7.84,7.67,7.01,8.76,7.19,6.98,7.92,7.04,8.91,6.41,6.90,6.07,6.73,7.41,7.44,7.76,2.47,1.75])*1e-2
sig  = np.array([0.34,0.35,0.37,0.42,0.55,0.57,0.65,0.80,0.85,1.19,1.12,1.33,1.52,1.79,2.22,2.57,2.68,1.77,1.77])*1e-2

def getdata():

    # put measured data into one tuple
    D    = data,sig

    return Q2,enu,D


def lnprob(theta):

    ''' log probability of step (posterior*prior) '''

    lp = xs.lnprior(theta)
    return lp + lnlike(theta)


def lnlike(theta):

    ''' ratio of cross-sections log likelihood '''

    GaS,MA,FS,muS = theta
    ncxs          = xs.NCxsec(Q2,enu,GaS,MA,FS,muS)
    ccxs          = xs.CCxsec(Q2,enu,MA)
    model         = ncxs/ccxs
    lnlikevec     = -0.5*np.square((data-model)/sig)

    return np.sum(lnlikevec)



