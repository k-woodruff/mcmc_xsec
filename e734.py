import numpy as np
import CCNCxsec as xs
import scipy.stats as st


''' E734 published data points '''

# Q2 bin centers
Q2      = np.linspace(0.45,1.05,7)
enu     = 1.

# nu-p nce cross section
data    = np.array([1.65,1.09,.803,.657,.447,.294,.205])*1e-39
sig     = np.array([0.33,0.17,0.12,0.098,0.09,0.073,0.063])*1e-39

# antinu-p cross section
databar = np.array([.756,.426,.283,.184,.129,.108,.101])*1e-39
sigbar  = np.array([0.164,0.062,0.037,0.027,0.023,0.022,0.027])*1e-39

def getdata():

    # put measured data into one tuple
    D = data,sig,databar,sigbar

    return Q2,enu,D

def lnprob(theta):

    ''' log probability of step (posterior*prior) '''

    lp = xs.lnprior(theta)
    return lp + lnlike(theta)

def lnlike(theta):

    ''' simultaneous nu/antinu nc-p xsec log likelihood '''

    GaS,MA,FS,muS           = theta
    model                   = xs.NCxsec(Q2,enu,GaS,MA,FS,muS)
    modelbar                = xs.antiNCxsec(Q2,enu,GaS,MA,FS,muS)
    lnlikevec               = -0.5*np.square((data-model)/sig)
    lnlikevecbar            = -0.5*np.square((databar-modelbar)/sigbar)

    return np.sum(lnlikevec) + np.sum(lnlikevecbar)

