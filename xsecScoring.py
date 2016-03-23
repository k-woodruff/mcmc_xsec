import numpy as np
import math
import scipy.stats as st
import scipy.optimize as op
import emcee
import triangle
import CCNCxsec as xs

def xsecRatio(theta,Q2,enu):
    GaS,MA,FS,muS = theta
    return xs.NCxsec(Q2,enu,GaS,MA,FS,muS)/xs.CCxsec(Q2,enu,MA)

def antixsecRatio(theta,Q2):
    GaS,MA,FS,muS = theta
    return xs.antiNCxsec(Q2,enu,GaS,MA,FS,muS)

def lnlike(theta,Q2,D,sig,Dbar,sigbar):
    GaS,MA,FS,muS = theta
    enu = 1.
    model = xs.NCxsec(Q2,enu,GaS,MA,FS,muS)/1.05
    modelbar = xs.antiNCxsec(Q2,enu,GaS,MA,FS,muS)/1.09
    lnlikevec = -0.5*np.square((D-model)/sig)
    lnlikevecbar = -0.5*np.square((Dbar-modelbar)/sigbar)
    return np.sum(lnlikevec) + np.sum(lnlikevecbar)

def lnlikeUB(theta,Q2,enu,D,sig):
    model = xsecRatio(theta,Q2,enu)
    lnlikevec = -0.5*np.square((D-model)/sig)
    return np.sum(lnlikevec)

def tryOpt():
    Q2,D,sig,Dbar,sigbar = xs.getE734dat()
    nll = lambda *args: -lnlikeOpt(*args)
    result = op.minimize(nll, [0.], args=(Q2,D,sig,Dbar,sigbar))
    Gresult = result["x"]
    return Gresult

def lnlikeOpt(theta,Q2,D,sig,Dbar,sigbar):
    GaS = theta
    MA = 1.049
    FS = 0.
    muS = 0.
    theta = GaS,MA,FS,muS
    model = xsecRatio(theta,Q2,enu)
    modelbar = antixsecRatio(theta,Q2,enu)
    lnlikevec = -0.5*np.square((D-model)/sig)
    lnlikevecbar = -0.5*np.square((Dbar-modelbar)/sigbar)
    return np.sum(lnlikevec) + np.sum(lnlikevecbar)

def MCburnin(nwalkers,steps):
    Q2,D,sig,Dbar,sigbar = xs.getE734dat()
    ndim = 4
    startpos = [[-.13,1.032,0.49,-0.39] + 5e-3*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Q2,D,sig,Dbar,sigbar))
    sampler.run_mcmc(startpos,8000)
    bipos = sampler.chain[:,-1,:].reshape((-1,ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Q2,D,sig,Dbar,sigbar))
    sampler.run_mcmc(bipos,steps)
    #samples = sampler.flatchain.reshape((-1, ndim))
    samples = sampler.chain[:,:,:].reshape((-1, ndim))
    triangle.corner(samples,labels=[r'$\Delta s$',r'$M_A$',r'$F_1^S$',r'$\mu_S$'],quantiles=[0.05,0.5,0.95])
    return samples

def PTMCburnin(nwalkers,steps):
    Q2,D,sig,Dbar,sigbar = xs.getE734dat()
    ndim = 4
    ntemps = 20
    #startpos = [[[-.13,1.032,0.49,-0.39] + 5e-3*np.random.randn(ndim) for i in range(nwalkers)] for i in range(ntemps)]
    startpos = [[[0,0,0,0] + 5e-3*np.random.randn(ndim) for i in range(nwalkers)] for i in range(ntemps)]
    sampler = emcee.PTSampler(ntemps,nwalkers, ndim, lnlike, lnprior, threads=6, loglargs=(Q2,D,sig,Dbar,sigbar))
    for p,logprob, loglike in sampler.sample(startpos,iterations=800):
        pass
    sampler.reset()
    #sampler.run_mcmc(startpos,800)
    #bipos = sampler.chain[:,-1,:].reshape((-1,ndim))
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike, lnprior, threads=6, loglargs=(Q2,D,sig,Dbar,sigbar))
    for p, logprob, loglike in sampler.sample(p, lnprob0=logprob, lnlike0=loglike,iterations=steps):
        pass
    #sampler.run_mcmc(bipos,steps)
    #samples = sampler.flatchain.reshape((-1, ndim))
    samples = sampler.chain[:,:,:,:].reshape((-1, ndim))
    triangle.corner(samples,labels=[r'$\Delta s$',r'$M_A$',r'$F_1^S$',r'$\mu_S$'],quantiles=[0.05,0.5,0.95])
    return samples

def MCRat(nwalkers,steps):
    Q2,enu,D,DBar = xs.getUBMCdat()
    ndim = 4
    startpos = [[-.13,1.032,0.49,-0.39] + 5e-3*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobUB, args=(Q2,enu,D,DBar))
    sampler.run_mcmc(startpos,8000)
    bipos = sampler.chain[:,-1,:].reshape((-1,ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobUB, args=(Q2,enu,D,DBar))
    sampler.run_mcmc(bipos,steps)
    samples = sampler.chain[:,:,:].reshape((-1, ndim))
    triangle.corner(samples,labels=[r'$\Delta s$',r'$M_A$',r'$F_1^S$',r'$\mu_S$'],quantiles=[0.05,0.5,0.95])
    return samples
    
def lnprob(theta,Q2,D,sig,Dbar,sigbar):
    lp = lnprior(theta)
    return lp + lnlike(theta,Q2,D,sig,Dbar,sigbar)

def lnprobUB(theta,Q2,enu,D,sig):
    lp = lnprior(theta)
    return lp + lnlikeUB(theta,Q2,enu,D,sig)

def lnprior(theta):
    GaS,MA,FS,muS = theta

    lpGaS = st.norm.logpdf(GaS,0.,.5)
    lpMA = st.norm.logpdf(MA,1.0,.5)
    lpFS  = st.norm.logpdf(FS,0.,.5)
    lpmuS = st.norm.logpdf(muS,0.,.5)
    '''
    if 
    if MA < 0.:
        lpMA = -np.inf
    else:
        lpMA = st.norm.logpdf(MA,1.05,1.)
    if -1. < GaS < 1. and .996 < MA < 1.068 and -.21 < FS < 1.19 and -1.09 < muS < .310:
        return 0.0
    return -np.inf
    '''
    return lpmuS+lpFS+lpGaS+lpMA








