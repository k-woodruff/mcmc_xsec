import numpy as np
import scipy.optimize as op
import CCNCxsec as xs
import emcee
import triangle

'''
Examples of ways to run different algorithms for single experiments
'''

# this defines which data we are using (ex: e734,uboone_mc,uboone_data)
import e734 as expmt

def optimization():
    # use scipy optimize to minimize neg. log likelihood
    nll     = lambda *args: -lnlikeOpt(*args)
    result  = op.minimize(nll, [0.])
    Gresult = result["x"]
    return Gresult

def lnlikeOpt(theta):
    # hold other paramters fixed to some value
    GaS   = theta
    MA    = 1.049
    FS    = 0.
    muS   = 0.
    theta = GaS,MA,FS,muS
    return expmt.lnlike(theta)

def MCEnsemble(nwalkers,steps,burnin=1000):
    # use emcee ensemble sampler to find prob. surface
    ndim     = 4
    startpos = [[-.13,1.032,0.49,-0.39] + 5e-3*np.random.randn(ndim) for i in range(nwalkers)]

    # burn-in for defined number of steps (default=1000)
    sampler  = emcee.EnsembleSampler(nwalkers, ndim, expmt.lnprob)
    sampler.run_mcmc(startpos,burnin)
    bipos    = sampler.chain[:,-1,:].reshape((-1,ndim))

    # run mcmc
    sampler  = emcee.EnsembleSampler(nwalkers, ndim, expmt.lnprob)
    sampler.run_mcmc(bipos,steps)

    samples  = sampler.chain[:,:,:].reshape((-1, ndim))
    triangle.corner(samples,labels=[r'$\Delta s$',r'$M_A$',r'$F_1^S$',r'$\mu_S$'],quantiles=[0.05,0.5,0.95])
    return samples

def MCParTemp(nwalkers,steps,ntemps=40,burnin=100,threads=6):
    # use emcee parallel tempering sampler to find prob. surface
    ndim = 4
    startpos = [[[0,0,0,0] + 5e-3*np.random.randn(ndim) for i in range(nwalkers)] for i in range(ntemps)]

    # burn-in for defined number of steps (default=100)
    sampler  = emcee.PTSampler(ntemps,nwalkers, ndim, expmt.lnlike, xs.lnprior, threads=6)
    for p,logprob, loglike in sampler.sample(startpos,iterations=burnin):
        pass
    sampler.reset()

    # run mcmc
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, expmt.lnlike, xs.lnprior, threads=threads)
    for p, logprob, loglike in sampler.sample(p, lnprob0=logprob, lnlike0=loglike,iterations=steps):
        pass

    samples = sampler.chain[:,:,:,:].reshape((-1, ndim))
    triangle.corner(samples,labels=[r'$\Delta s$',r'$M_A$',r'$F_1^S$',r'$\mu_S$'],quantiles=[0.05,0.5,0.95])
    return samples






