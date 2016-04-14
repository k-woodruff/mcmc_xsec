import numpy as np
import CCNCxsec as xs
import emcee
import triangle
import time

'''
Examples of ways to run different algorithms for single experiments
'''

# this defines which data we are using (ex: e734,uboone_mc,uboone_data)
import uboone_mc as expmt1
import e734 as expmt2

def MCEnsemble(nwalkers,steps,burnin=1000):
    # use emcee ensemble sampler to find prob. surface
    ndim     = 4
    startpos = [[-.13,1.032,0.49,-0.39] + 5e-3*np.random.randn(ndim) for i in range(nwalkers)]

    # burn-in for defined number of steps (default=1000)
    sampler  = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(startpos,burnin)
    bipos    = sampler.chain[:,-1,:].reshape((-1,ndim))

    # run mcmc
    sampler  = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(bipos,steps)

    samples  = sampler.chain[:,:,:].reshape((-1, ndim))
    triangle.corner(samples,labels=[r'$\Delta s$',r'$M_A$',r'$F_1^S$',r'$\mu_S$'],quantiles=[0.05,0.5,0.95])
    return samples

def MCParTemp(nwalkers,steps,ntemps=40,burnin=100,threads=1):
    t1 = time.time()
    # use emcee parallel tempering sampler to find prob. surface
    ndim = 4
    startpos = [[[0,0,0,0] + 5e-3*np.random.randn(ndim) for i in range(nwalkers)] for i in range(ntemps)]

    # time how long it should take
    t0 = time.time()
    testpos = [[[0,0,0,0] + 5e-3*np.random.randn(4) for i in range(8)] for i in range(8)]
    sampler  = emcee.PTSampler(8,8, 4, lnlike, xs.lnprior, threads=threads)
    for ptest,logprobtest, logliketest in sampler.sample(testpos,iterations=10):
        pass
    sampler.reset()
    tdiff = time.time() - t0
    esttime = ntemps/8.*nwalkers/8.*(burnin+steps)/10.*tdiff/60.
    print 'Estimated time = {} minutes'.format(esttime)

    # burn-in for defined number of steps (default=100)
    sampler  = emcee.PTSampler(ntemps,nwalkers, ndim, lnlike, xs.lnprior, threads=threads)
    for p,logprob, loglike in sampler.sample(startpos,iterations=burnin):
        pass
    sampler.reset()

    # run mcmc
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike, xs.lnprior, threads=threads)
    for p, logprob, loglike in sampler.sample(p, lnprob0=logprob, lnlike0=loglike,iterations=steps):
        pass

    print 'Time: {} minutes'.format(time.time() - t0)
    samples = sampler.chain[:,:,:,:].reshape((-1, ndim))
    triangle.corner(samples,labels=[r'$\Delta s$',r'$M_A$',r'$F_1^S$',r'$\mu_S$'],quantiles=[0.05,0.5,0.95])
    print sampler.thermodynamic_integration_log_evidence(fburnin=0)
    return samples

def lnlike(theta):
    lnlike1 = expmt1.lnlike(theta)
    lnlike2 = expmt2.lnlike(theta)
    return lnlike1 + lnlike2

def lnprob(theta):
    lnprob1 = expmt1.lnprob(theta)
    lnprob2 = expmt2.lnprob(theta)
    return lnprob1 + lnprob2

