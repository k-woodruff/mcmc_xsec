import numpy as np
import scipy.stats as st

Mp        = 0.938272
Mmu       = 0.106
Mpi       = 0.140
pival     = np.pi
Mv        = 0.840
sin2theta = 0.2277
hc        = .197e-13
GF        = 1.166e-5*hc
gA        = -1.267          # changed for sams
mup       = 2.7930
mun       = -1.913042

def NCpxsec(Q2,enu,GaS,MA,FS,muS):
    tau   = Q2/(4.*Mp**2)
    su    = 4.*Mp*enu - Q2
    denom = (1+tau)*(1+Q2/Mv**2)**2

    # form factors
    FAS = GaS/((1 + Q2/MA**2)**2)
    F1S = FS*Q2/denom 
    F2S = muS/denom 

    FA  = gA/(2.*(1 + Q2/MA**2)**2) + FAS/2.
    F1  = (0.5 - sin2theta)*(1+tau*(1+mup-mun))/denom - sin2theta*(1+tau*(1+mup+mun))/denom - 0.5*F1S
    F2  = (0.5 - sin2theta)*(mup-mun)/denom - sin2theta*(mup+mun)/denom

    # terms
    A   = (Q2/Mp**2)*((1+tau)*FA**2 - (1-tau)*F1**2 + tau*(1-tau)*F2**2 + 4.*tau*F1*F2)
    B   = (Q2/Mp**2)*FA*(F1 + F2)
    C   = 0.25*(FA**2 + F1**2 + tau*F2**2)
    
    # cross-section
    dsigNC = GF**2*Mp**2/(8.*pival*enu**2)*(A - (su/Mp**2)*B + (su**2/Mp**4)*C)

    return dsigNC

def NCnxsec(Q2,enu,GaS,MA,FS,muS):
    tau   = Q2/(4.*Mp**2)
    su    = 4.*Mp*enu - Q2
    denom = (1+tau)*(1+Q2/Mv**2)**2

    # form factors
    FAS = GaS/((1 + Q2/MA**2)**2)
    F1S = FS*Q2/denom 
    F2S = muS/denom 

    FA  = -1.*gA/(2.*(1 + Q2/MA**2)**2) + FAS/2.
    F1  = (0.5 - sin2theta)*(-1.)*(1+tau*(1+mup-mun))/denom - sin2theta*(1+tau*(1+mup+mun))/denom - 0.5*F1S
    F2  = (0.5 - sin2theta)*(-1.)*(mup-mun)/denom - sin2theta*(mup+mun)/denom

    # terms
    A   = (Q2/Mp**2)*((1+tau)*FA**2 - (1-tau)*F1**2 + tau*(1-tau)*F2**2 + 4.*tau*F1*F2)
    B   = (Q2/Mp**2)*FA*(F1 + F2)
    C   = 0.25*(FA**2 + F1**2 + tau*F2**2)
    
    # cross-section
    dsigNC = GF**2*Mp**2/(8.*pival*enu**2)*(A - (su/Mp**2)*B + (su**2/Mp**4)*C)

    return dsigNC

def CCnxsec(Q2,enu,MA):
    tau   = Q2/(4.*Mp**2)
    su    = 4.*Mp*enu - Q2 - Mmu**2
    denom = (1+tau)*(1+Q2/Mv**2)**2

    '''
    CCQEn cross section
    '''
    # form factors

    FA = gA/(1 + Q2/MA**2)**2
    F1 = (1 + tau*(1+mup-mun))/denom
    F2 = (mup-mun)/denom
    FP = 2.*Mp**2/(Mpi**2 + Q2)*FA

    # terms
    A = (Mmu**2 + Q2)/Mp**2*((1+tau)*FA**2 - (1-tau)*F1**2 + tau*(1-tau)*F2**2 \
          + 4.*tau*F1*F2 - Mmu**2/(4.*Mp**2)*((F1+F2)**2 + (FA+2.*FP)**2 - (Q2/Mp**2+4)*FP**2))
    B = (Q2/Mp**2)*FA*(F1 + F2)
    C = 0.25*(FA**2 + F1**2 + tau*F2**2)

    # cross-section
    dsigCC = GF**2*Mp**2/(8.*pival*enu**2)*(A - (su/Mp**2)*B + (su**2/Mp**4)*C)

    return dsigCC

def antiNCpxsec(Q2,enu,GaS,MA,FS,muS):
    tau   = Q2/(4.*Mp**2)
    su    = 4.*Mp*enu - Q2
    denom = (1+tau)*(1+Q2/Mv**2)**2

    # form factors
    FAS = GaS/((1 + Q2/MA**2)**2)
    F1S = FS*Q2/denom 
    F2S = muS/denom 

    FA  = gA/(2.*(1 + Q2/MA**2)**2) + FAS/2.
    F1  = (0.5 - sin2theta)*(1+tau*(1+mup-mun))/denom - sin2theta*(1+tau*(1+mup+mun))/denom - 0.5*F1S
    F2  = (0.5 - sin2theta)*(mup-mun)/denom - sin2theta*(mup+mun)/denom

    # terms
    A   = (Q2/Mp**2)*((1+tau)*FA**2 - (1-tau)*F1**2 + tau*(1-tau)*F2**2 + 4.*tau*F1*F2)
    B   = (Q2/Mp**2)*FA*(F1 + F2)
    C   = 0.25*(FA**2 + F1**2 + tau*F2**2)
    
    # cross-section
    dsigNC = GF**2*Mp**2/(8.*pival*enu**2)*(A + (su/Mp**2)*B + (su**2/Mp**4)*C)

    return dsigNC

def antiNCnxsec(Q2,enu,GaS,MA,FS,muS):
    tau = Q2/(4.*Mp**2)
    su  = 4.*Mp*enu - Q2
    alp = 1 - 2.*sin2theta
    gam = -2./3.*sin2theta

    '''
    anti-nu NCEp cross section
    '''
    # form factors
    denom = (1+Q2/Mv**2)**2
   
    FAS  = GaS/(1+Q2/MA**2)**2
    F1S  = FS*Q2/((1+tau)*denom)
    F2S  = muS/((1+tau)*denom)

    Fv0 = 1.5*(mup+mun)/((1+tau)*denom)
    Fv3 = 0.5*(mun-mup)/((1+tau)*denom)
    Gv0 = 1.5*(1+mup+mun)/denom
    Gv3 = 0.5*(1+mun-mup)/denom

    FA = -0.5*gA/(1 + Q2/MA**2)**2 - .5*FAS
    F2 = alp*Fv3 + gam*Fv0 - .5*F2S
    F1 = alp*Gv3 + gam*Gv0 - alp*Fv3 - gam*Fv0 - .5*F1S

    # terms
    A = Q2/Mp**2*((1+tau)*FA**2 - (1-tau)*F1**2 + tau*(1-tau)*F2**2 \
          + 4.*tau*F1*F2)
    B = Q2/Mp**2*FA*(F1 + F2)
    C = 1./4.*(FA**2 + F1**2 + tau*F2**2)

    # cross-section
    dsigNC = GF**2*Mp**2/(8.*pival*enu**2)*(A - B*su/Mp**2 + C*su**2/Mp**4)

    return dsigNC

def NCxsecSum(Q2,enu,GaS,MA,FS,muS):
    tau = Q2/(4.*Mp**2)
    su  = 4.*Mp*enu - Q2
    alp = 1 - 2.*sin2theta
    gam = -2./3.*sin2theta

    '''
    nu + anti-nu --- NCE cross section
    '''
    # form factors
    denom = (1+Q2/Mv**2)**2
   
    FAS  = GaS/(1+Q2/MA**2)**2
    F1S  = FS*Q2/((1+tau)*denom)
    F2S  = muS/((1+tau)*denom)

    Fv0 = 1.5*(mup+mun)/((1+tau)*denom)
    Fv3 = 0.5*(mup-mun)/((1+tau)*denom)
    Gv0 = 1.5*(1+mup+mun)/denom
    Gv3 = 0.5*(1+mup-mun)/denom

    FA = 0.5*gA/(1 + Q2/MA**2)**2 - .5*FAS
    F2 = alp*Fv3 + gam*Fv0 - .5*F2S
    F1 = alp*Gv3 + gam*Gv0 - alp*Fv3 - gam*Fv0 - .5*F1S

    # terms
    A = Q2/Mp**2*((1+tau)*FA**2 - (1-tau)*F1**2 + tau*(1-tau)*F2**2 \
          + 4.*tau*F1*F2)
    C = 1./4.*(FA**2 + F1**2 + tau*F2**2)

    sigsum = GF**2*Mp**2/(4.*pival*enu**2)*(A + C*su**2/Mp**4)

    return sigsum

def NCxsecDiff(Q2,enu,GaS,MA,FS,muS):
    tau = Q2/(4.*Mp**2)
    su  = 4.*Mp*enu - Q2
    alp = 1 - 2.*sin2theta
    gam = -2./3.*sin2theta

    '''
    NCE cross section
    '''
    # form factors
    denom = (1+Q2/Mv**2)**2
   
    FAS  = GaS/(1+Q2/MA**2)**2
    F1S  = FS*Q2/((1+tau)*denom)
    F2S  = muS/((1+tau)*denom)
    
    Gv0 = 1.5*(1+mup+mun)/denom
    Gv3 = 0.5*(1+mup-mun)/denom

    FA      = 0.5*gA/(1 + Q2/MA**2)**2 - .5*FAS
    F1F2sum = alp*Gv3 + gam*Gv0 - .5*(F1S + F2S)

    B = Q2/Mp**2*FA*(F1F2sum)
    
    sigdiff = GF**2/(4.*pival*enu**2)*B*su/Mp**2

    return sigdiff


def lnprior(theta,form='informative'):

    formopts = ['informative','uniform','reasonable']
    assert form in formopts, 'not a valid prior option'
  
    ''' prior log prob for each param. in theta '''
    ''' needs work/thought '''

    GaS,MA,FS,muS = theta

    if form == 'informative':
        # get prior from previous experiments
        lpGaS = st.norm.logpdf(GaS,0.,.5)
        lpMA  = st.norm.logpdf(MA,1.0,.5)
        lpFS  = st.norm.logpdf(FS,0.,.5)
        lpmuS = st.norm.logpdf(muS,0.,.5)

        return lpmuS+lpFS+lpGaS+lpMA

    if form == 'uniform':
        # uniform prior --> likelihood
        return 0.0

    if form == 'reasonable':
        # use sane, but loose priors on theta
        # if in these ranges p=1, otherwise p=0
        if -1. < GaS < 1. and .5 < MA < 2. and -1. < FS < 1.5 and -1.5 < muS < 1.:
            return 0.0
        return -np.inf

'''
Retired functions
'''

def oldNCpxsec(Q2,enu,GaS,MA,FS,muS):
    tau = Q2/(4.*Mp**2)
    su  = 4.*Mp*enu - Q2
    alp = 1 - 2.*sin2theta
    gam = -2./3.*sin2theta

    '''
    NCEp cross section
    '''
    # form factors
    denom = (1+Q2/Mv**2)**2

    FAS  = GaS/(1+Q2/MA**2)**2
    F1S  = FS*Q2/((1+tau)*denom)
    F2S  = muS/((1+tau)*denom)

    Fv0 = 1.5*(mup+mun)/((1+tau)*denom)
    Fv3 = 0.5*(mup-mun)/((1+tau)*denom)
    Gv0 = 1.5*(1+mup+mun)/denom
    Gv3 = 0.5*(1+mup-mun)/denom

    FA = .5*gA/(1 + Q2/MA**2)**2 - .5*FAS
    F2 = alp*Fv3 + gam*Fv0 - .5*F2S
    F1 = alp*Gv3 + gam*Gv0 - alp*Fv3 - gam*Fv0 - .5*F1S

    # terms
    A = Q2/Mp**2*((1+tau)*FA**2 - (1-tau)*F1**2 + tau*(1-tau)*F2**2 \
          + 4.*tau*F1*F2)
    B = Q2/Mp**2*FA*(F1 + F2)
    C = 1./4.*(FA**2 + F1**2 + tau*F2**2)

    # cross-section
    dsigNC = GF**2*Mp**2/(8.*pival*enu**2)*(A + B*su/Mp**2 + C*su**2/Mp**4)

    return dsigNC

def oldNCnxsec(Q2,enu,GaS,MA,FS,muS):
    tau = Q2/(4.*Mp**2)
    su  = 4.*Mp*enu - Q2
    alp = 1 - 2.*sin2theta
    gam = -2./3.*sin2theta

    '''
    NCEn cross section
    '''
    # form factors
    denom = (1+Q2/Mv**2)**2

    FAS  = GaS/(1+Q2/MA**2)**2
    F1S  = FS*Q2/((1+tau)*denom)
    F2S  = muS/((1+tau)*denom)

    Fv0 = 1.5*(mup+mun)/((1+tau)*denom)
    Fv3 = 0.5*(mun-mup)/((1+tau)*denom)
    Gv0 = 1.5*(1+mup+mun)/denom
    Gv3 = 0.5*(1+mun-mup)/denom

    FA = .5*gA/(1 + Q2/MA**2)**2 - .5*FAS
    F2 = alp*Fv3 + gam*Fv0 - .5*F2S
    F1 = alp*Gv3 + gam*Gv0 - alp*Fv3 - gam*Fv0 - .5*F1S

    # terms
    A = Q2/Mp**2*((1+tau)*FA**2 - (1-tau)*F1**2 + tau*(1-tau)*F2**2 \
          + 4.*tau*F1*F2)
    B = Q2/Mp**2*FA*(F1 + F2)
    C = 1./4.*(FA**2 + F1**2 + tau*F2**2)

    # cross-section
    dsigNC = GF**2*Mp**2/(8.*pival*enu**2)*(A + B*su/Mp**2 + C*su**2/Mp**4)

    return dsigNC

def oldantiNCpxsec(Q2,enu,GaS,MA,FS,muS):
    tau = Q2/(4.*Mp**2)
    su  = 4.*Mp*enu - Q2
    alp = 1 - 2.*sin2theta
    gam = -2./3.*sin2theta

    '''
    anti-nu NCEp cross section
    '''
    # form factors
    denom = (1+Q2/Mv**2)**2
   
    FAS  = GaS/(1+Q2/MA**2)**2
    F1S  = FS*Q2/((1+tau)*denom)
    F2S  = muS/((1+tau)*denom)

    Fv0 = 1.5*(mup+mun)/((1+tau)*denom)
    Fv3 = 0.5*(mup-mun)/((1+tau)*denom)
    Gv0 = 1.5*(1+mup+mun)/denom
    Gv3 = 0.5*(1+mup-mun)/denom

    FA = 0.5*gA/(1 + Q2/MA**2)**2 - .5*FAS
    F2 = alp*Fv3 + gam*Fv0 - .5*F2S
    F1 = alp*Gv3 + gam*Gv0 - alp*Fv3 - gam*Fv0 - .5*F1S

    # terms
    A = Q2/Mp**2*((1+tau)*FA**2 - (1-tau)*F1**2 + tau*(1-tau)*F2**2 \
          + 4.*tau*F1*F2)
    B = Q2/Mp**2*FA*(F1 + F2)
    C = 1./4.*(FA**2 + F1**2 + tau*F2**2)

    # cross-section
    dsigNC = GF**2*Mp**2/(8.*pival*enu**2)*(A - B*su/Mp**2 + C*su**2/Mp**4)

    return dsigNC



