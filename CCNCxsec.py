import numpy as np
import matplotlib.pyplot as plt

Mp        = 0.938272
Mmu       = 0.106
Mpi       = 0.140
pival     = np.pi
Mv        = 0.840
sin2theta = 0.2277
GF        = 1.166e-5
hc        = .197e-13
gA        = 1.267
mup       = 2.7930
mun       = -1.913042

def getE734dat():
    ''' E734 published data points '''
    # Q2 bin centers
    Q2   = np.linspace(0.45,1.05,7)
    # nu-p nce cross sections
    data = np.array([1.65,1.09,.803,.657,.447,.294,.205])*1e-39
    sig  = np.array([.11,.08,.076,.064,.073,.064,.057])*1e-39

    # nubar-p nce cross sections
    databar = np.array([.756,.426,.283,.184,.129,.108,.101])*1e-39
    sigbar  = np.array([.048,.026,.021,.019,.018,.018,.024])*1e-39

    return Q2,data,sig,databar,sigbar

def getUBMCdat():
    ''' MicroBooNE Sim. data from Genie (via LArSoft) '''
    # Q2 bin centers
    Q2   = np.linspace(0.1,1.85,19)
    # nu-p nce cross sections
    '''
    data = np.array([.1547,.1569,.1280,.12,.1188,.1376,.1441,.1290,.1111,.1579])
    sig  = np.array([.78,.93,1.12,1.41,1.92,2.83,3.74,4.85,6.76,9.81])*1e-2
    '''
    data = np.array([.1436,.1649,.1612,.1512,.1387,.1142,.1489,.0836,.0957,.1503,.1368,.1392,.1864,.1017,.1351,.12,.1333,.0833,.3333])
    sig  = np.array([.0108,.0112,.0125,.0139,.0156,.0160,.0213,.0174,.0224,.0336,.0365,.0448,.0612,.0436,.0644,.0733,.1004,.0867,.2222])
    # Using rough peaks
    #enu  = np.array([.7,.72,.79,.85,.99,1.0,1.1,1.15,1.22,1.26,1.3,1.4,1.45,1.45,1.48,1.7,1.75,1.75,1.85])
    # Using means
    enu  = np.array([.963,.982,1.04,1.10,1.19,1.23,1.31,1.40,1.43,1.56,1.46,1.61,1.69,1.81,1.04,2.13,2.10,2.59,2.81])

    return Q2,enu,data,sig

def NCxsec(Q2,enu,GaS,MA,FS,muS):
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

    return dsigNC*hc**2

def CCxsec(Q2,enu,MA):
    tau = Q2/(4.*Mp**2)
    su  = 4.*Mp*enu - Q2 - Mmu**2

    '''
    CCQE cross section
    '''
    # form factors
    denom = (1+Q2/Mv**2)**2

    FA = gA/(1 + Q2/MA**2)**2
    F1 = (1 + tau*(1+mup-mun))/((1+tau)*denom)
    F2 = (mup-mun)/((1+tau)*denom)
    FP = 2.*Mp**2/(Mpi**2 + Q2)*FA

    # terms
    A = (Mmu**2 + Q2)/Mp**2*((1+tau)*FA**2 - (1-tau)*F1**2 + tau*(1-tau)*F2**2 \
          + 4.*tau*F1*F2 - Mmu**2/(4.*Mp**2)*((F1+F2)**2 + (-1.*FA+2.*FP)**2 - (Q2/Mp**2+4)*FP**2))
    B = Q2/Mp**2*FA*(F1 + F2)
    C = 1./4.*(FA**2 + F1**2 + tau*F2**2)

    # cross-section
    dsigCC = GF**2*Mp**2/(8.*pival*enu**2)*(A + B*su/Mp**2 + C*su**2/Mp**4)

    return dsigCC*hc**2

def antiNCxsec(Q2,enu,GaS,MA,FS,muS):
    tau = Q2/(4.*Mp**2)
    su  = 4.*Mp*enu - Q2
    alp = 1 - 2.*sin2theta
    gam = -2./3.*sin2theta

    '''
    anti-nu NCE cross section
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

    return dsigNC*hc**2

def NCxsecSum(Q2,GaS,MA,FS,muS):
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

    return sigsum*hc**2

def NCxsecDiff(Q2,GaS,MA,FS,muS):
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

    return sigdiff*hc**2

