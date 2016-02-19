import numpy as np
import matplotlib.pyplot as plt
import math
import CCNCxsec as xs

def gridPlot2D(numpts):
    ''' get data '''
    Q2,data,sig,databar,sigbar = xs.getE734dat()

    ''' Make grid '''
    GaSvec = np.linspace(-0.5,0.1,numpts)
    MAvec  = np.linspace(0.996,1.068,numpts)

    GaS,MA = np.meshgrid(GaSvec,MAvec)
    lnlikes = np.zeros((len(GaS),len(MA)))

    # make lnlikelihood matrix
    for i in range(len(GaS)):
        for j in range(len(MA)):
            
            theta = GaS[i][j],MA[i][j],0.,0.
            #theta = GaS[i][j],1.032,0.,0.
            lnlikes[i][j] = lnlike(theta,Q2,data,sig,databar,sigbar)

    plt.pcolor(MA,GaS,lnlikes)


def gridPlot(numpts):
    ''' get data '''
    Q2,data,sig,databar,sigbar = xs.getE734dat()

    ''' Make grid '''
    # What values of each parameter to use
    GaSvec = np.linspace(-.3,.3,numpts)
    MAvec  = np.linspace(.5,1.5,numpts)
    FSvec  = np.linspace(0.,1.,numpts)
    muSvec = np.linspace(-1.,0.,numpts)
    '''
    GaSvec = np.linspace(-2.,3,numpts)
    MAvec  = np.linspace(.2,.92,numpts)
    FSvec  = np.linspace(-4.,3.,numpts)
    muSvec = np.linspace(-5,2.,numpts)
    '''

    GaS,MA,FS,muS = np.meshgrid(GaSvec,MAvec,FSvec,muSvec,indexing='ij')
    GaS = GaS.flatten()
    MA  = MA.flatten()
    FS  = FS.flatten()
    muS = muS.flatten()

    params  = np.vstack((GaS,MA,FS,muS))
    lnlikes = np.zeros(len(GaS))

    # make lnlikelihood matrix
    for i in range(len(GaS)):
        theta = params[:,i]
        lnlikes[i] = lnlike(theta,Q2,data,sig,databar,sigbar)

    # reshape matrices
    lnlikes = lnlikes.reshape(numpts,numpts,numpts,numpts)
    GaS,MA,FS,muS = np.meshgrid(GaSvec,MAvec,FSvec,muSvec,indexing='ij')

    lnGaSMA = np.zeros((len(GaSvec),len(MAvec)))
    # sum over other params to make contour plots
    for i in range(len(FS)):
        for j in range(len(muS)):
            lnGaSMA = lnGaSMA + lnlikes[:,:,i,j]

    lnGaSFS = np.zeros((len(GaSvec),len(FSvec)))
    # sum over other params to make contour plots
    for i in range(len(MA)):
        for j in range(len(muS)):
            lnGaSFS = lnGaSFS + lnlikes[:,i,:,j]

    lnGaSmuS = np.zeros((len(GaSvec),len(muSvec)))
    # sum over other params to make contour plots
    for i in range(len(MA)):
        for j in range(len(FS)):
            lnGaSmuS = lnGaSmuS + lnlikes[:,i,j,:]

    lnMAmuS = np.zeros((len(MAvec),len(muSvec)))
    for i in range(len(GaS)):
        for j in range(len(FS)):
            lnMAmuS = lnMAmuS + lnlikes[i,:,j,:]

    lnMAFS = np.zeros((len(MAvec),len(FSvec)))
    for i in range(len(GaS)):
        for j in range(len(muS)):
            lnMAFS = lnMAFS + lnlikes[i,:,:,j]

    lnFSmuS = np.zeros((len(FSvec),len(muSvec)))
    for i in range(len(GaS)):
        for j in range(len(MA)):
            lnFSmuS = lnFSmuS + lnlikes[i,j,:,:]


    ax = plt.subplot2grid((3,3),(0,0))
    #ax.pcolor(GaS[:,:,0,0],MA[:,:,0,0],lnGaSMA)
    lnGaSMA = lnGaSMA - lnGaSMA.max()
    ax.contour(GaS[:,:,0,0],MA[:,:,0,0],np.exp(lnGaSMA))
    ax.set_ylabel(r'$M_A$',fontsize=24)
    ax = plt.subplot2grid((3,3),(1,0))
    #ax.pcolor(GaS[:,0,:,0],FS[:,0,:,0],lnGaSFS)
    lnGaSFS = lnGaSFS - lnGaSFS.max()
    ax.contour(GaS[:,0,:,0],FS[:,0,:,0],np.exp(lnGaSFS))
    ax.set_ylabel(r'$F_1^S$',fontsize=24)
    ax = plt.subplot2grid((3,3),(2,0))
    #ax.pcolor(GaS[:,0,0,:],muS[:,0,0,:],lnGaSmuS)
    lnGaSmuS = lnGaSmuS - lnGaSmuS.max()
    ax.contour(GaS[:,0,0,:],muS[:,0,0,:],np.exp(lnGaSmuS))
    ax.set_xlabel(r'$G_A^S(0)$',fontsize=24)
    ax.set_ylabel(r'$\mu_S$',fontsize=24)

    ax = plt.subplot2grid((3,3),(2,1))
    #ax.pcolor(MA[0,:,0,:].T,muS[0,:,0,:].T,lnMAmuS.T)
    lnMAmuS = lnMAmuS - lnMAmuS.max()
    ax.contour(MA[0,:,0,:].T,muS[0,:,0,:].T,np.exp(lnMAmuS.T))
    ax.set_xlabel(r'$M_A$',fontsize=24)
    ax = plt.subplot2grid((3,3),(1,1))
    #ax.pcolor(MA[0,:,:,0].T,FS[0,:,:,0].T,lnMAFS.T)
    lnMAFS = lnMAFS - lnMAFS.max()
    ax.contour(MA[0,:,:,0].T,FS[0,:,:,0].T,np.exp(lnMAFS.T))

    ax = plt.subplot2grid((3,3),(2,2))
    lnFSmuS = lnFSmuS - lnFSmuS.max()
    #pc = ax.pcolor(FS[0,0,:,:],muS[0,0,:,:],lnFSmuS)
    pc = ax.contour(FS[0,0,:,:],muS[0,0,:,:],np.exp(lnFSmuS))
    ax.set_xlabel(r'$F_1^S$',fontsize=24)
    #plt.colorbar(pc)

    return lnlikes, GaS, MA, FS, muS
        
def gridPlotSD():
    ''' get data '''
    Q2,data,sig,databar,sigbar = xs.getE734dat()

    ''' make grid '''
    GaSvec = np.linspace(-.3,.3,20)
    MAvec  = np.linspace(.5,1.5,20)
    FSvec  = np.linspace(0.,1.,20)
    muSvec = np.linspace(-1.,0.,20)

    GaS,MA,FS,muS = np.meshgrid(GaSvec,MAvec,FSvec,muSvec,indexing='ij')
    GaS = GaS.flatten()
    MA  = MA.flatten()
    FS  = FS.flatten()
    muS = muS.flatten()

    params  = np.vstack((GaS,MA,FS,muS))
    lnlikes = np.zeros(len(GaS))

    for i in range(len(GaS)):
        theta = params[:,i]
        lnlikes[i] = lnlikeSD(theta,Q2,data,sig,databar,sigbar)

    # reshape matrices
    lnlikes = lnlikes.reshape(20,20,20,20)
    GaS,MA,FS,muS = np.meshgrid(GaSvec,MAvec,FSvec,muSvec,indexing='ij')

    lnGaSMA = np.zeros((len(GaSvec),len(MAvec)))
    # sum over other params to make contour plots
    for i in range(len(FS)):
        for j in range(len(muS)):
            lnGaSMA = lnGaSMA + lnlikes[:,:,i,j]

    lnGaSFS = np.zeros((len(GaSvec),len(FSvec)))
    # sum over other params to make contour plots
    for i in range(len(MA)):
        for j in range(len(muS)):
            lnGaSFS = lnGaSFS + lnlikes[:,i,:,j]

    lnGaSmuS = np.zeros((len(GaSvec),len(muSvec)))
    # sum over other params to make contour plots
    for i in range(len(MA)):
        for j in range(len(FS)):
            lnGaSmuS = lnGaSmuS + lnlikes[:,i,j,:]

    lnMAmuS = np.zeros((len(MAvec),len(muSvec)))
    for i in range(len(GaS)):
        for j in range(len(FS)):
            lnMAmuS = lnMAmuS + lnlikes[i,:,j,:]

    lnMAFS = np.zeros((len(MAvec),len(FSvec)))
    for i in range(len(GaS)):
        for j in range(len(muS)):
            lnMAFS = lnMAFS + lnlikes[i,:,:,j]

    lnFSmuS = np.zeros((len(FSvec),len(muSvec)))
    for i in range(len(GaS)):
        for j in range(len(MA)):
            lnFSmuS = lnFSmuS + lnlikes[i,j,:,:]


    ax = plt.subplot2grid((3,3),(0,0))
    ax.pcolor(GaS[:,:,0,0],MA[:,:,0,0],lnGaSMA)
    ax.set_ylabel(r'$M_A$',fontsize=24)
    ax = plt.subplot2grid((3,3),(1,0))
    ax.pcolor(GaS[:,0,:,0],FS[:,0,:,0],lnGaSFS)
    ax.set_ylabel(r'$F_1^S$',fontsize=24)
    ax = plt.subplot2grid((3,3),(2,0))
    ax.pcolor(GaS[:,0,0,:],muS[:,0,0,:],lnGaSmuS)
    ax.set_xlabel(r'$G_A^S(0)$',fontsize=24)
    ax.set_ylabel(r'$\mu_S$',fontsize=24)

    ax = plt.subplot2grid((3,3),(2,1))
    ax.pcolor(MA[0,:,0,:].T,muS[0,:,0,:].T,lnMAmuS.T)
    ax.set_xlabel(r'$M_A$',fontsize=24)
    ax = plt.subplot2grid((3,3),(1,1))
    ax.pcolor(MA[0,:,:,0].T,FS[0,:,:,0].T,lnMAFS.T)

    ax = plt.subplot2grid((3,3),(2,2))
    ax.pcolor(FS[0,0,:,:],muS[0,0,:,:],lnFSmuS)
    ax.set_xlabel(r'$F_1^S$',fontsize=24)

    return lnlikes, GaS, MA, FS, muS

def lnlike(theta,Q2,D,sig,Dbar,sigbar):
    GaS,MA,FS,muS = theta
    model = xs.NCxsec(Q2,GaS,MA,FS,muS)
    modelbar = xs.antiNCxsec(Q2,GaS,MA,FS,muS)
    lnlikevec = -0.5*np.square((D-model)/sig)
    lnlikevecbar = -0.5*np.square((Dbar-modelbar)/sigbar)
    return np.sum(lnlikevec) + np.sum(lnlikevecbar)

def lnlikeSD(theta,Q2,D,sig,Dbar,sigbar):
    GaS,MA,FS,muS = theta
    modeldiff = xs.NCxsecDiff(Q2,GaS,MA,FS,muS)
    modelsum  = xs.NCxsecSum(Q2,GaS,MA,FS,muS)
    sigcomb   = np.sqrt(sig**2 + sigbar**2)
    lnlikevecdiff = -0.5*np.square((D-Dbar - modeldiff)/sigcomb)
    lnlikevecsum  = -0.5*np.square((D+Dbar - modelsum)/sigcomb)
    return np.sum(lnlikevecdiff) + np.sum(lnlikevecsum)

def makeSamplePlot(samples):
    Q2,data,dataErr,databar,errbar = xs.getE734dat()
    enu = 1.
    #Q2,enu,data,dataErr = xs.getUBMCdat()

    y = np.zeros((len(Q2),len(samples[:,0])))
    for i in range(len(samples[:,0])):
        theta = samples[i,:]
        ncxs = xs.NCxsec(Q2,enu,theta[0],theta[1],theta[2],theta[3])
        #ccxs = xs.CCxsec(Q2,enu,theta[1])
        y[:,i] = xs.NCxsec(Q2,enu,theta[0],theta[1],theta[2],theta[3])
        #y[:,i] = ncxs/ccxs

    plt.plot(Q2,y,color='steelblue',alpha=0.05)
    plt.errorbar(Q2,data,yerr=dataErr,fmt='o',color='tomato')
    #plt.yscale('log')

def makePlot():
    Q2 = np.linspace(0.25,1.25,50)
    muS = 0.
    FS  = 0.
    MA  = 1.061
    GaS = -.12
    yTruth = xs.NCxsec(Q2,GaS,MA,FS,muS)
    yTruthBar = xs.antiNCxsec(Q2,GaS,MA,FS,muS)

    # Data points
    Q,data,dataErr,databar,errbar = xs.getE734dat()

    fig, ax = plt.subplots(nrows=2,ncols=2,sharex=True)

    # changing DeltaS in top left
    muS = 0.
    FS  = 0.
    MA  = 1.061
    GaSvec = np.linspace(-.5,.3,50)
    y = np.zeros((len(Q2),len(GaSvec)))
    ybar = np.zeros((len(Q2),len(GaSvec)))
    ymin = xs.NCxsec(Q2,GaSvec[0],MA,FS,muS)
    yminbar = xs.antiNCxsec(Q2,GaSvec[0],MA,FS,muS)
    for i,GaS in enumerate(GaSvec):
        y[:,i] = xs.NCxsec(Q2,GaS,MA,FS,muS)
        ybar[:,i] = xs.antiNCxsec(Q2,GaS,MA,FS,muS)
    
    ax[0,0].plot(Q2,y,color='steelblue',alpha=0.7)
    ax[0,0].plot(Q2,ymin,color='steelblue',linewidth=3)
    ax[0,0].plot(Q2,ybar,color='lightblue',alpha=0.7)
    ax[0,0].plot(Q2,yminbar,color='lightblue',linewidth=3)
    ax[0,0].plot(Q2,yTruth,color='black')
    ax[0,0].plot(Q2,yTruthBar,color='dimgray')
    ax[0,0].errorbar(Q,data,yerr=dataErr,fmt='o',color='tomato')
    ax[0,0].errorbar(Q,databar,yerr=errbar,fmt='o',color='steelblue')
    ax[0,0].set_yscale('log')
    ax[0,0].set_title(r'$\Delta S$ from -0.5 to 0.3')

    # changing MA in top right
    muS = 0.
    FS  = 0.
    MAvec  = np.linspace(0.75,1.5,50)
    GaS = -.12
    y = np.zeros((len(Q2),len(MAvec)))
    ybar = np.zeros((len(Q2),len(MAvec)))
    ymin = xs.NCxsec(Q2,GaS,MAvec[0],FS,muS)
    yminbar = xs.antiNCxsec(Q2,GaS,MAvec[0],FS,muS)
    for i,MA in enumerate(MAvec):
        y[:,i] = xs.NCxsec(Q2,GaS,MA,FS,muS)
        ybar[:,i] = xs.antiNCxsec(Q2,GaS,MA,FS,muS)

    ax[0,1].plot(Q2,y,color='green',alpha=0.7)
    ax[0,1].plot(Q2,ymin,color='green',linewidth=3)
    ax[0,1].plot(Q2,ybar,color='lightgreen',alpha=0.7)
    ax[0,1].plot(Q2,yminbar,color='lightgreen',linewidth=3)
    ax[0,1].plot(Q2,yTruth,color='black')
    ax[0,1].plot(Q2,yTruthBar,color='dimgray')
    ax[0,1].errorbar(Q,data,yerr=dataErr,fmt='o',color='tomato')
    ax[0,1].errorbar(Q,databar,yerr=errbar,fmt='o',color='steelblue')
    ax[0,1].set_yscale('log')
    ax[0,1].set_title(r'$M_A$ from 0.75 to 1.5')

    # changing FS in bottom left
    muS = 0.
    FSvec  = np.linspace(-0.5,1.0,50)
    MA  = 1.061
    GaS = -.12
    y = np.zeros((len(Q2),len(FSvec)))
    iybar = np.zeros((len(Q2),len(FSvec)))
    ymin = xs.NCxsec(Q2,GaS,MA,FSvec[0],muS)
    yminbar = xs.antiNCxsec(Q2,GaS,MA,FSvec[0],muS)
    for i,FS in enumerate(FSvec):
        y[:,i] = xs.NCxsec(Q2,GaS,MA,FS,muS)
        ybar[:,i] = xs.antiNCxsec(Q2,GaS,MA,FS,muS)

    ax[1,0].plot(Q2,y,color='darkorange',alpha=0.7)
    ax[1,0].plot(Q2,ymin,color='darkorange',linewidth=3)
    ax[1,0].plot(Q2,ybar,color='sandybrown',alpha=0.7)
    ax[1,0].plot(Q2,yminbar,color='sandybrown',linewidth=3)
    ax[1,0].plot(Q2,yTruth,color='black')
    ax[1,0].plot(Q2,yTruthBar,color='dimgray')
    ax[1,0].errorbar(Q,data,yerr=dataErr,fmt='o',color='tomato')
    ax[1,0].errorbar(Q,databar,yerr=errbar,fmt='o',color='steelblue')
    ax[1,0].set_yscale('log')
    ax[1,0].set_title(r'$F_1^S$ from -0.5 to 1.0')

    # changing muS in bottom right
    muSvec = np.linspace(-1.,0.5,50)
    FS  = 0.
    MA  = 1.061
    GaS = -.12
    y = np.zeros((len(Q2),len(muSvec)))
    ybar = np.zeros((len(Q2),len(muSvec)))
    ymin = xs.NCxsec(Q2,GaS,MA,FS,muSvec[0])
    yminbar = xs.antiNCxsec(Q2,GaS,MA,FS,muSvec[0])
    for i,muS in enumerate(muSvec):
        y[:,i] = xs.NCxsec(Q2,GaS,MA,FS,muS)
        ybar[:,i] = xs.antiNCxsec(Q2,GaS,MA,FS,muS)

    ax[1,1].plot(Q2,y,color='orchid',alpha=0.7)
    ax[1,1].plot(Q2,ymin,color='orchid',linewidth=3)
    ax[1,1].plot(Q2,ybar,color='thistle',alpha=0.7)
    ax[1,1].plot(Q2,yminbar,color='thistle',linewidth=3)
    ax[1,1].plot(Q2,yTruth,color='black')
    ax[1,1].plot(Q2,yTruthBar,color='dimgray')
    ax[1,1].errorbar(Q,data,yerr=dataErr,fmt='o',color='tomato')
    ax[1,1].errorbar(Q,databar,yerr=errbar,fmt='o',color='steelblue')
    ax[1,1].set_yscale('log')
    ax[1,1].set_title(r'$\mu_S$ from -1. to .5')

def makePlotSD():
    Q2 = np.linspace(0.25,1.25,50)
    muS = 0.
    FS  = 0.
    MA  = 1.061
    GaS = -.12
    yTruthdiff = xs.NCxsecDiff(Q2,GaS,MA,FS,muS)
    yTruthsum  = xs.NCxsecSum(Q2,GaS,MA,FS,muS)

    # Data points
    Q,data,dataErr,databar,errbar = xs.getE734dat()

    errComb = np.sqrt(dataErr**2 + errbar**2)

    fig, ax = plt.subplots(nrows=2,ncols=2,sharex=True)

    # changing DeltaS in top left
    muS = 0.
    FS  = 0.
    MA  = 1.061
    GaSvec = np.linspace(-.5,.3,50)
    ydiff = np.zeros((len(Q2),len(GaSvec)))
    ysum  = np.zeros((len(Q2),len(GaSvec)))
    ymindiff = xs.NCxsecDiff(Q2,GaSvec[0],MA,FS,muS)
    yminsum  = xs.NCxsecSum(Q2,GaSvec[0],MA,FS,muS)
    for i,GaS in enumerate(GaSvec):
        ydiff[:,i] = xs.NCxsecDiff(Q2,GaS,MA,FS,muS)
        ysum[:,i]  = xs.NCxsecSum(Q2,GaS,MA,FS,muS)
    
    ax[0,0].plot(Q2,ydiff,color='steelblue',alpha=0.7)
    ax[0,0].plot(Q2,ymindiff,color='steelblue',linewidth=3)
    ax[0,0].plot(Q2,ysum,color='lightblue',alpha=0.7)
    ax[0,0].plot(Q2,yminsum,color='lightblue',linewidth=3)
    ax[0,0].plot(Q2,yTruthdiff,color='black')
    ax[0,0].plot(Q2,yTruthsum,color='dimgray')
    ax[0,0].errorbar(Q,data-databar,yerr=errComb,fmt='o',color='tomato')
    ax[0,0].errorbar(Q,data+databar,yerr=errComb,fmt='o',color='steelblue')
    ax[0,0].set_yscale('log')
    ax[0,0].set_title(r'$\Delta S$ from -0.5 to 0.3')

    # changing MA in top right
    muS = 0.
    FS  = 0.
    MAvec  = np.linspace(0.75,1.5,50)
    GaS = -.12
    ydiff = np.zeros((len(Q2),len(MAvec)))
    ysum  = np.zeros((len(Q2),len(MAvec)))
    ymindiff = xs.NCxsecDiff(Q2,GaS,MAvec[0],FS,muS)
    yminsum  = xs.NCxsecSum(Q2,GaS,MAvec[0],FS,muS)
    for i,MA in enumerate(MAvec):
        ydiff[:,i] = xs.NCxsecDiff(Q2,GaS,MA,FS,muS)
        ysum[:,i]  = xs.NCxsecSum(Q2,GaS,MA,FS,muS)

    ax[0,1].plot(Q2,ydiff,color='green',alpha=0.7)
    ax[0,1].plot(Q2,ymindiff,color='green',linewidth=3)
    ax[0,1].plot(Q2,ysum,color='lightgreen',alpha=0.7)
    ax[0,1].plot(Q2,yminsum,color='lightgreen',linewidth=3)
    ax[0,1].plot(Q2,yTruthdiff,color='black')
    ax[0,1].plot(Q2,yTruthsum,color='dimgray')
    ax[0,1].errorbar(Q,data-databar,yerr=errComb,fmt='o',color='tomato')
    ax[0,1].errorbar(Q,data+databar,yerr=errComb,fmt='o',color='steelblue')
    ax[0,1].set_yscale('log')
    ax[0,1].set_title(r'$M_A$ from 0.75 to 1.5')

    # changing FS in bottom left
    muS = 0.
    FSvec  = np.linspace(-0.5,1.0,50)
    MA  = 1.061
    GaS = -.12
    ydiff = np.zeros((len(Q2),len(FSvec)))
    ysum  = np.zeros((len(Q2),len(FSvec)))
    ymindiff = xs.NCxsecDiff(Q2,GaS,MA,FSvec[0],muS)
    yminsum  = xs.NCxsecSum(Q2,GaS,MA,FSvec[0],muS)
    for i,FS in enumerate(FSvec):
        ydiff[:,i] = xs.NCxsecDiff(Q2,GaS,MA,FS,muS)
        ysum[:,i]  = xs.NCxsecSum(Q2,GaS,MA,FS,muS)

    ax[1,0].plot(Q2,ydiff,color='darkorange',alpha=0.7)
    ax[1,0].plot(Q2,ymindiff,color='darkorange',linewidth=3)
    ax[1,0].plot(Q2,ysum,color='sandybrown',alpha=0.7)
    ax[1,0].plot(Q2,yminsum,color='sandybrown',linewidth=3)
    ax[1,0].plot(Q2,yTruthdiff,color='black')
    ax[1,0].plot(Q2,yTruthsum,color='dimgray')
    ax[1,0].errorbar(Q,data-databar,yerr=errComb,fmt='o',color='tomato')
    ax[1,0].errorbar(Q,data+databar,yerr=errComb,fmt='o',color='steelblue')
    ax[1,0].set_yscale('log')
    ax[1,0].set_title(r'$F_1^S$ from -0.5 to 1.0')

    # changing muS in bottom right
    muSvec = np.linspace(-1.,0.5,50)
    FS  = 0.
    MA  = 1.061
    GaS = -.12
    ydiff = np.zeros((len(Q2),len(muSvec)))
    ysum  = np.zeros((len(Q2),len(muSvec)))
    ymindiff = xs.NCxsecDiff(Q2,GaS,MA,FS,muSvec[0])
    yminsum  = xs.NCxsecSum(Q2,GaS,MA,FS,muSvec[0])
    for i,muS in enumerate(muSvec):
        ydiff[:,i] = xs.NCxsecDiff(Q2,GaS,MA,FS,muS)
        ysum[:,i]  = xs.NCxsecSum(Q2,GaS,MA,FS,muS)

    ax[1,1].plot(Q2,ydiff,color='orchid',alpha=0.7)
    ax[1,1].plot(Q2,ymindiff,color='orchid',linewidth=3)
    ax[1,1].plot(Q2,ysum,color='thistle',alpha=0.7)
    ax[1,1].plot(Q2,yminsum,color='thistle',linewidth=3)
    ax[1,1].plot(Q2,yTruthdiff,color='black')
    ax[1,1].plot(Q2,yTruthsum,color='dimgray')
    ax[1,1].errorbar(Q,data-databar,yerr=errComb,fmt='o',color='tomato')
    ax[1,1].errorbar(Q,data+databar,yerr=errComb,fmt='o',color='steelblue')
    ax[1,1].set_yscale('log')
    ax[1,1].set_title(r'$\mu_S$ from -1. to .5')

