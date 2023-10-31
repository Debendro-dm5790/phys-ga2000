import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh,svd

def readAndReturnScaledData(filename, what):
    hdu_list = astropy.io.fits.open(filename)
    logwave = hdu_list['LOGWAVE'].data
    flux = hdu_list['FLUX'].data
    
    scaledFlux = flux.copy()
    A = np.zeros(len(flux))
    meanGalaxyFlux = np.zeros(len(flux))

    for i in range(len(flux)):
        A[i] = np.sum(flux[i,0:len(logwave) - 1]*10**(logwave[1:len(logwave)]) - flux[i,0:len(logwave) - 1]*10**(logwave[0:len(logwave)-1]))
        scaledFlux[i] = flux[i]/A[i]
        scaledFlux[i] = scaledFlux[i] - np.average(scaledFlux[i])
        meanGalaxyFlux[i] = np.average(scaledFlux[i])
        
    if what == 'flux':
        return logwave, scaledFlux
    elif what == 'sum':
        return logwave, A
    elif what == 'mean':
        return logwave, meanGalaxyFlux
    
def readAndReturnScaledTruncData(filename, what):
    hdu_list = astropy.io.fits.open(filename)
    logwave = hdu_list['LOGWAVE'].data
    flux = hdu_list['FLUX'].data[0:1000,:]
    
    scaledFlux = flux.copy()
    A = np.zeros(len(flux))
    meanGalaxyFlux = np.zeros(len(flux))

    for i in range(len(flux)):
        A[i] = np.sum(flux[i])
        scaledFlux[i] = flux[i]/A[i]
        scaledFlux[i] = scaledFlux[i] - np.average(scaledFlux[i])
        meanGalaxyFlux[i] = np.average(scaledFlux[i])
        
    if what == 'flux':
        return logwave, scaledFlux
    elif what == 'sum':
        return logwave, A
    elif what == 'mean':
        return logwave, meanGalaxyFlux
    

def makePlot(x,y,xlabel,ylabel,title,saveName):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(saveName)
    plt.close()
    
def makePlotMultiple(x,y1,y2,y3,y4,y5,xlabel,ylabel,legend,title,saveName):
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.plot(x,y4)
    plt.plot(x,y5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.title(title)
    plt.savefig(saveName)
    plt.close()
    
def CalculateCovAnd5EigenVec(scaledFlux):
    values, vectors = eigh(np.dot(scaledFlux.T, scaledFlux))
    idx = values.argsort()[::-1]   
    values = values[idx]
    vectors = vectors[:,idx]
    return vectors[0], vectors[1], vectors[2], vectors[3], vectors[4]

def CalculateCovAndLEigenVec(scaledFlux, l):
    values, vectors = eigh(np.dot(scaledFlux.T, scaledFlux))
    idx = values.argsort()[::-1]   
    values = values[idx]
    vectors = vectors[:,idx]
    eigvec=vectors[:,:l]
    return eigvec

def calculateCovAndEigenVal(scaledFlux):
    Cov = np.dot(scaledFlux.T, scaledFlux)
    values, vectors = eigh(Cov)
    return Cov, values

#def EigenVectorsThroughSVD(scaledFlux):
#    (U,D,VT) = np.linalg.svd(scaledFlux)
#   idx = np.argsort(D**2)[::-1]   
#   V = VT.T[:,idx]
#   return V[:, 0],V[:, 1],V[:, 2],V[:, 3],V[:, 4]
    
def EigenVectorsThroughSVD(scaledFlux):
    (U,D,VT) = np.linalg.svd(scaledFlux)
    V = VT
    return V[:, 0],V[:, 1],V[:, 2],V[:, 3],V[:, 4]

def readAndReturnMeansOriginalData(filename):
    hdu_list = astropy.io.fits.open(filename)
    logwave = hdu_list['LOGWAVE'].data
    flux = hdu_list['FLUX'].data
    
    scaledFlux = flux.copy()
    A = np.zeros(len(flux))
    meanGalaxyFlux = np.zeros(len(flux))

    for i in range(len(flux)):
        A[i] = np.sum(flux[i,0:len(logwave) - 1]*10**(logwave[1:len(logwave)]) - flux[i,0:len(logwave) - 1]*10**(logwave[0:len(logwave)-1]))
        scaledFlux[i] = flux[i]/A[i]
        scaledFlux[i] = scaledFlux[i] - np.average(scaledFlux[i])
        meanGalaxyFlux[i] = np.average(scaledFlux[i])
    
    return logwave, scaledFlux, flux, A, meanGalaxyFlux
    