import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
from numpy.linalg import eigh
import PCA
import timeit

logwave, scaledFlux, flux, Norms, meanGalaxyFlux = PCA.readAndReturnMeansOriginalData('specgrid.fits')

NcList = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

relevantPCs = PCA.CalculateCovAndLEigenVec(scaledFlux, 20)

for Nc in NcList:
    principalComponents = relevantPCs[:,:Nc]
    reduced_wavelength_data= np.dot(principalComponents.T,scaledFlux.T)
    estimateNormedSpectrum = np.dot(principalComponents, reduced_wavelength_data).T

    resid = ((estimateNormedSpectrum[3,:] - scaledFlux[3,:])/(scaledFlux[3,:]))**2
    
    plt.plot(logwave, estimateNormedSpectrum[3,:], label = 'PCA estimate')
    plt.plot(logwave, scaledFlux[3,:], label = 'Actual Normalized Spectrum')
    plt.xlabel(r'$\log_{10} \lambda $')
    plt.ylabel('Normalized Scaled Spectrum')
    plt.title('Comparing PCA Normed Estimate and Actual Normed Data for the Fourth Galaxy and Nc = ' + str(Nc))
    plt.legend()
    plt.savefig('PCAComparewithNc='+str(Nc)+'forFourthGalaxyScaledNormed.png')
    plt.show()
    
    plt.plot(logwave, resid)
    plt.xlabel(r'$\log_{10} \lambda $')
    plt.ylabel('Squared Fractional Residual')
    plt.title('Residual Plot for the Fourth Galaxy and Nc = ' + str(Nc))
    plt.savefig('ResidualswithNc='+str(Nc)+'forFourthGalaxyScaledNormed.png')
    plt.show()
    
    plt.plot(logwave[0:2000], resid[0:2000])
    plt.xlabel(r'$\log_{10} \lambda $')
    plt.ylabel('Squared Fractional Residual')
    plt.title('Left Truncated Residual Plot for the Fourth Galaxy and Nc = ' + str(Nc))
    plt.savefig('ResidualswithNc='+str(Nc)+'forFourthGalaxyScaledNormedLeftTruncated.png')
    plt.show()
    
    plt.plot(logwave[2500:], resid[2500:])
    plt.xlabel(r'$\log_{10} \lambda $')
    plt.ylabel('Squared Fractional Residual')
    plt.title('Right Truncated Residual Plot for the Fourth Galaxy and Nc = ' + str(Nc))
    plt.savefig('ResidualswithNc='+str(Nc)+'forFourthGalaxyScaledNormedRightTruncated.png')
    plt.show()
    
