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

    
    estimateOrginalSpectrum = Norms[3]*estimateNormedSpectrum[3,:] + np.average(estimateNormedSpectrum[3,:])
    #estimateOrginalSpectrum = Norms[num]*estimateNormedSpectrum[num,:] + Norms[num]*np.average(estimateNormedSpectrum[num,:])
    
    resid = ((estimateOrginalSpectrum - flux[3,:])/(flux[3,:]))**2
    
    plt.plot(logwave, resid)
    plt.xlabel(r'$\log_{10} \lambda $')
    plt.ylabel('Squared Fractional Residual')
    plt.title('Residual Plot for the Fourth Galaxy and Nc = ' + str(Nc))
    plt.savefig('ResidualswithNc='+str(Nc)+'forFourthGalaxy.png')
    plt.show()
    
for Nc in NcList:
    principalComponents = relevantPCs[:,:Nc]
    reduced_wavelength_data= np.dot(principalComponents.T,scaledFlux.T)
    estimateNormedSpectrum = np.dot(principalComponents, reduced_wavelength_data).T

    
    estimateOrginalSpectrum = Norms[3]*estimateNormedSpectrum[3,:] + Norms[3]*meanGalaxyFlux[3]
    #estimateOrginalSpectrum = Norms[num]*estimateNormedSpectrum[num,:] + Norms[num]*np.average(estimateNormedSpectrum[num,:])
    
    resid = ((estimateOrginalSpectrum - flux[3,:])/(flux[3,:]))**2
    
    plt.plot(logwave[0:3900], resid[0:3900])
    plt.xlabel(r'$\log_{10} \lambda $')
    plt.ylabel('Squared Fractional Residual')
    plt.title('Residual Plot for the Fourth Galaxy and Nc = ' + str(Nc))
    plt.savefig('ResidualswithNc='+str(Nc)+'forFourthGalaxyTrunc.png')
    plt.show()