import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
from numpy.linalg import eigh
import PCA
import timeit

logwave, scaledFlux, flux, Norms, meanGalaxyFlux = PCA.readAndReturnMeansOriginalData('specgrid.fits')

Nc = 5

principalComponents = PCA.CalculateCovAndLEigenVec(scaledFlux, Nc)
reduced_wavelength_data= np.dot(principalComponents.T,scaledFlux.T)
estimateNormedSpectrum = np.dot(principalComponents, reduced_wavelength_data).T

c0List = []
c1List = []
c2List = []

galaxies = [0,1,2,3,4,5,6]

for num in galaxies:
    c0List.append(np.dot(principalComponents[:,0],estimateNormedSpectrum[num,:]))
    c1List.append(np.dot(principalComponents[:,1],estimateNormedSpectrum[num,:]))
    c2List.append(np.dot(principalComponents[:,2],estimateNormedSpectrum[num,:]))
    
    estimateOrginalSpectrum = Norms[num]*estimateNormedSpectrum[num,:] + np.average(estimateNormedSpectrum[num,:])#
    #estimateOrginalSpectrum = Norms[num]*estimateNormedSpectrum[num,:] + Norms[num]*meanGalaxyFlux[num]
    #estimateOrginalSpectrum = Norms[num]*estimateNormedSpectrum[num,:] + Norms[num]*np.average(estimateNormedSpectrum[num,:])
    #estimateOrginalSpectrum = (estimateNormedSpectrum[num] + meanGalaxyFlux[num])*Norms[num]
    
    plt.plot(logwave,estimateOrginalSpectrum, label = 'Estimate with Nc = ' + str(Nc))
    plt.plot(logwave, flux[num,:], label = 'Original Spectrum')
    plt.legend()
    plt.xlabel(r'$\log_{10} \lambda $')
    plt.ylabel(r'Spectrum $10^{-17} s^{−1} cm^{-2} A^{−1} \rm{erg}$')
    plt.title('Comparing PCA results with 5 Principle Components with Original Data')
    plt.savefig('PCAwithNc='+str(Nc)+'forGalaxy'+str(num)+'.png')
    plt.show()
    
plt.plot(c0List, c1List, '*')
plt.xlabel('Coefficient c0 for the first 7 Galaxies')
plt.ylabel('Coefficient c1 for the first 7 Galaxies')
plt.title('c0 versus c1')
plt.savefig('C0versusC1For5PrincipleComponents.png')
plt.show()

plt.plot(c0List, c2List, '*')
plt.xlabel('Coefficient c0 for the first 7 Galaxies')
plt.ylabel('Coefficient c2 for the first 7 Galaxies')
plt.title('c0 versus c2')
plt.savefig('C0versusC2For5PrincipleComponents.png')


#plt.plot(logwave, estimateNormedSpectrum[1,:], label = 'Nc = 5')
#plt.plot(logwave, scaledFlux[1,:], label = 'original data')
#plt.legend()
#plt.show()