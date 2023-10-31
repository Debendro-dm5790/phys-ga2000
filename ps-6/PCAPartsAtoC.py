import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
from numpy.linalg import eigh

hdu_list = astropy.io.fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

#Sample Galaxies are plotted. 

plt.plot(logwave,flux[0])
plt.xlabel(r'$\log_{10} \lambda $')
plt.ylabel(r'Spectrum $10^{-17} s^{−1} cm^{-2} A^{−1} \rm{erg}$')
plt.title('First Galaxy Spectrum')
plt.savefig('Galaxy1.png')
plt.show()

plt.plot(logwave,flux[1], 'r')
plt.xlabel(r'$\log_{10} \lambda $')
plt.ylabel(r'Spectrum $10^{-17} s^{−1} cm^{-2} A^{−1} \rm{erg}$')
plt.title('Second Galaxy Spectrum')
plt.savefig('Galaxy2.png')
plt.show()

plt.plot(logwave,flux[20], 'g')
plt.xlabel(r'$\log_{10} \lambda $')
plt.ylabel(r'Spectrum $10^{-17} s^{−1} cm^{-2} A^{−1} \rm{erg}$')
plt.title('Twenty-First Galaxy Spectrum')
plt.savefig('Galaxy21.png')
plt.show()

plt.plot(logwave,flux[50], 'm')
plt.xlabel(r'$\log_{10} \lambda $')
plt.ylabel(r'Spectrum $10^{-17} s^{−1} cm^{-2} A^{−1} \rm{erg}$')
plt.title('Fifty-First Galaxy Spectrum')
plt.savefig('Galaxy51.png')
plt.show()

# Now we will normalize the flux data and scale off the mean normalized flux
scaledFlux = flux.copy()
A = np.zeros(len(flux))
meanGalaxyFlux = np.zeros(len(flux))

for i in range(len(flux)):
    A[i] = np.sum(flux[i,0:len(logwave) - 1]*10**(logwave[1:len(logwave)]) - flux[i,0:len(logwave) - 1]*10**(logwave[0:len(logwave)-1]))
    scaledFlux[i] = flux[i]/A[i]
    scaledFlux[i] = scaledFlux[i] - np.average(scaledFlux[i])
    meanGalaxyFlux[i] = np.average(scaledFlux[i])
   
# Now we will plot the scaled and normalized versions of the previous plots
plt.plot(logwave,scaledFlux[0])
plt.xlabel(r'$\log_{10} \lambda $')
plt.ylabel('Normalized Spectrum')
plt.title('Normalized First Galaxy Spectrum')
plt.savefig('Galaxy1Normed.png')
plt.show()

plt.plot(logwave,scaledFlux[1], 'r')
plt.xlabel(r'$\log_{10} \lambda $')
plt.ylabel('Normalized Spectrum')
plt.title('Normalized Second Galaxy Spectrum')
plt.savefig('Galaxy2Normed.png')
plt.show()

plt.plot(logwave,scaledFlux[20], 'g')
plt.xlabel(r'$\log_{10} \lambda $')
plt.ylabel('Normalized Spectrum')
plt.title('Normalized Twenty-First Galaxy Spectrum')
plt.savefig('Galaxy21Normed.png')
plt.show()

plt.plot(logwave,scaledFlux[50], 'm')
plt.xlabel(r'$\log_{10} \lambda $')
plt.ylabel('Normalized Spectrum')
plt.title('Normalized Fifty-First Galaxy Spectrum')
plt.savefig('Galaxy51Normed.png')
plt.show() 


