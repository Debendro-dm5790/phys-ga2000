import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
from numpy.linalg import eigh
import PCA
import timeit

logwave, scaledFlux = PCA.readAndReturnScaledData('specgrid.fits','flux')

start_time = timeit.default_timer()

e1,e2,e3,e4,e5 = PCA.CalculateCovAnd5EigenVec(scaledFlux)

timeElapsed = timeit.default_timer() - start_time

print('Time it took to compute eigenvectors with numpy.linalg.eigh is ' + str(timeElapsed) + ' seconds.')
PCA.makePlotMultiple(logwave,e1,e2,e3,e4,e5,r'$\log_{10} \lambda $','Eigenvector Value',['Vector 1', 'Vector 2','Vector 3', 'Vector 4', 'Vector 5'],'First Five Eigenvectors','5Eigenvec.png')
PCA.makePlot(logwave,e1,r'$\log_{10} \lambda $','Eigenvector Value','First Eigenvector','e1.png')
PCA.makePlot(logwave,e2,r'$\log_{10} \lambda $','Eigenvector Value','Second Eigenvector','e2.png')
PCA.makePlot(logwave,e3,r'$\log_{10} \lambda $','Eigenvector Value','Third Eigenvector','e3.png')
PCA.makePlot(logwave,e4,r'$\log_{10} \lambda $','Eigenvector Value','Fourth Eigenvector','e4.png')
PCA.makePlot(logwave,e5,r'$\log_{10} \lambda $','Eigenvector Value','Fifth Eigenvector','e5.png')