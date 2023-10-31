import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
from numpy.linalg import eigh
import PCA
import timeit

logwave, scaledFlux = PCA.readAndReturnScaledTruncData('specgrid.fits','flux')


start_time = timeit.default_timer()
e1, e2, e3, e4, e5 = PCA.EigenVectorsThroughSVD(scaledFlux)
timeElapsed = timeit.default_timer() - start_time

print('Time it took to compute eigenvectors with numpy.linalg.svd is ' + str(timeElapsed) + ' seconds.')
PCA.makePlotMultiple(logwave,e1,e2,e3,e4,e5,r'$\log_{10} \lambda $','Eigenvector Value',['Vector 1', 'Vector 2','Vector 3', 'Vector 4', 'Vector 5'],'First Five Eigenvectors SVD','5EigenvecSVD.png')
PCA.makePlot(logwave,e1,r'$\log_{10} \lambda $','Eigenvector Value','First Eigenvector SVD','e1SVD.png')
PCA.makePlot(logwave,e2,r'$\log_{10} \lambda $','Eigenvector Value','Second Eigenvector SVD','e2SVD.png')
PCA.makePlot(logwave,e3,r'$\log_{10} \lambda $','Eigenvector Value','Third Eigenvector SVD','e3SVD.png')
PCA.makePlot(logwave,e4,r'$\log_{10} \lambda $','Eigenvector Value','Fourth Eigenvector SVD','e4SVD.png')
PCA.makePlot(logwave,e5,r'$\log_{10} \lambda $','Eigenvector Value','Fifth Eigenvector SVD','e5SVD.png')

start_time2 = timeit.default_timer()
E1,E2,E3,E4,E5 = PCA.CalculateCovAnd5EigenVec(scaledFlux)
timeElapsed2 = timeit.default_timer() - start_time2

print('Time it took to compute eigenvectors with numpy.linalg.eigh is ' + str(timeElapsed2) + ' seconds.')
PCA.makePlotMultiple(logwave,E1,E2,E3,E4,E5,r'$\log_{10} \lambda $','Eigenvector Value',['Vector 1', 'Vector 2','Vector 3', 'Vector 4', 'Vector 5'],'First Five Eigenvectors Smaller Subset','5EigenvecSmall.png')
PCA.makePlot(logwave,E1,r'$\log_{10} \lambda $','Eigenvector Value','First Eigenvector Smaller Subset','e1Small.png')
PCA.makePlot(logwave,E2,r'$\log_{10} \lambda $','Eigenvector Value','Second Eigenvector Smaller Subset','e2Small.png')
PCA.makePlot(logwave,E3,r'$\log_{10} \lambda $','Eigenvector Value','Third Eigenvector Smaller Subset','e3Small.png')
PCA.makePlot(logwave,E4,r'$\log_{10} \lambda $','Eigenvector Value','Fourth Eigenvector Smaller Subset','e4Small.png')
PCA.makePlot(logwave,E5,r'$\log_{10} \lambda $','Eigenvector Value','Fifth Eigenvector Smaller Subset','e5Small.png')


(U,D,VT) = np.linalg.svd(scaledFlux)
D[::-1].sort()
print('Condition number for R is ')
print(D[0]/D[len(D) - 1])

(U,D_cov,VT) = np.linalg.svd(np.dot(scaledFlux.T, scaledFlux)/len(scaledFlux))
D_cov[::-1].sort()
print('Condition number for the Covariance matrix is ')
print(D_cov[0]/D_cov[len(D_cov) - 1])