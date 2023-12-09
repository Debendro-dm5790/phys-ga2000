import numpy as np
import matplotlib.pyplot as plt

'''
Define the the size of the x-directional spacial grid
'''
N = np.int32(1000)

'''
Define the length of the infinite square well potential. Unit is meters
'''
L = np.float32(1e-8)

'''
Define the spatial grid
'''
xgrid = np.linspace(0, L, N, dtype=np.float32)

'''
Define the initial wavefunction along with its constant parameters
'''
x_0 = L/2
sigma = np.float32(1e-10)
kappa = np.float32(5e10)

psi = np.exp(-1*(xgrid - x_0)**2/(2*sigma**2))*np.exp(kappa*xgrid*1j)

'''
Define the nonzero elements of the matrix A. 
'''
a = L/N
h_bar = np.float32(6.62607015e-34)
m = np.float32(9.109e-31)
h = np.float32(1e-18)


a1 = 1 + ((h*h_bar)/(2*m*a**2))*1j
a2 = -1*h*(h_bar/(4*m*a**2))*1j

b1 = 1 - ((h*h_bar)/(2*m*a**2))*1j 
b2 = h*(h_bar/(4*m*a**2))*1j

matrixDim = N

A = np.zeros((matrixDim, matrixDim)).astype(complex)

for i in range(matrixDim - 1):
	A[i, i] = a1
	A[i , i+1] = a2
	A[i+1, i] = a2

'''
Define a function that performs one step of the Crank-Nicolson method
'''

def OneStep(A,psi):
    v = np.zeros(matrixDim, dtype = np.float32).astype(complex)
    v[0] = b1 * psi[0] + b2 * psi[1]
    v[matrixDim - 1] = b2 * psi[matrixDim - 2] + b1 * psi[matrixDim - 1]
	
    for i in list(np.arange(1, matrixDim - 1)):
        v[i] = b1 * psi[i] + b2 * (psi[i+1] + psi[i-1])
        
    psi = np.linalg.solve(A, v)
    
    psi[0] = 0
    psi[len(psi)-1] = 0

    return psi

'''
Plot the initial wavefunction
'''
plt.plot(xgrid, np.real(psi))
plt.xlabel('Length (meters)')
plt.ylabel('Real Part of the Wavefunction')
plt.title('Real Part of Initial Wavefunction')
plt.savefig('InitialWF.png')

'''
Start the propagation code
'''
currentWavefunction = psi.copy()

for t in range(5000):
    currentWavefunction = OneStep(A,currentWavefunction)
    print(t)
    if t == 500: 
       plt.plot(xgrid, np.real(currentWavefunction))
       plt.xlabel('Length (meters)')
       plt.ylabel('Real Part of the Wavefunction')
       plt.title('Real Part of Developed Wavefunction at time ' + str(500*h) + ' seconds')
       plt.savefig('NewWF_1.png')
    elif t == 1000:
       plt.plot(xgrid, np.real(currentWavefunction))
       plt.xlabel('Length (meters)')
       plt.ylabel('Real Part of the Wavefunction')
       plt.title('Real Part of Developed Wavefunction at time ' + str(1000*h) + ' seconds')
       plt.savefig('NewWF_2.png')
    elif t == 1500:
       plt.plot(xgrid, np.real(currentWavefunction))
       plt.xlabel('Length (meters)')
       plt.ylabel('Real Part of the Wavefunction')
       plt.title('Real Part of Developed Wavefunction at time ' + str(1500*h) + ' seconds')
       plt.savefig('NewWF_3.png')
    elif t == 2000:
       plt.plot(xgrid, np.real(currentWavefunction))
       plt.xlabel('Length (meters)')
       plt.ylabel('Real Part of the Wavefunction')
       plt.title('Real Part of Developed Wavefunction at time ' + str(2000*h) + ' seconds')
       plt.savefig('NewWF_4.png')
    elif t == 2500:
       plt.plot(xgrid, np.real(currentWavefunction))
       plt.xlabel('Length (meters)')
       plt.ylabel('Real Part of the Wavefunction')
       plt.title('Real Part of Developed Wavefunction at time ' + str(2500*h) + ' seconds')
       plt.savefig('NewWF_5.png')
    elif t == 3000:
       plt.plot(xgrid, np.real(currentWavefunction))
       plt.xlabel('Length (meters)')
       plt.ylabel('Real Part of the Wavefunction')
       plt.title('Real Part of Developed Wavefunction at time ' + str(3000*h) + ' seconds')
       plt.savefig('NewWF_6.png')
    elif t == 3500:
       plt.plot(xgrid, np.real(currentWavefunction))
       plt.xlabel('Length (meters)')
       plt.ylabel('Real Part of the Wavefunction')
       plt.title('Real Part of Developed Wavefunction at time ' + str(3500*h) + ' seconds')
       plt.savefig('NewWF_7.png')
    elif t == 4000:
       plt.plot(xgrid, np.real(currentWavefunction))
       plt.xlabel('Length (meters)')
       plt.ylabel('Real Part of the Wavefunction')
       plt.title('Real Part of Developed Wavefunction at time ' + str(4000*h) + ' seconds')
       plt.savefig('NewWF_8.png')
    elif t == 4500:
       plt.plot(xgrid, np.real(currentWavefunction))
       plt.xlabel('Length (meters)')
       plt.ylabel('Real Part of the Wavefunction')
       plt.title('Real Part of Developed Wavefunction at time ' + str(4500*h) + ' seconds')
       plt.savefig('NewWF_9.png')
    elif t == 4999:
       plt.plot(xgrid, np.real(currentWavefunction))
       plt.xlabel('Length (meters)')
       plt.ylabel('Real Part of the Wavefunction')
       plt.title('Real Part of Developed Wavefunction at time ' + str(4999*h) + ' seconds')
       plt.savefig('NewWF_10.png')

    
    
