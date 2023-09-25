# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import timeit

'''
We implement a function that performs matrix multiplication with a nested 
for loop. The function takes the matrix dimension N as an input and creates
three N x N matrices called A, B, and C. Matrices A and B are matrices
filled with ones while C is a matrix filled with zeros. The goal is to 
calculate C = AB. The for loops fill up the elements of the matrix C.
We use the python module timeit to determine the time it takes the 
computer to determine the matrix C. This time is stored
in the variable called timeElapsed. The function returns timeElapsed.
'''
def matrixMultForLoops(N):
    A = np.int32(np.ones([N,N]))
    B = A.copy()
    C = np.int32(np.zeros([N,N]))
    
    start_time = timeit.default_timer()
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i,j] += A[i,k]*B[k,j] 
                
    timeElapsed = timeit.default_timer() - start_time
    
    return timeElapsed

'''
We implement a function that performs matrix multplication with numpy's 
method dot(). The function takes the matrix dimension N as an input and 
creates two N x N matrices of all ones called A and B. The goal is to 
calculate C = AB. We again use the python module timeit to determine the 
time it takes the computer to determine the matrix C. This time is stored
in the variable called timeElapsed. The function returns timeElapsed. 
'''

def matrixMultDot(N):
    A = np.int32(np.ones([N,N]))
    B = A.copy()
    
    start_time = timeit.default_timer()
    
    C = np.int32(np.dot(A,B))
    
    timeElapsed = timeit.default_timer() - start_time
    
    return timeElapsed

def powerLawModel(x,a,p):
    return a*x**p

'''
We create an array NArray of 32-bit integers whose elements are the various
matrix dimensions we want to test. We consider the following dimensions:
10,20,30,40,50,60,70,80,90,100,200,300,400,500, and 600. We use a for loop
to iterate through all of these dimensions. For each iteration, we call
the first and second functions and get the times it took the computer to 
calculat the matrix product with for loops and numpy.dot(). These times
are stored in the arrays timeArrayForLoop and timeArrayDot, respectively.
'''
NArray = np.int32(np.array([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600]))
timeArrayForLoop = np.float32(np.zeros(len(NArray)))
timeArrayDot = np.float32(np.zeros(len(NArray)))

for i in range(len(NArray)):
    print("Matrix dimension is " + str(NArray[i]))
    timeArrayForLoop[i] = matrixMultForLoops(NArray[i])
    timeArrayDot[i] = matrixMultDot(NArray[i])
    
'''
We make two log-log plots. In the first (second) plot we plot the matrix 
dimensions and the execution times for the for loop (numpy.dot()).
For each plot, we perform a power law fit to determine the appropriate
function of the form time = a*(matrix dimension)^p that fits the data. 
We then determine the fitted value of the powers. We expect the power to 
be three in each case and we get a value close to three.
'''
    
Ncontinuous=np.linspace(1,700,10000)
bestParamsForLoop = curve_fit(powerLawModel, NArray[3:], timeArrayForLoop[3:])
bestParamsDot = curve_fit(powerLawModel, NArray[2:], timeArrayDot[2:])

'''
We see that numpy.dot() has a faster execution time
'''

plt.loglog(NArray, timeArrayForLoop, '*', color='blue')
plt.loglog(Ncontinuous, bestParamsForLoop[0][0]*Ncontinuous**bestParamsForLoop[0][1])
plt.xlabel('N, Matrix Size')
plt.ylabel('Computation Time, Seconds')
plt.title('Computation Time for Matrix Multiplication with For Loops')
plt.savefig('MatrixMultForLoop.png')
plt.show()

print("The power of N for for loops is " + str(bestParamsForLoop[0][1]))

plt.loglog(NArray, timeArrayDot, '*', color='blue')
plt.loglog(Ncontinuous, bestParamsDot[0][0]*Ncontinuous**bestParamsDot[0][1])
plt.xlabel('N, Matrix Size')
plt.ylabel('Computation Time, Seconds')
plt.title('Computation Time for Matrix Multiplication with Numpy.Dot()')
plt.savefig('MatrixMultDot.png')
plt.show()
    
print("The power of N for numpy.dot() is " + str(bestParamsDot[0][1]))