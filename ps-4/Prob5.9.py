import numpy as np
import matplotlib.pyplot as plt

######################################################################
# http://www-personal.umich.edu/~mejn/computational-physics/gaussxw.py
# Functions to calculate integration points and weights for Gaussian
# quadrature
#
# x,w = gaussxw(N) returns integration points x and integration
#           weights w such that sum_i w[i]*f(x[i]) is the Nth-order
#           Gaussian approximation to the integral int_{-1}^1 f(x) dx
# x,w = gaussxwab(N,a,b) returns integration points and weights
#           mapped to the interval [a,b], so that sum_i w[i]*f(x[i])
#           is the Nth-order Gaussian approximation to the integral
#           int_a^b f(x) dx
#
# This code finds the zeros of the nth Legendre polynomial using
# Newton's method, starting from the approximation given in Abramowitz
# and Stegun 22.16.6.  The Legendre polynomial itself is evaluated
# using the recurrence relation given in Abramowitz and Stegun
# 22.7.10.  The function has been checked against other sources for
# values of N up to 1000.  It is compatible with version 2 and version
# 3 of Python.
#
# Written by Mark Newman <mejn@umich.edu>, June 4, 2011
# You may use, share, or modify this file freely
#
######################################################################

from numpy import ones,copy,cos,tan,pi,linspace

def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w
'''
This function rescales and returns the sample points and weights to the bounds a,b
'''
def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w
############################################################################################################
#begin my code
 
'''
We implement a function called integrand that returns the dimensionless integrand
presented in the exercise. It does not return the extra dimensionful factors that
are multiplied to the integral.
'''
def integrand(x):
    return (x**4*np.exp(x))/((np.exp(x) - 1)**2)

'''
We implement a function called Cv that calculates the heat capacity of our simulated
system. If first uses the function gaussxw to get the sampled values and weights
for Gauss-Legendre quadrature and then rescales them to the bounds presented in the
exercise description by calling the gaussxwab function. Using the rescaled sampled
points and weights, we compute the integral and then we multiply it by the multiplicative
factor made from the sample volume, the number density, the Boltzmann constant, the 
temperature, and the Debeye temperature which are the function's input parameters 
vol, rho, kB, temp, and debT, respectively. The function also has an input 
parameter called N, which is the number of sample points we want. '
'''
def Cv(temp, vol, rho, debT, N, kB):
    x,w = gaussxw(N)
    a = 0
    #xp = 0.5*(b-a)*x + 0.5*(b+a) # sample points, rescaled to bounds a,b
    #wp = 0.5*(b-a)*w # rescale weights to bounds a, b
    xp, wp = gaussxwab(N,a,debT/temp)
    s = 9*vol*rho*kB*(temp/debT)**3*sum(integrand(xp)*wp) # add them up!
    return s

'''
We define the constants V, rho debTemp, N, and kB as the sample volume, the number
density, the Debeye temperature, the number of sample points, and the Boltzmann 
constant, respectively. We use 50 sample points. We also define an array of 
temperatures ranging from 5 to 500 Kelvin. We store it in a variable called TArray.
We also define an array to store the corresponding heat capacities. We initialize 
it as an array of zeros with the same size as the array TArray. We store this array 
in a variable called heatCaps. 
'''

V = np.float32(1000/(100**3))   # Volume in cubic meters
rho = np.float32(6.022e28)      # Density in atoms per cubic meter
debTemp = np.int32(428)      # Debye Temperature 
N = np.int32(50)              # Number of sample points
kB = np.float32(1.380649e-23)   # Boltzmann constant

TArray = np.linspace(5,500,500, dtype = np.float32)
heatCaps = np.float32(np.zeros(len(TArray)))

'''
With a for loop, we fill the array heatCaps with the corresponding heat capacity
and print a plot of the heat capacity versus temperature for our simulated sample
of Aluminum. We save the plot as a .png file called HeatCap.png
'''
counter = 0
for temp in TArray:
    heatCaps[counter] = Cv(temp, V, rho, debTemp, N, kB)
    counter += 1

plt.plot(TArray, heatCaps)
plt.xlabel('Temperature in Kelvin')
plt.ylabel('Heat Capacities Joule/Kelvin')
plt.title('Heat Capacity Plot of Aluminum')
plt.savefig('HeatCap.png')
plt.show()

'''
We repeat this procedure for sample sizes of 10,20,30,40,50,60, and 70, which we 
store in an array called NArray. For heat sample size, we determine the corresponding
heat capacity curve and store it into a list called heatCapList. This list is two
dimensional. We plot the individual heat capacitity plots onto a single point to 
see whether these curves converge. They indeed do and the output is saved as a .png
file HeatCapConvergenceOverall.png. This plot tests overall convergence.
'''
heatCapList = []
NArray = np.array([10,20,30,40,50,60,70], dtype = np.int32)

for Num in NArray:
    counter = 0
    heatCaps = np.float32(np.zeros(len(TArray)))
    for temp in TArray:
        heatCaps[counter] = Cv(temp, V, rho, debTemp, Num, kB)
        counter += 1
    heatCapList.append(heatCaps)
    
plt.plot(TArray, heatCapList[0])
plt.plot(TArray, heatCapList[1])
plt.plot(TArray, heatCapList[2])
plt.plot(TArray, heatCapList[3])
plt.plot(TArray, heatCapList[4])
plt.plot(TArray, heatCapList[5])
plt.plot(TArray, heatCapList[6])
plt.xlabel('Temperature in Kelvin')
plt.ylabel('Heat Capacities Joule/Kelvin')
plt.title('Convergence of Heat Capacity for Various N')
plt.legend(['N = 10', 'N = 20', 'N = 30', 'N = 40', 'N = 50', 'N = 60', 'N = 70'])
plt.savefig('HeatCapConvergenceOverall.png')
plt.show()

'''
We use the two-dimensional list to determine whehter the computed heat capacity
at a certain temperature converge as the number of sample points vary. We make 
plots and save them as .png files. We indeed see that the values are precise.  
'''

sampleCV1 = [heatCapList[0][0],heatCapList[1][0],heatCapList[2][0],heatCapList[3][0], heatCapList[4][0], heatCapList[5][0], heatCapList[6][0]]
sampleCV2 = [heatCapList[0][50],heatCapList[1][50],heatCapList[2][50],heatCapList[3][50], heatCapList[4][50], heatCapList[5][50], heatCapList[6][50]]
sampleCV3 = [heatCapList[0][100],heatCapList[1][100],heatCapList[2][100],heatCapList[3][100], heatCapList[4][100], heatCapList[5][100], heatCapList[6][100]]
sampleCV4 = [heatCapList[0][400],heatCapList[1][400],heatCapList[2][400],heatCapList[3][400], heatCapList[4][400], heatCapList[5][400], heatCapList[6][400]]

plt.plot(NArray, sampleCV1, '*')
plt.xlabel('N, number of sample points')
plt.ylabel('Heat Capacities Joule/Kelvin')
plt.legend(['T = ' + str(TArray[0]) + ' K'])
plt.title('Convergence of Heat Capacities as N Varies')
plt.savefig('HeatCapConvergenceSample1.png')
plt.show()

plt.plot(NArray, sampleCV2, '*')
plt.xlabel('N, number of sample points')
plt.ylabel('Heat Capacities Joule/Kelvin')
plt.legend(['T = ' + str(TArray[50]) + ' K'])
plt.title('Convergence of Heat Capacities as N Varies')
plt.savefig('HeatCapConvergenceSample2.png')
plt.show()

plt.plot(NArray, sampleCV3, '*')
plt.xlabel('N, number of sample points')
plt.ylabel('Heat Capacities Joule/Kelvin')
plt.legend(['T = ' + str(TArray[100]) + ' K'])
plt.title('Convergence of Heat Capacities as N Varies')
plt.savefig('HeatCapConvergenceSample3.png')
plt.show()

plt.plot(NArray, sampleCV4, '*')
plt.xlabel('N, number of sample points')
plt.ylabel('Heat Capacities Joule/Kelvin')
plt.legend(['T = ' + str(TArray[400]) + ' K'])
plt.title('Convergence of Heat Capacities as N Varies')
plt.savefig('HeatCapConvergenceSample4.png')
plt.show()