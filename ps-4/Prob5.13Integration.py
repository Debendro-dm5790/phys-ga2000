import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.special import roots_hermite
from numpy.polynomial.hermite import hermgauss
from scipy import integrate

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

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w
############################################################################################################
#begin my code

def H(n,x):
    H_previous_previous = 1
    H_previous = 2*x
    H_current = 0
    
    if n == 0:
        H_current = 1
    elif n == 1:
        H_current = 2*x
    else:
        for i in range(n-1):
            H_current = 2*x*H_previous - 2*(i+1)*H_previous_previous
            H_previous_previous = H_previous
            H_previous = H_current
         
    return H_current
'''
We first the rescale the integration limits to -1 and 1 and rewrite the integrand.
This is implemented in the function called integrand(z,n), where n is the index
of the wavefunction and the Hermite polynomial. We use Gauss-Legendre and 
Gauss-Hermite quadrature to compute the root neab square position for n = 5 and 100
sample points. For each case we get a value of approximate 2.345
'''

def integrand(z,n):
    return (((1+z**2)*z**2)/(1-z**2)**4)*np.exp(-(z/(1-z**2))**2)*H(n,z/(1-z**2))**2

def rootMeanSquare(n, N):
    x,w = gaussxw(N)
    xp, wp = gaussxwab(N,-1,1)
    rms = np.sqrt((1/(2**n*m.factorial(n)*np.sqrt(np.pi)))*sum(integrand(xp,n)*wp)) # add them up!
    return rms

rms = rootMeanSquare(5, 100)

print('The uncertainty in position for the n = 5 state is ' + str(rms) + ' via Gauss-Legendre quadrature.')

def rootMeanSquareHermite(a, b, n, N):
    x,w = roots_hermite(N) # hermite polynomial roots
    #xp = 0.5*(b-a)*x + 0.5*(b+a) # sample points, rescaled to bounds a,b
    #wp = 0.5*(b-a)*w # rescale weights to bounds a, b
    xp, wp = gaussxwab(N,a,b)
    rms = np.sqrt((1/(2**n*m.factorial(n)*np.sqrt(np.pi)))*sum(integrand(xp,n)*wp)) # add them up!
    return rms


rms2 = rootMeanSquareHermite(-1, 1, 5, 100)

print('The uncertainty in position for the n = 5 state is ' + str(rms) + ' via Guass-Hermite quadrature')

xSimpson = np.linspace(-0.99999999, 0.99999999, 800001)
n = 5
ySimpson = integrand(xSimpson, n)
integralSimp = integrate.simps(ySimpson, xSimpson)
rmsExact = np.sqrt((1/(2**n*m.factorial(n)*np.sqrt(np.pi)))*integralSimp)
print('The more exact value of the uncertainty in positon is ' + str(rmsExact))