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

'''
We again use the functions gaussxw() and gaussxwab()
'''

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

'''
We define the rescaled version of the integrand, which scales out the a**4 term in
the original integrand's denominator and define the new integrand in a function
called integrand(). This means that the limits of integration are 0 and 1 instead
of 0 and a.'
'''
def integrand(x):
    return 1/np.sqrt(1 - x**4)

'''
We define a function called Period() which takes as input the amplitude, mass, and
the number of sample points. These correspond to the input variables a, mass, and 
N. The function computes the scaled integral using Gauss-Legendre quadrature and 
multiplies it by the appropriate constants. The period is returned. 
'''

def Period(a, mass, N):
    x,w = gaussxw(N)
    xp, wp = gaussxwab(N,0,1)
    s = np.sqrt(8*mass)*(1/a)*sum(integrand(xp)*wp) # add them up!
    return s

'''
We consider a sample size of 20 and a mass of 1. We make a plot of amplitude versus
Period and save it as a .png file called Anharmonic.png
'''
aArray = np.linspace(0,2,500, dtype = np.float32)
periodArray = np.zeros(len(aArray))
N = 20
mass = 1

counter = 0
for a in aArray:
   periodArray[counter] = Period(a, mass, N)
   counter += 1
   
plt.plot(aArray, periodArray)
plt.xlabel('Amplitude a, meters')
plt.ylabel('Period, seconds')
plt.title('Period of a Anharmonic Osscillator')
plt.savefig('Anharmonic.png')
plt.show()

'''
Since the energy is conserved, we can think of the system as a particle rolling
along the potential landscape. We note that V(x) = x^4 gets steeper as x increases 
at a much faster rate than V(x) = x^2 and so it should roll with greater speed as
its amplitude increases. The larger speed compensates the extra distance, making the 
period smaller and smaller. However, when the particle starts with smaller amplitudes,
the potential landscape gets flatter and flatter, and so its initial speed gets smaller
and smaller. Although distance does decrease, so does its speed and it takes the particle
longer and longer to travel. Very close to the origin, the particle rolling in
the potential landscape has speed very close to zero. It takes it extremely long
to go back and forth. 
'''


