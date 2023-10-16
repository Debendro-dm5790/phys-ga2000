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
We first implement a function that returns the original integrand of the gamma function integral.
We make three plots on a single graph for the cases where the a in the integrand is 2,3, and 4 
for 0 < x < 5.
'''

def integrand(a,x):
    return x**(a-1)*np.exp(-1*x)

a = np.array([2,3,4], dtype = np.int32)
x = np.linspace(0,5,100)
integrandList = []

for i in range(len(a)):
    integrandList.append(integrand(a[i], x))
    
plt.plot(x, integrandList[0])
plt.plot(x, integrandList[1])
plt.plot(x, integrandList[2])
plt.xlabel('x')
plt.ylabel('Gamma function integrand value')
plt.title('Gamma function Integrand Plot')
plt.legend(['a  = 2', 'a = 3', 'a = 4'])
plt.savefig('GammaIntegrand.png')
plt.show()

'''
We note that that in the original integrand, as x grows larger, x^{a - 1} becomes very large and 
e^{-x} becomes very small. This can cause numerical overflow or underflow, making the computation
difficult. In order to overcome this difficulty, we rewrite x^{a - 1} as e^{(a - 1)ln x} and so 
the integrand becomes e^{(a - 1)ln x - x}. We note that both terms in the exponent grow as x grows
, with x growing faster than ln x. We note that in the exponent, we add two numbers with dissimilar
magnitudes and opposite signs. This does not need to numerical errors; when two numbers with similar
magnitudes and opposite signs are added, errors blow up. 
 
After rewriting the orginal integrand in terms of the natural logarithm, we rescale the limits 
of integration to [0, 1] by using the substitution z = x/(c+x). We want to extremum location 
of the orginal integrand, which is at x = a - 1, to be mapped to the new location z = 1/2. 
This allows us to determine c to be a - 1.
'''
def modifiedIntegrand(a,z):
    return np.exp(((a - 1)*np.log((z*(a - 1))/(1 - z))) - ((z*(a - 1))/(1 - z)))*((a - 1)/(1 - z)**2)

'''
We finally use Gauss-Legendre quadrature to compute this gamma function integral for a = 3/2, 3, 6 and 10.
'''

def gamma(a):
    N = 20
    xp, wp = gaussxwab(N,0,1)
    s = sum(modifiedIntegrand(a, xp)*wp)
    return s

print('Gamma(3/2) is ' + str(gamma(np.float32(1.5))))
print('Gamma(3) is ' + str(gamma(np.int32(3))))
print('Gamma(6) is ' + str(gamma(np.int32(6))))
print('Gamma(10) is ' + str(gamma(np.int32(10))))