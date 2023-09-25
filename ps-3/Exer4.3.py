# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

'''
We first implement a python function that approximates the derivative 
of the function f(x) = x(x-1) at x = 1. The function takes two parameters,
value, which is the x value at which the derivative is approximated, and 
delta, which is the small deviation from the input x value. The function
returns an approximate value of the derivative at the input x value as a 
single precision floating point number. We note that this code inherently 
has approximation error. 
'''

def derivativeApprox(value, delta):
    return np.float32(((value + delta)*(value + delta - 1) - (value*(value - 1)))/delta)

'''
We implement a variable called deltaVal and an numpy array called deltaList
. The first variable stores the deviation used in the first part of the 
problem as a single precision floating point number. This variable has a 
value of 0.01. The array stores the deviations used in the second part of 
the problem as single precision floating point numbers. These deviations 
are 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, and 1e-14
'''
deltaVal = np.float32(1e-2)
deltaList = np.float32( np.array([1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]))

'''
PART A:
The actual value of the derivative is 1, but the computer gives 1.01. 
There are three main sources of this error:
    (1) Addition of a large number (ex: 1) and a small number (ex: 0.01) will 
    lead to loss of precision. This will become more prominent as delta values
    become smaller and smaller. 
    (2) The addition of two numbers with similar magnitudes but differing signs
    will amplify numerical errors. 
    (3) Approximation error due to the approximate derivative formula
'''

print('The approximate derivative of f(x) at x = 1 is ' + str(derivativeApprox(value = np.int32(1), delta = deltaVal)))

'''
PART B:
We implement a for loop to determinate the computer's estimate of the 
derivative for the various deviations found in the array deltaList.    
Since delta is getting smaller, we should expect some improvement in the 
derivative calculations; however, as delta gets smaller precision should 
decrease due to reasons (1) and (2).
'''

for i in range(len(deltaList)):
    derivativeEstimate = derivativeApprox(np.int32(1), deltaList[i])
    print('When delta is ' + str(deltaList[i]) + ', the approximated derivative at x = 1 is ' + str(derivativeEstimate))
    

