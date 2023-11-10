import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def func(x):
    return ((x-0.3)**2)*np.exp(x)

xList = np.linspace(-3,1,500)

plt.plot(xList, func(xList))
plt.xlabel('x values')
plt.ylabel('f(x) values')
plt.title('Plot of the Function')
plt.savefig('FunctionPlot.png')
plt.show()

def quadraticBracketingStep(x_lower, x_middle, x_upper, prevPrevStep):
    a = x_lower
    b = x_middle
    c = x_upper
    
    fa = func(a)
    fb = func(b)
    fc = func(c)
    
    numerator = (((b - a)**2)*(fb - fc)) - (((b - c)**2)*(fb - fa)) 
    denominator = ((b - a)*(fb - fc)) - ((b - c)*(fb - fa))
    
    minModel = b - 0.5*(numerator/denominator)
    
    status = 1
    
    if minModel < x_lower or minModel > x_upper or minModel > prevPrevStep:
        status = 0  #failure. Revert to golden section search
    
    return status, minModel
    
        
def golden(a,b,c, optType):
    goldRatio = (3 - np.sqrt(5))/2
    x = 0
    
    if optType == 'Min':
        if ((b - a) > (c - b)):
            x = b
            b = b - goldRatio*(b - a)
        else:
            x = b + goldRatio*(c - b)
        fb = func(b)
        fx = func(x)
        if fb < fx:
            c = x
        else:
            a = b
            b = x
            
        return a,b,c
    elif optType == 'Max':
        if ((b - a) > (c - b)):
            x = b
            b = b - goldRatio*(b - a)
        else:
            x = b + goldRatio*(c - b)
        fb = func(b)
        fx = func(x)
        if fb < fx:
            a = b
            b = x
        else:
            c = x
            
        return a,b,c
    

        