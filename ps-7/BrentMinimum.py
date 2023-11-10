import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import Brent


myBrentMin = 0
myBrentMax = 0

status = 1
tol = 1.e-5

'''
Find Minimum
'''
a = 0
b = 0.75
bOriginal = 1
c = 1
tol = 1.e-6

'''
Step 0
'''
status, x = Brent.quadraticBracketingStep(a,b,c,100000)
prevPrevStep = []
prevPrevStep.append(x)

'''
Step 1
'''
bOriginal = b 
status, b = Brent.quadraticBracketingStep(a,b,c,100000)
prevPrevStep.append(b)

if b < bOriginal:
    c = bOriginal
else:
    a = bOriginal
    
'''
Now beginning step 2
'''
stepNum = 2

while status == 1 and np.abs(bOriginal - b) > tol:
  print('Still Using Parabolic Approximtations')
  bOriginal = b 
  status, b = Brent.quadraticBracketingStep(a,b,c,prevPrevStep[stepNum-2])
  if b < bOriginal:
      c = bOriginal
  else:
      a = bOriginal
  prevPrevStep.append(b)
  stepNum += 1
   
   
if status == 1:
    myBrentMin = b
else:
    print('Using parabolic steps is not appropriate. Going to perform golden section search.')
    while np.abs(c - a) > tol:
        print('Using Golden Section search')
        a,b,c = Brent.golden(a,b,c, 'Min')
        
    myBrentMin = b
    
print('Minimum based on my implementation of Brent is at ' + str(myBrentMin) + ' and function value is ' + str(Brent.func(myBrentMin)))
   
xmin, fval, iter, funcalls = optimize.brent(Brent.func, brack = (-1,0.75,1), full_output=True)

print('Minimum based on Python implementation of Brent is at ' + str(xmin) + ' and function value is ' + str(fval))

    

              
            
