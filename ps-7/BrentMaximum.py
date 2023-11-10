import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import Brent

'''
Find Maximum
'''
a = -3
b = -2
c = -1
bOriginal = 1000
tol = 1.e-5


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
  status, b = Brent.quadraticBracketingStep(a,b,c,prevPrevStep[stepNum - 2])
  if b < bOriginal:
      c = bOriginal
  else:
      a = bOriginal
  prevPrevStep.append(b)
  stepNum += 1
   
if status == 1:
    myBrentMax = b
else:
    print('Using parabolic steps is not appropriate. Going to perform golden section search.')
    while np.abs(c - a) > tol:
        print('Using Golden Section search')
        a,b,c = Brent.golden(a,b,c, 'Max')
        
    myBrentMax = b
    
print('Maximum based on my implementation of Brent is at ' + str(myBrentMax) + ' and function value is ' + str(Brent.func(myBrentMax)))
   
def func2(x):
    return -1*((x-0.3)**2)*np.exp(x)

xmin, fval, iter, funcalls = optimize.brent(func2, brack = np.array([-3,-2,-0.5]), full_output=True)

print('Maximum based on Python implementation of Brent is at ' + str(xmin) + ' and function value is ' + str(-fval))

