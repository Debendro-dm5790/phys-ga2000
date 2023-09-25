import numpy as np
import matplotlib.pyplot as plt
from random import random

'''
We simulate the radioactive decay of a sample of 1000 Thallium atoms. The 
sample size is stored in a variable called NumOfAtoms and the half life of
Thallium, in seconds, is stored as a single precision floating point number
in a variable called tau.
'''
tau = np.float32(3.053*60)
NumOfAtoms = 1000
decayTimesList = np.float32(np.zeros(NumOfAtoms))

ThalliumPoints = np.float32(np.zeros(NumOfAtoms+1)) + 1000
LeadPoints = np.float32(np.zeros(NumOfAtoms+1))

'''
Using the transformation method, we generate 1000 random numbers from a 
non-uniform distribution. These numbers corresponding to the times a Thallium
nucleus decays. We use numpy.sort() to sort the times in increasing order.
The i-th (i = 1,2,3,...) time value in the sorted list corresponds to the 
time when i nuclei have decayed into Lead. The array named ThalliumPoints 
will contain a list of Thallium nuclei present in the sample over time. 
ThalliumPoints[0] is the number of Thallium nuclei when there have been no
decays. ThalliumPoints[1] is the number of Thallium nuclei after the first
decay, which is 999. Thallium[i] is the number of Thallium nuclei after
the i-th decay and is equal to Thallium[i-1] - 1. As a result, we initialize
ThalliumPoints to contain elements of value 1000 and we adapt the values stored
in it accordingly through a for loop 
'''

for i in range(NumOfAtoms):
    decayTimesList[i] = ((-1*tau)/np.log(2))*np.log(1 - random())
    
decayTimesList = np.sort(decayTimesList)

counter = 1
for time in decayTimesList:
    ThalliumPoints[counter] = ThalliumPoints[counter - 1] - 1
    LeadPoints[counter] = LeadPoints[counter - 1] + 1
    counter += 1
    
decayTimesList = np.append(decayTimesList, [0])
decayTimesList = np.sort(decayTimesList)

'''
We generate a plot showing the population of Lead and Thallium nuclei
in our sample. 
'''
    
plt.plot(decayTimesList, ThalliumPoints, color = 'green')
plt.plot(decayTimesList, LeadPoints, color = 'red')
plt.xlabel('Time in Seconds')
plt.ylabel('Number of Atoms')
plt.title('Radioactive Decay of Thallium')
plt.legend(['Thallium', 'Lead'])
plt.savefig('NonUniformRandomDistThallium.png')
plt.plot()