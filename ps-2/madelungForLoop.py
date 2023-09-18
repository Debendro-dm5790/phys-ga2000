# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:14:55 2023

@author: mooke
"""

import matplotlib.pyplot as plt
import numpy as np
import timeit

def madelungForLoop(L):
    print('Now starting L = ' + str(L))
    length = np.int32(L)
    lenList = np.linspace(start = -1*length, stop = length, num = 2*length + 1, dtype = np.int32)

    M1 = 0

    start_time = timeit.default_timer()
    for i in lenList:
        for j in lenList:
            for k in lenList:
                if i !=0 and j != 0 and k != 0:
                    root = np.float32(np.sqrt(i**2 + j**2 + k**2))
                    sum = i + j + k
                    if sum%2 == 0:
                        M1 += (root)**(-1)
                    else:
                        M1 += -1*(root)**(-1)
    timeElapsed = timeit.default_timer() - start_time

    print('Time to run for loop is ' + str(timeElapsed) + ' seconds.')
    print('Madelung Constant for L = ' + str(L) + ' is ')
    print(M1)
    print('')
    
    return M1

LArray = np.array([10,20,30,40,50,60,70,80,90,100,150,200,250,300])
MList = np.array([])

for L in LArray:
    M = madelungForLoop(L)
    MList = np.append(MList, M)
    
plt.plot(LArray, MList, 'o')
plt.xlabel('L, the Number of Atoms in All Directions')
plt.ylabel('Estimated Value of Madelung Constant')
plt.title('Approximating the Madelung Constant with a For Loop')
plt.savefig('MadelungForLoops.png')
plt.show()
