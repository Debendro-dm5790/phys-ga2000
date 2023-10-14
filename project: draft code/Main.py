# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import Initialize as Init 
import Evolution as Evolv

length = np.float32(1)
numPoints = np.int32(700)
spaceDim = np.int32(1)

gamma = np.float32(1.4)
v0 = np.float32(0)
rhoL = np.float32(100)
rhoR = np.float32(10)
PressL = np.float32(80)
PressR = np.float32(10)

xList, yList, tList = Init.createSpaceTimeGrid(length, numPoints, spaceDim)

u1, u2, u3 = Init.setAndGetInitialVals(gamma, numPoints+2, v0, rhoL, rhoR, PressL, PressR)

deltaX = np.float32(xList[1]-xList[0])
currentTime = 0

for t in range(len(tList)):
    print('Time step is' + str(t+1))
    
    timeStep, F1_HLL, F2_HLL, F3_HLL = Evolv.computeTimeStepandFHLL(deltaX, gamma, numPoints+1, u1, u2, u3)
    DivF1, DivF2, DivF3 = Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL, F2_HLL, F3_HLL)
    u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
    
    #plt.plot(xList, (gamma - 1)*(u3[1:len(u1)-1] - 0.5*(u2[1:len(u1)-1]**2/u1[1:len(u1)-1])), '*')
    #plt.xlabel('Position')
    #plt.ylabel('Pressure')
    #plt.title('Pressure at timestep ' + str(t))
    #plt.savefig('PressureFig'+str(t+1)+'.png')
    #plt.show()
        
    plt.plot(xList, u1[1:len(u1)-1])
    plt.title('Mass Density')
    plt.show()
        
        #plt.plot(xList, u3[1:len(u1)-1], '*')
        #plt.title('Energy Density')
        #plt.show()
    
    currentTime += timeStep