import numpy as np
import matplotlib.pyplot as plt

import Initialize as Init 
import Evolution as Evolv
import SaveData as Save
import MakePlots as MP

import timeit

'''
Initialize all variables and constants related to the simulated system, the spatial grids,
the initial conditions of the problem, and the parameters of the functions defined in the modules
Initialize.py and Evolution.py. These will be common for both the lower order 1D and the higher order 
1D Sod Shock Tube problems.
'''

length = np.float32(1)
numPointsArray = np.array([100,200,400], dtype = np.int32)
timePoints = np.array([400,800,1600], dtype = np.int32)
spaceDim = np.int32(1)
timeFrac = np.int32(5)

gamma = np.float32(1.4)
v0 = np.float32(0)
rhoL = np.float32(100)
rhoR = np.float32(10)
PressL = np.float32(80)
PressR = np.float32(10)

savedTimes = np.array([0.08,0.16,0.24,0.32], dtype = np.float32)

'''
Initialize two numpy arrays that store the times for the lower order and higher order methods, 
respectively. 
'''
timesForLowerOrder = np.zeros(len(savedTimes)*len(numPointsArray), dtype = np.float32)
timesForHigherOrder = timesForLowerOrder.copy()

'''
Start the lower order method. 

Create the for loop that will go through the various sizes of the spatial grid. 
There will be three cases: 
    
Case 1: Number of spatial grid points is numPointsArray[0] = 100
Case 2: Number of spatial grid points is numPointsArray[1] = 200
Case 3: Number of spatial grid points is numPointsArray[2] = 400
'''

timeCounter = 0
timeIndex = 0

for numPoints in numPointsArray:
    xList, yList, tList = Init.createSpaceTimeGrid(length, numPoints, timePoints[timeCounter], spaceDim)
    u1, u2, u3 = Init.setAndGetInitialVals(gamma, numPoints+2, v0, rhoL, rhoR, PressL, PressR)
    deltaX = np.float32(xList[1]-xList[0])
    currentTime = 0
    timeStep = 0
    
    for t in range(len(tList)):
        if currentTime < savedTimes[0] and savedTimes[0] < currentTime + timeStep:
            start_time = timeit.default_timer()
            
            timeStep, F1_HLL, F2_HLL, F3_HLL = Evolv.computeTimeStepandFHLL(deltaX, gamma, numPoints+1, timeFrac, u1, u2, u3)
            DivF1, DivF2, DivF3 = Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL, F2_HLL, F3_HLL)
            timeStep = savedTimes[0] - currentTime
            u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
            
            timeElapsed = timeit.default_timer() - start_time
            
            timesForLowerOrder[timeIndex] = timeElapsed 
            
            timeIndex += 1
            
            currentTime += timeStep
        elif currentTime < savedTimes[1] and savedTimes[1] < currentTime + timeStep:
            start_time = timeit.default_timer()
            
            timeStep, F1_HLL, F2_HLL, F3_HLL = Evolv.computeTimeStepandFHLL(deltaX, gamma, numPoints+1, timeFrac, u1, u2, u3)
            DivF1, DivF2, DivF3 = Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL, F2_HLL, F3_HLL)
            timeStep = savedTimes[1] - currentTime
            u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
            
            timeElapsed = timeit.default_timer() - start_time
            
            timesForLowerOrder[timeIndex] = timeElapsed 
            
            timeIndex += 1
            
            currentTime += timeStep
        elif currentTime < savedTimes[2] and savedTimes[2] < currentTime + timeStep:
            start_time = timeit.default_timer()
            
            timeStep, F1_HLL, F2_HLL, F3_HLL = Evolv.computeTimeStepandFHLL(deltaX, gamma, numPoints+1, timeFrac, u1, u2, u3)
            DivF1, DivF2, DivF3 = Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL, F2_HLL, F3_HLL)
            timeStep = savedTimes[2] - currentTime
            u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
            
            timeElapsed = timeit.default_timer() - start_time
            
            timesForLowerOrder[timeIndex] = timeElapsed 
            
            timeIndex += 1
            
            currentTime += timeStep
        elif currentTime < savedTimes[3] and savedTimes[3] < currentTime + timeStep:
            start_time = timeit.default_timer()
            
            timeStep, F1_HLL, F2_HLL, F3_HLL = Evolv.computeTimeStepandFHLL(deltaX, gamma, numPoints+1, timeFrac, u1, u2, u3)
            DivF1, DivF2, DivF3 = Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL, F2_HLL, F3_HLL)
            timeStep = savedTimes[3] - currentTime
            u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
            
            timeElapsed = timeit.default_timer() - start_time
            
            timesForLowerOrder[timeIndex] = timeElapsed 
            
            timeIndex += 1
            
            currentTime += timeStep
        else: 
            timeStep, F1_HLL, F2_HLL, F3_HLL = Evolv.computeTimeStepandFHLL(deltaX, gamma, numPoints+1, timeFrac, u1, u2, u3)
            DivF1, DivF2, DivF3 = Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL, F2_HLL, F3_HLL)
            u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
            currentTime += timeStep
            
    timeCounter += 1
    
'''
Start the higher order Runge-Kutta method. 

Create the for loop that will go through the various sizes of the spatial grid. 
There will be three cases: 
    
Case 1: Number of spatial grid points is numPointsArray[0] = 100
Case 2: Number of spatial grid points is numPointsArray[1] = 200
Case 3: Number of spatial grid points is numPointsArray[2] = 400
'''
theta = np.float32(1.5)

timeCounter = 0
timeIndex = 0

for numPoints in numPointsArray:
    print(numPoints)
    xList, yList, tList = Init.createSpaceTimeGrid(length, numPoints, timePoints[timeCounter], spaceDim)
    u1, u2, u3 = Init.setAndGetInitialVals(gamma, numPoints+4, v0, rhoL, rhoR, PressL, PressR)
    deltaX = np.float32(xList[1]-xList[0])
    currentTime = 0
    timeStep = 0
    
    for t in range(len(tList)):
        if currentTime < savedTimes[0] and savedTimes[0] < currentTime + timeStep:
            start_time = timeit.default_timer()
            
            timeStep, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1, u2, u3)
            DivF1, DivF2, DivF3 = Evolv.computeFluxDivergenceRungeKutta(deltaX, numPoints, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta)
            timeStep = savedTimes[0] - currentTime
            u1, u2, u3 = Evolv.updateQuantitiesRungeKutta(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
            
            timeElapsed = timeit.default_timer() - start_time
            
            timesForHigherOrder[timeIndex] = timeElapsed 
            
            timeIndex += 1
            
            currentTime += timeStep
        elif currentTime < savedTimes[1] and savedTimes[1] < currentTime + timeStep:
            start_time = timeit.default_timer()
            
            timeStep, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1, u2, u3)
            DivF1, DivF2, DivF3 = Evolv.computeFluxDivergenceRungeKutta(deltaX, numPoints, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta)
            timeStep = savedTimes[1] - currentTime
            u1, u2, u3 = Evolv.updateQuantitiesRungeKutta(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
            
            timeElapsed = timeit.default_timer() - start_time
            
            timesForHigherOrder[timeIndex] = timeElapsed 
            
            timeIndex += 1
            
            currentTime += timeStep
        elif currentTime < savedTimes[2] and savedTimes[2] < currentTime + timeStep:
            start_time = timeit.default_timer()
            
            timeStep, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1, u2, u3)
            DivF1, DivF2, DivF3 = Evolv.computeFluxDivergenceRungeKutta(deltaX, numPoints, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta)
            timeStep = savedTimes[2] - currentTime
            u1, u2, u3 = Evolv.updateQuantitiesRungeKutta(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
            
            timeElapsed = timeit.default_timer() - start_time
            
            timesForHigherOrder[timeIndex] = timeElapsed 
            
            timeIndex += 1
            
            currentTime += timeStep
        elif currentTime < savedTimes[3] and savedTimes[3] < currentTime + timeStep:
            start_time = timeit.default_timer()
            
            timeStep, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1, u2, u3)
            DivF1, DivF2, DivF3 = Evolv.computeFluxDivergenceRungeKutta(deltaX, numPoints, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta)
            timeStep = savedTimes[3] - currentTime
            u1, u2, u3 = Evolv.updateQuantitiesRungeKutta(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
            
            timeElapsed = timeit.default_timer() - start_time
            
            timesForHigherOrder[timeIndex] = timeElapsed 
            
            timeIndex += 1
            
            currentTime += timeStep
        else: 
            timeStep, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1, u2, u3)
            DivF1, DivF2, DivF3 = Evolv.computeFluxDivergenceRungeKutta(deltaX, numPoints, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta)
            u1, u2, u3 = Evolv.updateQuantitiesRungeKutta(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
            currentTime += timeStep
            
    timeCounter += 1
    
numOfSavedTimes = len(savedTimes)
MP.MakeFigure3(savedTimes, timesForLowerOrder[0:numOfSavedTimes], timesForHigherOrder[0:numOfSavedTimes], 'Time Counts', 'Execution Time (seconds)', 'Comparing Execution Times for Lower Order and Runge-Kutta Methods For System Size = 100', ['Lower Order', 'Runge-Kutta'], 'TimeCompareSize=100.png')
MP.MakeFigure3(savedTimes, timesForLowerOrder[numOfSavedTimes:2*numOfSavedTimes], timesForHigherOrder[numOfSavedTimes:2*numOfSavedTimes], 'Time Counts', 'Execution Time (seconds)', 'Comparing Execution Times for Lower Order and Runge-Kutta Methods For System Size = 200', ['Lower Order', 'Runge-Kutta'], 'TimeCompareSize=200.png')
MP.MakeFigure3(savedTimes, timesForLowerOrder[2*numOfSavedTimes:3*numOfSavedTimes], timesForHigherOrder[2*numOfSavedTimes:3*numOfSavedTimes], 'Time Counts', 'Execution Time (seconds)', 'Comparing Execution Times for Lower Order and Runge-Kutta Methods For System Size = 400', ['Lower Order', 'Runge-Kutta'], 'TimeCompareSize=400.png')
