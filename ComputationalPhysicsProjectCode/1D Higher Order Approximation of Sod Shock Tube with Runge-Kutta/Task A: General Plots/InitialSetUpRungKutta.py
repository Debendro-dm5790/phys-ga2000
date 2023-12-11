import numpy as np
import matplotlib.pyplot as plt

import Initialize as Init 
import Evolution as Evolv
import SaveData as Save
import MakePlots as MP

'''
Initialize all variables and constants related to the simulated system, the spatial grids,
the initial conditions of the problem, and the parameters of the functions defined in the modules
Initialize.py and Evolution.py
'''

length = np.float32(1)
numPoints = np.int32(400)
timePoints = np.int32(1600)
spaceDim = np.int32(1)
timeFrac = np.int32(5)
theta = np.float32(1.5)

gamma = np.float32(1.4)
v0 = np.float32(0)
rhoL = np.float32(100)
rhoR = np.float32(10)
PressL = np.float32(80)
PressR = np.float32(10)

fileList = []
savedTimes = np.array([0,0.08,0.16,0.24,0.32], dtype = np.float32)

'''
Begin the simulation of the 1D Riemann Problem with the higher order Runge-Kutta method
by initializing the system and getting and saving the initial values.
'''
xList, yList, tList = Init.createSpaceTimeGrid(length, numPoints, timePoints, spaceDim)
u1, u2, u3 = Init.setAndGetInitialVals(gamma, numPoints+4, v0, rhoL, rhoR, PressL, PressR)
deltaX = np.float32(xList[1]-xList[0])
currentTime = 0

'''
Save Initial Density Data
'''
filename = 'RungeKuttaDensityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
Save.SaveDataAsCSVFile(xList, u1[2:len(u1)-2], filename, fields)
fileList.append(filename)
print(filename)
            
'''
Save Initial Velocity Data
'''
filename = 'RungeKuttaVelocityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
Save.SaveDataAsCSVFile(xList, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
fileList.append(filename)
print(filename)
            
'''
Save Initial Pressure Data
'''
filename = 'RungeKuttaPressureDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[2:len(u1)-2] - 0.5*(u2[2:len(u1)-2]**2/u1[2:len(u1)-2])), filename, fields)
fileList.append(filename)
print(filename)

'''
Start and continue the evolution of the 1D hydrodynamic system by using the higher order Runge-Kutta method
'''
for t in range(len(tList)):
    timeStep, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1, u2, u3)
    DivF1, DivF2, DivF3 = Evolv.computeFluxDivergenceRungeKutta(deltaX, numPoints, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta)
        
    if currentTime < savedTimes[1] and savedTimes[1] < currentTime + timeStep:
        timeStep = savedTimes[1] - currentTime
        
        u1, u2, u3 = Evolv.doEntireRungeKuttaTimeStep(numPoints, u1, u2, u3, timeStep, DivF1, DivF2, DivF3, deltaX, gamma, theta, timeFrac)  
        currentTime += timeStep
            
        '''
        Save Density Data
        '''
        filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaVelocityDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
        Save.SaveDataAsCSVFile(xList, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
        
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[2:len(u1)-2] - 0.5*(u2[2:len(u1)-2]**2/u1[2:len(u1)-2])), filename, fields)
        fileList.append(filename)
        print(filename)
    elif currentTime < savedTimes[2] and savedTimes[2] < currentTime + timeStep:
        timeStep = savedTimes[2] - currentTime
        
        u1, u2, u3 = Evolv.doEntireRungeKuttaTimeStep(numPoints, u1, u2, u3, timeStep, DivF1, DivF2, DivF3, deltaX, gamma, theta, timeFrac)
        
        currentTime += timeStep
        '''
        Save Density Data
        '''
        filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaVelocityDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
        Save.SaveDataAsCSVFile(xList, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[2:len(u1)-2] - 0.5*(u2[2:len(u1)-2]**2/u1[2:len(u1)-2])), filename, fields)
        fileList.append(filename)
        print(filename)
    elif currentTime < savedTimes[3] and savedTimes[3] < currentTime + timeStep:
        timeStep = savedTimes[3] - currentTime
        
        u1, u2, u3 = Evolv.doEntireRungeKuttaTimeStep(numPoints, u1, u2, u3, timeStep, DivF1, DivF2, DivF3, deltaX, gamma, theta, timeFrac)
        
        currentTime += timeStep
        '''
        Save Density Data
        '''
        filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaVelocityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
        Save.SaveDataAsCSVFile(xList, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[2:len(u1)-2] - 0.5*(u2[2:len(u1)-2]**2/u1[2:len(u1)-2])), filename, fields)
        fileList.append(filename)
        print(filename)
    elif currentTime < savedTimes[4] and savedTimes[4] < currentTime + timeStep:
        timeStep = savedTimes[4] - currentTime
        
        u1, u2, u3 = Evolv.doEntireRungeKuttaTimeStep(numPoints, u1, u2, u3, timeStep, DivF1, DivF2, DivF3, deltaX, gamma, theta, timeFrac)
        
        currentTime += timeStep
            
        '''
        Save Density Data
        '''
        filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaVelocityDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
        Save.SaveDataAsCSVFile(xList, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[2:len(u1)-2] - 0.5*(u2[2:len(u1)-2]**2/u1[2:len(u1)-2])), filename, fields)
        fileList.append(filename)
        print(filename)
    else:
        u1, u2, u3 = Evolv.doEntireRungeKuttaTimeStep(numPoints, u1, u2, u3, timeStep, DivF1, DivF2, DivF3, deltaX, gamma, theta, timeFrac)
        
        currentTime += timeStep
    
x, d1 = MP.ReadAndReturnData(fileList[0])
x, d2 = MP.ReadAndReturnData(fileList[3])
x, d3 = MP.ReadAndReturnData(fileList[6])
x, d4 = MP.ReadAndReturnData(fileList[9])
x, d5 = MP.ReadAndReturnData(fileList[12])
MP.MakeFigure(x, d1, d2, d3, d4, d5, 'Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time for Grid Size = 400 Runge-Kutta', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'RungeKuttaDensitiesGridSize=400.png')

x, v1 = MP.ReadAndReturnData(fileList[1])
x, v2 = MP.ReadAndReturnData(fileList[4])
x, v3 = MP.ReadAndReturnData(fileList[7])
x, v4 = MP.ReadAndReturnData(fileList[10])
x, v5 = MP.ReadAndReturnData(fileList[13])
MP.MakeFigure(x, v1, v2, v3, v4, v5, 'Position (meters)', 'Velocity (meters per second)', 'Velocities Over Time for Grid Size = 400 Runge-Kutta', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'RungeKuttaVelocitiesGridSize=400.png')

x, P1 = MP.ReadAndReturnData(fileList[2])
x, P2 = MP.ReadAndReturnData(fileList[5])
x, P3 = MP.ReadAndReturnData(fileList[8])
x, P4 = MP.ReadAndReturnData(fileList[11])
x, P5 = MP.ReadAndReturnData(fileList[14])
MP.MakeFigure(x, P1, P2, P3, P4, P5, 'Position (meters)', 'Pressure (Pascal)', 'Pressures Over Time for Grid Size = 400 Runge-Kutta', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'RungeKuttaPressuresGridSize=400.png')


