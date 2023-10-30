import numpy as np
import matplotlib.pyplot as plt

import Initialize as Init 
import Evolution as Evolv
import SaveData as Save
import MakePlots as MP

'''
Initialize all variables and constants related to the simulated system, the spatial grids,
the initial conditions of the problem, and the parameters of the functions defined in the modules
Initialize.py and Evolution.py. Since space is being inverted, the variable length will be set
to a negative number. 
'''

length = np.float32(-1)
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

fileList = []
savedTimes = []

'''
Create the for loop that will go through the various sizes of the spatial grid. 
There will be three cases: 
    
Case 1: Number of spatial grid points is numPointsArray[0] = 100
Case 2: Number of spatial grid points is numPointsArray[1] = 200
Case 3: Number of spatial grid points is numPointsArray[2] = 400
'''
timeCounter = 0

for numPoints in numPointsArray:
    xList, yList, tList = Init.createSpaceTimeGrid(length, numPoints, timePoints[timeCounter], spaceDim)
    u1, u2, u3 = Init.setAndGetInitialVals(gamma, numPoints+2, v0, rhoL, rhoR, PressL, PressR)
    deltaX = np.float32(xList[1]-xList[0])
    currentTime = 0
    
    savedTimes.append(currentTime)
    '''
    Save Initial Density Data
    '''
    filename = 'SpaceInvertDensityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
    fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
    Save.SaveDataAsCSVFile(xList, u1[1:len(u1)-1], filename, fields)
    fileList.append(filename)
    print(filename)
            
    '''
    Save Initial Velocity Data
    '''
    filename = 'SpaceInvertVelocityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
    fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
    Save.SaveDataAsCSVFile(xList, u2[1:len(u1)-1]/u1[1:len(u1)-1], filename, fields)
    fileList.append(filename)
    print(filename)
            
    '''
    Save Initial Pressure Data
    '''
    filename = 'SpaceInvertPressureDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
    fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
    Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[1:len(u1)-1] - 0.5*(u2[1:len(u1)-1]**2/u1[1:len(u1)-1])), filename, fields)
    fileList.append(filename)
    print(filename)

    for t in range(len(tList)):
        timeStep, F1_HLL, F2_HLL, F3_HLL = Evolv.computeTimeStepandFHLL(deltaX, gamma, numPoints+1, timeFrac, u1, u2, u3)
        DivF1, DivF2, DivF3 = Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL, F2_HLL, F3_HLL)
        u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
        currentTime += timeStep
        
        if t == np.int32(len(tList)//4) or t == 2 * np.int32(len(tList)//4) or t == 3 * np.int32(len(tList)//4) or t == len(tList) - 1:
            savedTimes.append(currentTime)
            '''
            Save Density Data
            '''
            filename = 'SpaceInvertDensityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
            fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
            Save.SaveDataAsCSVFile(xList, u1[1:len(u1)-1], filename, fields)
            fileList.append(filename)
            print(filename)
            
            '''
            Save Velocity Data
            '''
            filename = 'SpaceInvertVelocityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
            fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
            Save.SaveDataAsCSVFile(xList, u2[1:len(u1)-1]/u1[1:len(u1)-1], filename, fields)
            fileList.append(filename)
            print(filename)
            
            '''
            Save Pressure Data
            '''
            filename = 'SpaceInvertPressureDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
            fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
            Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[1:len(u1)-1] - 0.5*(u2[1:len(u1)-1]**2/u1[1:len(u1)-1])), filename, fields)
            fileList.append(filename)
            print(filename)
            
    timeCounter += 1
            
x, d1 = MP.ReadAndReturnData(fileList[0])
x, d2 = MP.ReadAndReturnData(fileList[3])
x, d3 = MP.ReadAndReturnData(fileList[6])
x, d4 = MP.ReadAndReturnData(fileList[9])
x, d5 = MP.ReadAndReturnData(fileList[12])
MP.MakeFigure(x, d1, d2, d3, d4, d5, 'Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time for Grid Size = 100', [str(np.abs(savedTimes[0])) + ' sec', str(np.abs(savedTimes[1])) + ' sec', str(np.abs(savedTimes[2])) + ' sec', str(np.abs(savedTimes[3])) + ' sec', str(np.abs(savedTimes[4])) + ' sec'], 'SpaceInvertDensitiesGridSize=100.png')

x, v1 = MP.ReadAndReturnData(fileList[1])
x, v2 = MP.ReadAndReturnData(fileList[4])
x, v3 = MP.ReadAndReturnData(fileList[7])
x, v4 = MP.ReadAndReturnData(fileList[10])
x, v5 = MP.ReadAndReturnData(fileList[13])
MP.MakeFigure(x, v1, v2, v3, v4, v5, 'Position (meters)', 'Velocity (meters per second)', 'Velocities Over Time for Grid Size = 100', [str(np.abs(savedTimes[0])) + ' sec', str(np.abs(savedTimes[1])) + ' sec', str(np.abs(savedTimes[2])) + ' sec', str(np.abs(savedTimes[3])) + ' sec', str(np.abs(savedTimes[4])) + ' sec'], 'SpaceInvertVelocitiesGridSize=100.png')

x, P1 = MP.ReadAndReturnData(fileList[2])
x, P2 = MP.ReadAndReturnData(fileList[5])
x, P3 = MP.ReadAndReturnData(fileList[8])
x, P4 = MP.ReadAndReturnData(fileList[11])
x, P5 = MP.ReadAndReturnData(fileList[14])
MP.MakeFigure(x, P1, P2, P3, P4, P5, 'Position (meters)', 'Pressure (Pascal)', 'Pressures Over Time for Grid Size = 100', [str(np.abs(savedTimes[0])) + ' sec', str(np.abs(savedTimes[1])) + ' sec', str(np.abs(savedTimes[2])) + ' sec', str(np.abs(savedTimes[3])) + ' sec', str(np.abs(savedTimes[4])) + ' sec'], 'SpaceInvertPressuresGridSize=100.png')


x, d1 = MP.ReadAndReturnData(fileList[15])
x, d2 = MP.ReadAndReturnData(fileList[18])
x, d3 = MP.ReadAndReturnData(fileList[21])
x, d4 = MP.ReadAndReturnData(fileList[24])
x, d5 = MP.ReadAndReturnData(fileList[27])
MP.MakeFigure(x, d1, d2, d3, d4, d5, 'Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time for Grid Size = 200', [str(np.abs(savedTimes[5])) + ' sec', str(np.abs(savedTimes[6]))+ ' sec', str(np.abs(savedTimes[7]))+ ' sec', str(np.abs(savedTimes[8]))+ ' sec', str(np.abs(savedTimes[9]))+ ' sec'], 'SpaceInvertDensitiesGridSize=200.png')

x, v1 = MP.ReadAndReturnData(fileList[16])
x, v2 = MP.ReadAndReturnData(fileList[19])
x, v3 = MP.ReadAndReturnData(fileList[22])
x, v4 = MP.ReadAndReturnData(fileList[25])
x, v5 = MP.ReadAndReturnData(fileList[28])
MP.MakeFigure(x, v1, v2, v3, v4, v5, 'Position (meters)', 'Velocity (meters per second)', 'Velocities Over Time for Grid Size = 200', [str(np.abs(savedTimes[5])) + ' sec', str(np.abs(savedTimes[6]))+ ' sec', str(np.abs(savedTimes[7]))+ ' sec', str(np.abs(savedTimes[8]))+ ' sec', str(np.abs(savedTimes[9]))+ ' sec'], 'SpaceInvertVelocitiesGridSize=200.png')

x, P1 = MP.ReadAndReturnData(fileList[17])
x, P2 = MP.ReadAndReturnData(fileList[20])
x, P3 = MP.ReadAndReturnData(fileList[23])
x, P4 = MP.ReadAndReturnData(fileList[26])
x, P5 = MP.ReadAndReturnData(fileList[29])
MP.MakeFigure(x, P1, P2, P3, P4, P5, 'Position (meters)', 'Pressure (Pascal)', 'Pressures Over Time for Grid Size = 200', [str(np.abs(savedTimes[5])) + ' sec', str(np.abs(savedTimes[6])) + ' sec', str(np.abs(savedTimes[7]))+ ' sec', str(np.abs(savedTimes[8]))+ ' sec', str(np.abs(savedTimes[9]))+ ' sec'], 'SpaceInvertPressuresGridSize=200.png')

#
x, d1 = MP.ReadAndReturnData(fileList[30])
x, d2 = MP.ReadAndReturnData(fileList[33])
x, d3 = MP.ReadAndReturnData(fileList[36])
x, d4 = MP.ReadAndReturnData(fileList[39])
x, d5 = MP.ReadAndReturnData(fileList[42])
MP.MakeFigure(x, d1, d2, d3, d4, d5, 'Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time for Grid Size = 400', [str(np.abs(savedTimes[10])) + ' sec', str(np.abs(savedTimes[11])) + ' sec', str(np.abs(savedTimes[12]))+ ' sec', str(np.abs(savedTimes[13]))+ ' sec', str(np.abs(savedTimes[14]))+ ' sec'], 'SpaceInvertDensitiesGridSize=400.png')

x, v1 = MP.ReadAndReturnData(fileList[31])
x, v2 = MP.ReadAndReturnData(fileList[34])
x, v3 = MP.ReadAndReturnData(fileList[37])
x, v4 = MP.ReadAndReturnData(fileList[40])
x, v5 = MP.ReadAndReturnData(fileList[43])
MP.MakeFigure(x, v1, v2, v3, v4, v5, 'Position (meters)', 'Velocity (meters per second)', 'Velocities Over Time for Grid Size = 400', [str(np.abs(savedTimes[10])) + ' sec', str(np.abs(savedTimes[11]))+ ' sec', str(np.abs(savedTimes[12]))+ ' sec', str(np.abs(savedTimes[13]))+ ' sec', str(np.abs(savedTimes[14]))+ ' sec'], 'SpaceInvertVelocitiesGridSize=400.png')

x, P1 = MP.ReadAndReturnData(fileList[32])
x, P2 = MP.ReadAndReturnData(fileList[35])
x, P3 = MP.ReadAndReturnData(fileList[38])
x, P4 = MP.ReadAndReturnData(fileList[41])
x, P5 = MP.ReadAndReturnData(fileList[44])
MP.MakeFigure(x, P1, P2, P3, P4, P5, 'Position (meters)', 'Pressure (Pascal)', 'Pressures Over Time for Grid Size = 400', [str(np.abs(savedTimes[10])) + ' sec', str(np.abs(savedTimes[11]))+ ' sec', str(np.abs(savedTimes[12]))+ ' sec', str(np.abs(savedTimes[13]))+ ' sec', str(np.abs(savedTimes[14]))+ ' sec'], 'SpaceInvertPressuresGridSize=400.png')


