import numpy as np
import matplotlib.pyplot as plt

import Initialize as Init 
import Evolution as Evolv
import SaveData as Save
import MakePlots as MP

length = np.float32(4)
numPoints = np.int32(700)
timePoints = np.int32(1600)
spaceDim = np.int32(1)
timeFrac = np.int32(5)

gamma = np.float32(1.4)
v0 = np.float32(0)
rhoL = np.float32(10)
rhoR = np.float32(1)
PressL = np.float32(100)
PressR = np.float32(1)

fileList = []
savedTimes = []

xList, yList, tList = Init.createSpaceTimeGrid(length, numPoints, timePoints, spaceDim)
u1, u2, u3 = Init.setAndGetInitialVals(gamma, numPoints+2, v0, rhoL, rhoR, PressL, PressR)
deltaX = np.float32(xList[1]-xList[0])
currentTime = 0
    
savedTimes.append(currentTime)
'''
Save Initial Density Data
'''
filename = 'DensityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
Save.SaveDataAsCSVFile(xList, u1[1:len(u1)-1], filename, fields)
fileList.append(filename)
print(filename)
            
'''
Save Initial Velocity Data
'''
filename = 'VelocityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
Save.SaveDataAsCSVFile(xList, u2[1:len(u1)-1]/u1[1:len(u1)-1], filename, fields)
fileList.append(filename)
print(filename)
            
'''
Save Initial Pressure Data
'''
filename = 'PressureDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
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
        filename = 'DensityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, u1[1:len(u1)-1], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'VelocityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
        Save.SaveDataAsCSVFile(xList, u2[1:len(u1)-1]/u1[1:len(u1)-1], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'PressureDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[1:len(u1)-1] - 0.5*(u2[1:len(u1)-1]**2/u1[1:len(u1)-1])), filename, fields)
        fileList.append(filename)
        print(filename)
            
            
x, d1 = MP.ReadAndReturnData(fileList[0])
x, d2 = MP.ReadAndReturnData(fileList[3])
x, d3 = MP.ReadAndReturnData(fileList[6])
x, d4 = MP.ReadAndReturnData(fileList[9])
x, d5 = MP.ReadAndReturnData(fileList[12])
MP.MakeFigure(x, d1, d2, d3, d4, d5, 'Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time for Grid Size = 700', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'DensitiesGridSize=700.png')

x, v1 = MP.ReadAndReturnData(fileList[1])
x, v2 = MP.ReadAndReturnData(fileList[4])
x, v3 = MP.ReadAndReturnData(fileList[7])
x, v4 = MP.ReadAndReturnData(fileList[10])
x, v5 = MP.ReadAndReturnData(fileList[13])
MP.MakeFigure(x, v1, v2, v3, v4, v5, 'Position (meters)', 'Velocity (meters per second)', 'Velocities Over Time for Grid Size = 700', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'VelocitiesGridSize=700.png')

x, P1 = MP.ReadAndReturnData(fileList[2])
x, P2 = MP.ReadAndReturnData(fileList[5])
x, P3 = MP.ReadAndReturnData(fileList[8])
x, P4 = MP.ReadAndReturnData(fileList[11])
x, P5 = MP.ReadAndReturnData(fileList[14])
MP.MakeFigure(x, P1, P2, P3, P4, P5, 'Position (meters)', 'Pressure (Pascal)', 'Pressures Over Time for Grid Size = 700', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'PressuresGridSize=700.png')


