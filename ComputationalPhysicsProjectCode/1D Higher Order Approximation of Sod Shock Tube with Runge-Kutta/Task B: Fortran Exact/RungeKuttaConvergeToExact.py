import numpy as np
import matplotlib.pyplot as plt

import Initialize as Init 
import Evolution as Evolv
import SaveData as Save
import MakePlots as MP

'''
Retrieve and store the exact data found in the Fortran .data files
'''
time1, xList, rho1, press1, vel1 = MP.ReadAndReturnFortranData('output_001.dat')
time2, xList2, rho2, press2, vel2 = MP.ReadAndReturnFortranData('output_004.dat')
time3, xList3, rho3, press3, vel3 = MP.ReadAndReturnFortranData('output_007.dat')
time4, xList4, rho4, press4, vel4 = MP.ReadAndReturnFortranData('output_010.dat')
time5, xList5, rho5, press5, vel5 = MP.ReadAndReturnFortranData('output_014.dat')

xList = xList[0:400]

rho1 = rho1[0:400]
rho2 = rho2[0:400]
rho3 = rho3[0:400]
rho4 = rho4[0:400]
rho5 = rho5[0:400]

press1 = press1[0:400]
press2 = press2[0:400]
press3 = press3[0:400]
press4 = press4[0:400]
press5 = press5[0:400]

vel1 = vel1[0:400]
vel2 = vel2[0:400]
vel3 = vel3[0:400]
vel4 = vel4[0:400]
vel5 = vel5[0:400]

'''
Begin the generation of approximate solutions with Python
'''

'''
Initialize all variables and constants related to the simulated system, the spatial grids,
the initial conditions of the problem, and the parameters of the functions defined in the modules
Initialize.py and Evolution.py
'''

length = np.float32(4)
numPointsArray = np.array([100,200], dtype = np.int32)
timePoints = np.array([1600,1600], dtype = np.int32)
spaceDim = np.int32(1)
timeFrac = np.int32(5)

gamma = np.float32(1.4)
v0 = np.float32(0)
rhoL = np.float32(10)
rhoR = np.float32(1)
PressL = np.float32(100)
PressR = np.float32(1)
theta = np.float32(1.5)

fileList = []
savedTimes = np.array([0,0.08,0.15,0.23,0.33], dtype = np.float32)

'''
Create the for loop that will go through the various sizes of the spatial grid. 
There will be two cases: 
    
Case 1: Number of spatial grid points is numPointsArray[0] = 100
Case 2: Number of spatial grid points is numPointsArray[1] = 200
'''
timeCounter = 0

for num in numPointsArray:
    x_List, yList, tList = Init.createSpaceTimeGrid(length, num, timePoints[timeCounter], spaceDim)
    u1, u2, u3 = Init.setAndGetInitialVals(gamma, num+4, v0, rhoL, rhoR, PressL, PressR)
    deltaX = np.float32(x_List[1]-x_List[0])
    currentTime = 0
    
    '''
    Save Initial Density Data
    '''
    filename = 'RungeKuttaDensityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(num) + '.csv'
    fields = np.array(['Position (centimeters)', 'Mass Density (grams per cubic centimeter)'])
    Save.SaveDataAsCSVFile(x_List, u1[2:len(u1)-2], filename, fields)
    fileList.append(filename)
    print(filename)
            
    '''
    Save Initial Velocity Data
    '''
    filename = 'RungeKuttaVelocityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(num) + '.csv'
    fields = np.array(['Position (centimeters)', 'Velocity (centimeters per second)'])
    Save.SaveDataAsCSVFile(x_List, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
    fileList.append(filename)
    print(filename)
            
    '''
    Save Initial Pressure Data
    '''
    filename = 'RungeKuttaPressureDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(num) + '.csv'
    fields = np.array(['Position (centimeters)', 'Pressure (ergs per cubic centimeter)'])
    Save.SaveDataAsCSVFile(x_List, (gamma - 1)*(u3[2:len(u1)-2] - 0.5*(u2[2:len(u1)-2]**2/u1[2:len(u1)-2])), filename, fields)
    fileList.append(filename)
    print(filename)
    
    for t in range(len(tList)):
        timeStep, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, num+1, timeFrac, u1, u2, u3)
        DivF1, DivF2, DivF3 = Evolv.computeFluxDivergenceRungeKutta(deltaX, num, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta)
        
        if currentTime < savedTimes[1] and savedTimes[1] < currentTime + timeStep:
            timeStep = savedTimes[1] - currentTime
        
            u1, u2, u3 = Evolv.doEntireRungeKuttaTimeStep(num, u1, u2, u3, timeStep, DivF1, DivF2, DivF3, deltaX, gamma, theta, timeFrac)
            
            currentTime += timeStep
            
            
            '''
            Save Density Data
            '''
            filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Mass Density (grams per cubic centimeter)'])
            Save.SaveDataAsCSVFile(x_List, u1[2:len(u1)-2], filename, fields)
            fileList.append(filename)
            print(filename)
            
            '''
            Save Velocity Data
            '''
            filename = 'RungeKuttaVelocityDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Velocity (centimeters per second)'])
            Save.SaveDataAsCSVFile(x_List, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
            fileList.append(filename)
            print(filename)
            
            '''
            Save Pressure Data
            '''
            filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Pressure (ergs per cubic centimeter)'])
            Save.SaveDataAsCSVFile(x_List, (gamma - 1)*(u3[2:len(u1)-2] - 0.5*(u2[2:len(u1)-2]**2/u1[2:len(u1)-2])), filename, fields)
            fileList.append(filename)
            print(filename)
        elif currentTime < savedTimes[2] and savedTimes[2] < currentTime + timeStep:
            timeStep = savedTimes[2] - currentTime
            
            u1, u2, u3 = Evolv.doEntireRungeKuttaTimeStep(num, u1, u2, u3, timeStep, DivF1, DivF2, DivF3, deltaX, gamma, theta, timeFrac)

            currentTime += timeStep
            '''
            Save Density Data
            '''
            filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Mass Density (grams per cubic centimeter)'])
            Save.SaveDataAsCSVFile(x_List, u1[2:len(u1)-2], filename, fields)
            fileList.append(filename)
            print(filename)
            
            '''
            Save Velocity Data
            '''
            filename = 'RungeKuttaVelocityDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Velocity (centimeters per second)'])
            Save.SaveDataAsCSVFile(x_List, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
            fileList.append(filename)
            print(filename)
            
            '''
            Save Pressure Data
            '''
            filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Pressure (ergs per cubic centimeter)'])
            Save.SaveDataAsCSVFile(x_List, (gamma - 1)*(u3[2:len(u1)-2] - 0.5*(u2[2:len(u1)-2]**2/u1[2:len(u1)-2])), filename, fields)
            fileList.append(filename)
            print(filename)
        elif currentTime < savedTimes[3] and savedTimes[3] < currentTime + timeStep:
            timeStep = savedTimes[3] - currentTime
            
            u1, u2, u3 = Evolv.doEntireRungeKuttaTimeStep(num, u1, u2, u3, timeStep, DivF1, DivF2, DivF3, deltaX, gamma, theta, timeFrac)

            currentTime += timeStep
            '''
            Save Density Data
            '''
            filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Mass Density (grams per cubic centimeter)'])
            Save.SaveDataAsCSVFile(x_List, u1[2:len(u1)-2], filename, fields)
            fileList.append(filename)
            print(filename)
            
            '''
            Save Velocity Data
            '''
            filename = 'RungeKuttaVelocityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Velocity (centimeters per second)'])
            Save.SaveDataAsCSVFile(x_List, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
            fileList.append(filename)
            print(filename)
            
            '''
            Save Pressure Data
            '''
            filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Pressure (ergs per cubic centimeter)'])
            Save.SaveDataAsCSVFile(x_List, (gamma - 1)*(u3[2:len(u1)-2] - 0.5*(u2[2:len(u1)-2]**2/u1[2:len(u1)-2])), filename, fields)
            fileList.append(filename)
            print(filename)
        elif currentTime < savedTimes[4] and savedTimes[4] < currentTime + timeStep:
            timeStep = savedTimes[4] - currentTime
        
            u1, u2, u3 = Evolv.doEntireRungeKuttaTimeStep(num, u1, u2, u3, timeStep, DivF1, DivF2, DivF3, deltaX, gamma, theta, timeFrac)

            currentTime += timeStep
            
            '''
            Save Density Data
            '''
            filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Mass Density (grams per cubic centimeter)'])
            Save.SaveDataAsCSVFile(x_List, u1[2:len(u1)-2], filename, fields)
            fileList.append(filename)
            print(filename)
            
            '''
            Save Velocity Data
            '''
            filename = 'RungeKuttaVelocityDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Velocity (centimeters per second)'])
            Save.SaveDataAsCSVFile(x_List, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
            fileList.append(filename)
            print(filename)
            
            '''
            Save Pressure Data
            '''
            filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(num) + '.csv'
            fields = np.array(['Position (centimeters)', 'Pressure (ergs per cubic centimeter)'])
            Save.SaveDataAsCSVFile(x_List, (gamma - 1)*(u3[2:len(u1)-2] - 0.5*(u2[2:len(u1)-2]**2/u1[2:len(u1)-2])), filename, fields)
            fileList.append(filename)
            print(filename)
        else:
            u1, u2, u3 = Evolv.doEntireRungeKuttaTimeStep(num, u1, u2, u3, timeStep, DivF1, DivF2, DivF3, deltaX, gamma, theta, timeFrac)

            currentTime += timeStep
        
    timeCounter += 1
    
x, d1 = MP.ReadAndReturnData('RungeKuttaDensityDataAtTime=0.08withSpatialGridSize=100.csv')
y, d2 = MP.ReadAndReturnData('RungeKuttaDensityDataAtTime=0.08withSpatialGridSize=200.csv')

xlabel = 'Position (centimeters)'
ylabel = 'Density (Grams per Cubic Centimeter)'
title = 'Densities at Similar Times As Grid Size Varies'
saveName = 'RungeKuttaDensityConverge.png'

MP.MakeFigure2(x, y, xList, d1, d2, rho2, xlabel, ylabel,title, ['0.08 sec and N = 100', '0.08 sec and N = 200', '0.08 sec Exact'], saveName)


x, d1 = MP.ReadAndReturnData('RungeKuttaPressureDataAtTime=0.08withSpatialGridSize=100.csv')
y, d2 = MP.ReadAndReturnData('RungeKuttaPressureDataAtTime=0.08withSpatialGridSize=200.csv')

xlabel = 'Position (centimeters)'
ylabel = 'Pressure (Ergs per Cubic Centimeter)'
title = 'Pressures at Similar Times As Grid Size Varies'
saveName = 'RungeKuttaPressureConverge.png'

MP.MakeFigure2(x, y, xList, d1, d2, press2, xlabel, ylabel,title, ['0.08 sec and N = 100', '0.08 sec and N = 200', '0.08 sec Exact'], saveName)

x, d1 = MP.ReadAndReturnData('RungeKuttaVelocityDataAtTime=0.08withSpatialGridSize=100.csv')
y, d2 = MP.ReadAndReturnData('RungeKuttaVelocityDataAtTime=0.08withSpatialGridSize=200.csv')

xlabel = 'Position (centimeters)'
ylabel = 'Velocity (centimeters per second)'
title = 'Velocities at Similar Times As Grid Size Varies'
saveName = 'RungeKuttaVelocityConverge.png'

MP.MakeFigure2(x, y, xList, d1, d2, vel2, xlabel, ylabel,title, ['0.08 sec and N = 100', '0.08 sec and N = 200', '0.08 sec Exact'], saveName)
