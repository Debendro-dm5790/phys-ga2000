import numpy as np
import matplotlib.pyplot as plt

import Initialize_2d as Init 
import Evolution_2d as Evolv
import SaveData_2d as Save
import MakePlots_2d as MP

'''
Initialize all variables and constants related to the simulated system, the spatial grids,
the initial conditions of the problem, and the parameters of the functions defined in the modules
Initialize_2d.py and Evolution_2d.py
'''

task = 3 #Choose the intial conditions. Task can be 1,2, or 3.

if(task == 1):#Task 1
    length = np.float32(1)
    numPoints = np.int32(20) #should be even
    timePoints = np.int32(1600) #should be even
    spaceDim = np.int32(2)
    timeFrac = np.int32(5)
    theta = np.float32(1.5)
    
    gamma = np.float32(1.4)
    v0x = np.float32(0)
    v0y = np.float32(0)
    rhoL = np.float32(100)
    rhoR = np.float32(10)
    PressL = np.float32(80)
    PressR = np.float32(10)
    PeriodicX = False
    PeriodicY = True


elif(task == 2):#Task 2
    length = np.float32(1)
    numPoints = np.int32(20) #should be even
    timePoints = np.int32(1600) #should be even
    spaceDim = np.int32(2)
    timeFrac = np.int32(5)
    theta = np.float32(1.5)
    
    gamma = np.float32(1.4)
    v0x = np.float32(5) #relative velocity is 2*v0x, choices are v0x=2,5,10
    v0y = np.float32(0)
    rhoL = np.float32(1)
    rhoR = np.float32(1)
    PressL = np.float32(40)
    PressR = np.float32(40)
    PeriodicX = True
    PeriodicY = True


elif(task == 3):#Task 3
    length = np.float32(1)
    numPoints = np.int32(20) #should be even
    timePoints = np.int32(6000) #should be even
    spaceDim = np.int32(2)
    timeFrac = np.int32(5)
    theta = np.float32(1.5)
    
    gamma = np.float32(1.4)
    v0x = np.float32(0)
    v0y = np.float32(0)
    rhoL = np.float32(1) #the low value
    rhoR = np.float32(100) #the high value
    PressL = np.float32(10)
    PressR = np.float32(10)
    PeriodicX = False
    PeriodicY = False


fileList = []
savedTimes = np.array([0,0.08,0.16,0.24,0.32], dtype = np.float32)

if(task == 3):
	savedTimes = np.array([0,0.04,0.06,0.08,0.2], dtype = np.float32)

'''
Begin the simulation of the 1D Riemann Problem with the higher order Runge-Kutta method
by initializing the system and getting and saving the initial values.
'''
xList, yList, tList = Init.createSpaceTimeGrid(length, numPoints, timePoints, spaceDim)
u1, u2, u3, u4 = Init.setInitialVals(task, gamma, numPoints+4, v0x, v0y, rhoL, rhoR, PressL, PressR)
deltaX = np.float32(xList[1]-xList[0])
currentTime = 0

'''
Save Initial Density Data
'''
filename = 'RungeKuttaDensityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
Save.SaveDataAsCSVFile(xList, yList, u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
fileList.append(filename)
print(filename)

'''
Save Initial Velocity Data
'''
filename = 'RungeKuttaXVelocityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (meters)', 'Velocity-x (meters per second)'])
Save.SaveDataAsCSVFile(xList, yList, u2[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
fileList.append(filename)
print(filename)

filename = 'RungeKuttaYVelocityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (meters)', 'Velocity-y (meters per second)'])
Save.SaveDataAsCSVFile(xList, yList, u3[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
fileList.append(filename)
print(filename)
            
'''
Save Initial Pressure Data
'''
filename = 'RungeKuttaPressureDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
Save.SaveDataAsCSVFile(xList, yList, (gamma - 1)*(u4[2:len(u1)-2, 2:len(u1)-2] - 0.5*(u2[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2]) \
                       - 0.5*(u3[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2])), filename, fields)
fileList.append(filename)
print(filename)

'''
check initial conditions
'''
x, y, d1 = MP.ReadAndReturnData(fileList[0])
MP.MakeFigure1(x, y, d1, 'x-Position (meters)', 'y-Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Densities Over Time at t=' + str(currentTime) + '.png')
        
x, y, v1 = MP.ReadAndReturnData(fileList[1])
MP.MakeFigure1(x, y, v1, 'x-Position (meters)', 'y-Position (meters)', 'Velocity-x(meters per second)', 'X-Velocities Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'X-velocity Over Time at t=' + str(currentTime) + '.png')
            
x, y, v1 = MP.ReadAndReturnData(fileList[2])
MP.MakeFigure1(x, y, v1, 'x-Position (meters)', 'y-Position (meters)', 'Velocity-y(meters per second)', 'Y-Velocities Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Y-velocity Over Time at t=' + str(currentTime) + '.png')
            
x, y, P1 = MP.ReadAndReturnData(fileList[3])
MP.MakeFigure1(x, y, P1, 'x-Position (meters)', 'y-Position (meters)', 'Pressure (Pascal)', 'Pressures Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Pressure Over Time at t=' + str(currentTime) + '.png')

count=int(0)

'''
Start and continue the evolution of the 2D hydrodynamic system by using the higher order Runge-Kutta method
'''
for t in range(len(tList)):
	
    timeStep, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta , F4_HLL_RungeKutta,\
           G1_HLL_RungeKutta, G2_HLL_RungeKutta, G3_HLL_RungeKutta, G4_HLL_RungeKutta \
           = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1, u2, u3, u4)
    DivF1, DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4 = \
    Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta,\
    F4_HLL_RungeKutta, G1_HLL_RungeKutta, G2_HLL_RungeKutta, G3_HLL_RungeKutta, G4_HLL_RungeKutta)
        
    if currentTime < savedTimes[1] and savedTimes[1] < currentTime + timeStep:
        timeStep = savedTimes[1] - currentTime
        
        u1_1, u2_1, u3_1, u4_1 = Evolv.updateQuantitiesRungeKutta_order1(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, DivF1, \
        DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4)
        
        tt, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1 , F4_HLL_RungeKutta_1,\
               G1_HLL_RungeKutta_1, G2_HLL_RungeKutta_1, G3_HLL_RungeKutta_1, G4_HLL_RungeKutta_1 \
               = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_1, u2_1, u3_1, u4_1)
        DivF1_1, DivF2_1, DivF3_1, DivF4_1, DivG1_1, DivG2_1, DivG3_1, DivG4_1 = \
        Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1,\
        F4_HLL_RungeKutta_1, G1_HLL_RungeKutta_1, G2_HLL_RungeKutta_1, G3_HLL_RungeKutta_1, G4_HLL_RungeKutta_1)
        
        u1_2, u2_2, u3_2, u4_2 = Evolv.updateQuantitiesRungeKutta_order2(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_1, u2_1, u3_1, u4_1, DivF1_1, \
        DivF2_1, DivF3_1, DivF4_1, DivG1_1, DivG2_1, DivG3_1, DivG4_1)
        
        tt2, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2 , F4_HLL_RungeKutta_2,\
               G1_HLL_RungeKutta_2, G2_HLL_RungeKutta_2, G3_HLL_RungeKutta_2, G4_HLL_RungeKutta_2 \
               = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_2, u2_2, u3_2, u4_2)
        DivF1_2, DivF2_2, DivF3_2, DivF4_2, DivG1_2, DivG2_2, DivG3_2, DivG4_2 = \
        Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2,\
        F4_HLL_RungeKutta_2, G1_HLL_RungeKutta_2, G2_HLL_RungeKutta_2, G3_HLL_RungeKutta_2, G4_HLL_RungeKutta_2)
        
        u1, u2, u3, u4 = Evolv.updateQuantitiesRungeKutta(task, timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_2, u2_2, u3_2, u4_2, DivF1_2, \
        DivF2_2, DivF3_2, DivF4_2, DivG1_2, DivG2_2, DivG3_2, DivG4_2)
        
        currentTime += timeStep
            
        '''
        Save Density Data
        '''
        filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, yList, u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
        
        if(task == 3):
            x, y, d1 = MP.ReadAndReturnData(filename)
            d1 = np.maximum(0,d1)
            MP.MakeFigure1(x, y, d1, 'x-Position (meters)', 'y-Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Densities Over Time at t=' + str(currentTime) + '.png')

            
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaxVelocityDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity-x (meters per second)'])
        Save.SaveDataAsCSVFile(xList, yList, u2[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
     
        
        filename = 'RungeKuttayVelocityDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity-y (meters per second)'])
        Save.SaveDataAsCSVFile(xList, yList, u3[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
        
        
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, yList, (gamma - 1)*(u4[2:len(u1)-2, 2:len(u1)-2] - 0.5*(u2[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2]) \
                       - 0.5*(u3[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2])), filename, fields)
        fileList.append(filename)
        print(filename)
       
        
    elif currentTime < savedTimes[2] and savedTimes[2] < currentTime + timeStep:
        timeStep = savedTimes[2] - currentTime
        
        u1_1, u2_1, u3_1, u4_1 = Evolv.updateQuantitiesRungeKutta_order1(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, DivF1, \
        DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4)
        
        tt, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1 , F4_HLL_RungeKutta_1,\
               G1_HLL_RungeKutta_1, G2_HLL_RungeKutta_1, G3_HLL_RungeKutta_1, G4_HLL_RungeKutta_1 \
               = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_1, u2_1, u3_1, u4_1)
        DivF1_1, DivF2_1, DivF3_1, DivF4_1, DivG1_1, DivG2_1, DivG3_1, DivG4_1 = \
        Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1,\
        F4_HLL_RungeKutta_1, G1_HLL_RungeKutta_1, G2_HLL_RungeKutta_1, G3_HLL_RungeKutta_1, G4_HLL_RungeKutta_1)
        
        u1_2, u2_2, u3_2, u4_2 = Evolv.updateQuantitiesRungeKutta_order2(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_1, u2_1, u3_1, u4_1, DivF1_1, \
        DivF2_1, DivF3_1, DivF4_1, DivG1_1, DivG2_1, DivG3_1, DivG4_1)
        
        tt2, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2 , F4_HLL_RungeKutta_2,\
               G1_HLL_RungeKutta_2, G2_HLL_RungeKutta_2, G3_HLL_RungeKutta_2, G4_HLL_RungeKutta_2 \
               = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_2, u2_2, u3_2, u4_2)
        DivF1_2, DivF2_2, DivF3_2, DivF4_2, DivG1_2, DivG2_2, DivG3_2, DivG4_2 = \
        Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2,\
        F4_HLL_RungeKutta_2, G1_HLL_RungeKutta_2, G2_HLL_RungeKutta_2, G3_HLL_RungeKutta_2, G4_HLL_RungeKutta_2)
        
        u1, u2, u3, u4 = Evolv.updateQuantitiesRungeKutta(task, timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_2, u2_2, u3_2, u4_2, DivF1_2, \
        DivF2_2, DivF3_2, DivF4_2, DivG1_2, DivG2_2, DivG3_2, DivG4_2)
        
        currentTime += timeStep
        '''
        Save Density Data
        '''
        filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, yList, u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
        
        if(task == 3):
            x, y, d1 = MP.ReadAndReturnData(filename)
            d1 = np.maximum(0,d1)
            MP.MakeFigure1(x, y, d1, 'x-Position (meters)', 'y-Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Densities Over Time at t=' + str(currentTime) + '.png')
            
            
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaxVelocityDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity-x (meters per second)'])
        Save.SaveDataAsCSVFile(xList, yList, u2[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
        
        filename = 'RungeKuttayVelocityDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity-y (meters per second)'])
        Save.SaveDataAsCSVFile(xList, yList, u3[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, yList, (gamma - 1)*(u4[2:len(u1)-2, 2:len(u1)-2] - 0.5*(u2[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2]) \
                       - 0.5*(u3[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2])), filename, fields)
        fileList.append(filename)
        print(filename)
        
    elif currentTime < savedTimes[3] and savedTimes[3] < currentTime + timeStep:
        timeStep = savedTimes[3] - currentTime
            
        u1_1, u2_1, u3_1, u4_1 = Evolv.updateQuantitiesRungeKutta_order1(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, DivF1, \
        DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4)
        
        tt, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1 , F4_HLL_RungeKutta_1,\
               G1_HLL_RungeKutta_1, G2_HLL_RungeKutta_1, G3_HLL_RungeKutta_1, G4_HLL_RungeKutta_1 \
               = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_1, u2_1, u3_1, u4_1)
        DivF1_1, DivF2_1, DivF3_1, DivF4_1, DivG1_1, DivG2_1, DivG3_1, DivG4_1 = \
        Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1,\
        F4_HLL_RungeKutta_1, G1_HLL_RungeKutta_1, G2_HLL_RungeKutta_1, G3_HLL_RungeKutta_1, G4_HLL_RungeKutta_1)
        
        u1_2, u2_2, u3_2, u4_2 = Evolv.updateQuantitiesRungeKutta_order2(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_1, u2_1, u3_1, u4_1, DivF1_1, \
        DivF2_1, DivF3_1, DivF4_1, DivG1_1, DivG2_1, DivG3_1, DivG4_1)
        
        tt2, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2 , F4_HLL_RungeKutta_2,\
               G1_HLL_RungeKutta_2, G2_HLL_RungeKutta_2, G3_HLL_RungeKutta_2, G4_HLL_RungeKutta_2 \
               = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_2, u2_2, u3_2, u4_2)
        DivF1_2, DivF2_2, DivF3_2, DivF4_2, DivG1_2, DivG2_2, DivG3_2, DivG4_2 = \
        Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2,\
        F4_HLL_RungeKutta_2, G1_HLL_RungeKutta_2, G2_HLL_RungeKutta_2, G3_HLL_RungeKutta_2, G4_HLL_RungeKutta_2)
        
        u1, u2, u3, u4 = Evolv.updateQuantitiesRungeKutta(task, timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_2, u2_2, u3_2, u4_2, DivF1_2, \
        DivF2_2, DivF3_2, DivF4_2, DivG1_2, DivG2_2, DivG3_2, DivG4_2)
        
        currentTime += timeStep
        '''
        Save Density Data
        '''
        filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, yList, u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
        
        if(task == 3):
            x, y, d1 = MP.ReadAndReturnData(filename)
            d1 = np.maximum(0,d1)
            MP.MakeFigure1(x, y, d1, 'x-Position (meters)', 'y-Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Densities Over Time at t=' + str(currentTime) + '.png')
          
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaxVelocityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity-x (meters per second)'])
        Save.SaveDataAsCSVFile(xList, yList, u2[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
        
        filename = 'RungeKuttayVelocityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity-y (meters per second)'])
        Save.SaveDataAsCSVFile(xList, yList, u3[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, yList, (gamma - 1)*(u4[2:len(u1)-2, 2:len(u1)-2] - 0.5*(u2[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2]) \
                       - 0.5*(u3[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2])), filename, fields)
        fileList.append(filename)
        print(filename)
        
    elif currentTime < savedTimes[4] and savedTimes[4] < currentTime + timeStep:
        timeStep = savedTimes[4] - currentTime
        
        u1_1, u2_1, u3_1, u4_1 = Evolv.updateQuantitiesRungeKutta_order1(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, DivF1, \
        DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4)
        
        tt, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1 , F4_HLL_RungeKutta_1,\
               G1_HLL_RungeKutta_1, G2_HLL_RungeKutta_1, G3_HLL_RungeKutta_1, G4_HLL_RungeKutta_1 \
               = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_1, u2_1, u3_1, u4_1)
        DivF1_1, DivF2_1, DivF3_1, DivF4_1, DivG1_1, DivG2_1, DivG3_1, DivG4_1 = \
        Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1,\
        F4_HLL_RungeKutta_1, G1_HLL_RungeKutta_1, G2_HLL_RungeKutta_1, G3_HLL_RungeKutta_1, G4_HLL_RungeKutta_1)
        
        u1_2, u2_2, u3_2, u4_2 = Evolv.updateQuantitiesRungeKutta_order2(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_1, u2_1, u3_1, u4_1, DivF1_1, \
        DivF2_1, DivF3_1, DivF4_1, DivG1_1, DivG2_1, DivG3_1, DivG4_1)
        
        tt2, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2 , F4_HLL_RungeKutta_2,\
               G1_HLL_RungeKutta_2, G2_HLL_RungeKutta_2, G3_HLL_RungeKutta_2, G4_HLL_RungeKutta_2 \
               = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_2, u2_2, u3_2, u4_2)
        DivF1_2, DivF2_2, DivF3_2, DivF4_2, DivG1_2, DivG2_2, DivG3_2, DivG4_2 = \
        Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2,\
        F4_HLL_RungeKutta_2, G1_HLL_RungeKutta_2, G2_HLL_RungeKutta_2, G3_HLL_RungeKutta_2, G4_HLL_RungeKutta_2)
        
        u1, u2, u3, u4 = Evolv.updateQuantitiesRungeKutta(task, timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_2, u2_2, u3_2, u4_2, DivF1_2, \
        DivF2_2, DivF3_2, DivF4_2, DivG1_2, DivG2_2, DivG3_2, DivG4_2)
        
        currentTime += timeStep
            
        '''
        Save Density Data
        '''
        filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, yList, u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
        
        if(task == 3):
            x, y, d1 = MP.ReadAndReturnData(filename)
            d1 = np.maximum(0,d1)
            MP.MakeFigure1(x, y, d1, 'x-Position (meters)', 'y-Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Densities Over Time at t=' + str(currentTime) + '.png')
          
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaxVelocityDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity-x (meters per second)'])
        Save.SaveDataAsCSVFile(xList, yList, u2[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
        
        filename = 'RungeKuttayVelocityDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity-y (meters per second)'])
        Save.SaveDataAsCSVFile(xList, yList, u3[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, yList, (gamma - 1)*(u4[2:len(u1)-2, 2:len(u1)-2] - 0.5*(u2[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2]) \
                       - 0.5*(u3[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2])), filename, fields)
        fileList.append(filename)
        print(filename)
        
        break
    
    else:
		#for chosen time steps
        '''
        if (count%20 == 0):
            filename = 'RungeKuttaDensityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
            fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
            Save.SaveDataAsCSVFile(xList, yList, u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
            print(filename)
            x, y, d1 = MP.ReadAndReturnData(filename)
            MP.MakeFigure1(x, y, d1, 'x-Position (meters)', 'y-Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Densities Over Time at t=' + str(currentTime) + '.png')
           
        
            filename = 'RungeKuttaxVelocityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
            fields = np.array(['Position (meters)', 'Velocity-x (meters per second)'])
            Save.SaveDataAsCSVFile(xList, yList, u2[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
            print(filename)
            x, y, v1 = MP.ReadAndReturnData(filename)
            MP.MakeFigure1(x, y, v1, 'x-Position (meters)', 'y-Position (meters)', 'Velocity-x(meters per second)', 'X-Velocities Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'X-velocity Over Time at t=' + str(currentTime) + '.png')
            
            
            filename = 'RungeKuttayVelocityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
            fields = np.array(['Position (meters)', 'Velocity-y (meters per second)'])
            Save.SaveDataAsCSVFile(xList, yList, u3[2:len(u1)-2, 2:len(u1)-2]/u1[2:len(u1)-2, 2:len(u1)-2], filename, fields)
            print(filename)
            x, y, v1 = MP.ReadAndReturnData(filename)
            MP.MakeFigure1(x, y, v1, 'x-Position (meters)', 'y-Position (meters)', 'Velocity-y(meters per second)', 'Y-Velocities Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Y-velocity Over Time at t=' + str(currentTime) + '.png')
                
            filename = 'RungeKuttaPressureDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
            fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
            Save.SaveDataAsCSVFile(xList, yList, (gamma - 1)*(u4[2:len(u1)-2, 2:len(u1)-2] - 0.5*(u2[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2]) \
                           - 0.5*(u3[2:len(u1)-2, 2:len(u1)-2]**2/u1[2:len(u1)-2, 2:len(u1)-2])), filename, fields)
            print(filename)
            x, y, P1 = MP.ReadAndReturnData(filename)
            MP.MakeFigure1(x, y, P1, 'x-Position (meters)', 'y-Position (meters)', 'Pressure (Pascal)', 'Pressures Over Time at t=' + str(currentTime) + ' sec', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Pressure Over Time at t=' + str(currentTime) + '.png')
        '''
            
        u1_1, u2_1, u3_1, u4_1 = Evolv.updateQuantitiesRungeKutta_order1(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, DivF1, \
        DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4)
        
        tt, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1 , F4_HLL_RungeKutta_1,\
               G1_HLL_RungeKutta_1, G2_HLL_RungeKutta_1, G3_HLL_RungeKutta_1, G4_HLL_RungeKutta_1 \
               = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_1, u2_1, u3_1, u4_1)
        DivF1_1, DivF2_1, DivF3_1, DivF4_1, DivG1_1, DivG2_1, DivG3_1, DivG4_1 = \
        Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1,\
        F4_HLL_RungeKutta_1, G1_HLL_RungeKutta_1, G2_HLL_RungeKutta_1, G3_HLL_RungeKutta_1, G4_HLL_RungeKutta_1)
        
        u1_2, u2_2, u3_2, u4_2 = Evolv.updateQuantitiesRungeKutta_order2(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_1, u2_1, u3_1, u4_1, DivF1_1, \
        DivF2_1, DivF3_1, DivF4_1, DivG1_1, DivG2_1, DivG3_1, DivG4_1)
        
        tt2, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2 , F4_HLL_RungeKutta_2,\
               G1_HLL_RungeKutta_2, G2_HLL_RungeKutta_2, G3_HLL_RungeKutta_2, G4_HLL_RungeKutta_2 \
               = Evolv.computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_2, u2_2, u3_2, u4_2)
        DivF1_2, DivF2_2, DivF3_2, DivF4_2, DivG1_2, DivG2_2, DivG3_2, DivG4_2 = \
        Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2,\
        F4_HLL_RungeKutta_2, G1_HLL_RungeKutta_2, G2_HLL_RungeKutta_2, G3_HLL_RungeKutta_2, G4_HLL_RungeKutta_2)
        
        u1, u2, u3, u4 = Evolv.updateQuantitiesRungeKutta(task, timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_2, u2_2, u3_2, u4_2, DivF1_2, \
        DivF2_2, DivF3_2, DivF4_2, DivG1_2, DivG2_2, DivG3_2, DivG4_2)
        
        '''
        count += 1
        
        if(count>100):
            break
        '''
        currentTime += timeStep
    
'''
make plots
''' 
if (task != 3):
    x, y, d1 = MP.ReadAndReturnData(fileList[0])
    x, y, d2 = MP.ReadAndReturnData(fileList[4])
    x, y, d3 = MP.ReadAndReturnData(fileList[8])
    x, y, d4 = MP.ReadAndReturnData(fileList[12])
    x, y, d5 = MP.ReadAndReturnData(fileList[16])
    MP.MakeFigure(x, y, d1, d2, d3, d4, d5, 'x-Position (meters)', 'y-Position (meters)', 'Density (kg per cubic meter)', 'Densities Over Time for Grid Size = 20 Runge-Kutta', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'RungeKattaDensitiesGridSize=400.png')
    
    x, y, v1 = MP.ReadAndReturnData(fileList[1])
    x, y, v2 = MP.ReadAndReturnData(fileList[5])
    x, y, v3 = MP.ReadAndReturnData(fileList[9])
    x, y, v4 = MP.ReadAndReturnData(fileList[13])
    x, y, v5 = MP.ReadAndReturnData(fileList[17])
    MP.MakeFigure(x, y, v1, v2, v3, v4, v5, 'x-Position (meters)', 'y-Position (meters)', 'Velocity-x(meters per second)', 'X-Velocities Over Time for Grid Size = 20 Runge-Kutta', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'RungeKuttaXVelocitiesGridSize=400.png')
    
    x, y, v1 = MP.ReadAndReturnData(fileList[2])
    x, y, v2 = MP.ReadAndReturnData(fileList[6])
    x, y, v3 = MP.ReadAndReturnData(fileList[10])
    x, y, v4 = MP.ReadAndReturnData(fileList[14])
    x, y, v5 = MP.ReadAndReturnData(fileList[18])
    MP.MakeFigure(x, y, v1, v2, v3, v4, v5, 'x-Position (meters)', 'y-Position (meters)', 'Velocity-y(meters per second)', 'Y-Velocities Over Time for Grid Size = 20 Runge-Kutta', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'RungeKuttaYVelocitiesGridSize=400.png')
    
    x, y, P1 = MP.ReadAndReturnData(fileList[3])
    x, y, P2 = MP.ReadAndReturnData(fileList[7])
    x, y, P3 = MP.ReadAndReturnData(fileList[11])
    x, y, P4 = MP.ReadAndReturnData(fileList[15])
    x, y, P5 = MP.ReadAndReturnData(fileList[19])
    MP.MakeFigure(x, y, P1, P2, P3, P4, P5, 'x-Position (meters)', 'y-Position (meters)', 'Pressure (Pascal)', 'Pressures Over Time for Grid Size = 20 Runge-Kutta', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'RungeKuttaPressuresGridSize=400.png')



