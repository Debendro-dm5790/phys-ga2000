import numpy as np
import matplotlib.pyplot as plt

import Initialize as Init 
import Evolution as Evolv
import SaveData as Save
import MakePlots as MP

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

numPoints = np.int32(len(xList))
timePoints = np.int32(1600)
timeFrac = np.int32(5)

gamma = np.float32(1.4)
v0 = np.float32(0)
rhoL = np.float32(10)
rhoR = np.float32(1)
PressL = np.float32(100)
PressR = np.float32(1)
theta = np.float32(1.5)

fileList = []
savedTimes = np.array([time1,time2,time3,time4,time5], dtype = np.float32)

tList = np.zeros(timePoints, dtype = np.float32)
timeCounter = 0

u1, u2, u3 = Init.setAndGetInitialVals(gamma, numPoints+4, v0, rhoL, rhoR, PressL, PressR)

deltaX = np.float32(xList[1]-xList[0])
currentTime = 0

'''
Save Initial Density Data
'''
filename = 'RungeKuttaExactCompareDensityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (cm)', 'Mass Density (g per cubic centimeter)'])
Save.SaveDataAsCSVFile(xList, u1[2:len(u1)-2], filename, fields)
fileList.append(filename)
print(filename)
            
'''
Save Initial Velocity Data
'''
filename = 'RungeKuttaExactCompareVelocityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (cm)', 'Velocity (cm per second)'])
Save.SaveDataAsCSVFile(xList, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
fileList.append(filename)
print(filename)
            
'''
Save Initial Pressure Data
'''
filename = 'RungeKuttaExactComparePressureDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (cm)', 'Pressure (erg per cubic centimeter)'])
Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[2:len(u1)-2] - 0.5*(u2[2:len(u1)-2]**2/u1[2:len(u1)-2])), filename, fields)
fileList.append(filename)
print(filename)

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
        filename = 'RungeKuttaExactCompareDensityDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Mass Density (grams per cubic centimeter)'])
        Save.SaveDataAsCSVFile(xList, u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaExactCompareVelocityDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Velocity (cm per second)'])
        Save.SaveDataAsCSVFile(xList, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
        
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaExactComparePressureDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Pressure (erg per cubic centimeter)'])
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
        filename = 'RungeKuttaExactCompareDensityDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Mass Density (grams per cubic centimeter)'])
        Save.SaveDataAsCSVFile(xList, u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaExactCompareVelocityDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Velocity (cm per second)'])
        Save.SaveDataAsCSVFile(xList, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaExactComparePressureDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Pressure (erg per cubic centimeter)'])
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
        filename = 'RungeKuttaExactCompareDensityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Mass Density (grams per cubic centimeter)'])
        Save.SaveDataAsCSVFile(xList, u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaExactCompareVelocityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Velocity (cm per second)'])
        Save.SaveDataAsCSVFile(xList, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaExactComparePressureDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Pressure (erg per cubic centimeter)'])
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
        filename = 'RungeKuttaExactCompareDensityDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Mass Density (grams per cubic centimeter)'])
        Save.SaveDataAsCSVFile(xList, u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'RungeKuttaExactCompareVelocityDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Velocity (cm per second)'])
        Save.SaveDataAsCSVFile(xList, u2[2:len(u1)-2]/u1[2:len(u1)-2], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'RungeKuttaExactComparePressureDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (cm)', 'Pressure (erg per cubic centimeter)'])
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
MP.MakeFigure(x, d1, d2, d3, d4, d5, 'Position (cm)', 'Density (grams per cubic centimeter)', 'Densities Over Time', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'DensitiesRK.png')

x, v1 = MP.ReadAndReturnData(fileList[1])
x, v2 = MP.ReadAndReturnData(fileList[4])
x, v3 = MP.ReadAndReturnData(fileList[7])
x, v4 = MP.ReadAndReturnData(fileList[10])
x, v5 = MP.ReadAndReturnData(fileList[13])
MP.MakeFigure(x, v1, v2, v3, v4, v5, 'Position (cm)', 'Velocity (centimeters per second)', 'Velocities Over Time', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'VelocitiesRK.png')

x, P1 = MP.ReadAndReturnData(fileList[2])
x, P2 = MP.ReadAndReturnData(fileList[5])
x, P3 = MP.ReadAndReturnData(fileList[8])
x, P4 = MP.ReadAndReturnData(fileList[11])
x, P5 = MP.ReadAndReturnData(fileList[14])
MP.MakeFigure(x, P1, P2, P3, P4, P5, 'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Pressures Over Time', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'PressuresRK.png')

MP.MakeFigure3(xList,d1,rho1,'Position (cm)', 'Density (grams per cubic centimeter)', 'Comparing Densities at Time ' + str(time1) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxDensityTime='+str(time1)+'.png')
MP.MakeFigure3(xList,d2,rho2,'Position (cm)', 'Density (grams per cubic centimeter)', 'Comparing Densities at Time ' + str(time2) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxDensityTime='+str(time2)+'.png')
MP.MakeFigure3(xList,d3,rho3,'Position (cm)', 'Density (grams per cubic centimeter)', 'Comparing Densities at Time ' + str(time3) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxDensityTime='+str(time3)+'.png')
MP.MakeFigure3(xList,d4,rho4,'Position (cm)', 'Density (grams per cubic centimeter)', 'Comparing Densities at Time ' + str(time4) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxDensityTime='+str(time4)+'.png')
MP.MakeFigure3(xList,d5,rho5,'Position (cm)', 'Density (grams per cubic centimeter)', 'Comparing Densities at Time ' + str(time5) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxDensityTime='+str(time5)+'.png')

MP.MakeFigure3(xList,v1,vel1,'Position (cm)', 'Velocity (centimeters per second)', 'Comparing Velocities at Time ' + str(time1) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxVelocityTime='+str(time1)+'.png')
MP.MakeFigure3(xList,v2,vel2,'Position (cm)', 'Velocity (centimeters per second)', 'Comparing Velocities at Time ' + str(time2) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxVelocityTime='+str(time2)+'.png')
MP.MakeFigure3(xList,v3,vel3,'Position (cm)', 'Velocity (centimeters per second)', 'Comparing Velocities at Time ' + str(time3) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxVelocityTime='+str(time3)+'.png')
MP.MakeFigure3(xList,v4,vel4,'Position (cm)', 'Velocity (centimeters per second)', 'Comparing Velocities at Time ' + str(time4) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxVelocityTime='+str(time4)+'.png')
MP.MakeFigure3(xList,v5,vel5,'Position (cm)', 'Velocity (centimeters per second)', 'Comparing Velocities at Time ' + str(time5) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxVelocityTime='+str(time5)+'.png')

MP.MakeFigure3(xList,P1,press1,'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Comparing Pressures at Time ' + str(time1) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxPressureTime='+str(time1)+'.png')
MP.MakeFigure3(xList,P2,press2,'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Comparing Pressures at Time ' + str(time2) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxPressureTime='+str(time2)+'.png')
MP.MakeFigure3(xList,P3,press3,'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Comparing Pressures at Time ' + str(time3) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxPressureTime='+str(time3)+'.png')
MP.MakeFigure3(xList,P4,press4,'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Comparing Pressures at Time ' + str(time4) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxPressureTime='+str(time4)+'.png')
MP.MakeFigure3(xList,P5,press5,'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Comparing Pressures at Time ' + str(time5) + ' sec', ['Runge-Kutta Approximate', 'Fortran Exact'], 'FortranRKApproxPressureTime='+str(time5)+'.png')

MP.MakeFigure4(xList,(np.array(d1, dtype = np.float32) - rho1)/rho1,'Position (cm)', 'Density Fractional Residuals', 'Density Residuals at Time ' + str(time1) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxDensityResTime='+str(time1)+'.png', -2, 2)
MP.MakeFigure4(xList,(np.array(d2, dtype = np.float32) - rho2)/rho2,'Position (cm)', 'Density Fractional Residuals', 'Density Residuals at Time ' + str(time2) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxDensityResTime='+str(time2)+'.png', -1, 1)
MP.MakeFigure4(xList,(np.array(d3, dtype = np.float32) - rho3)/rho3,'Position (cm)', 'Density Fractional Residuals', 'Density Residuals at Time ' + str(time3) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxDensityResTime='+str(time3)+'.png',-0.4, 0.4)
MP.MakeFigure4(xList,(np.array(d4, dtype = np.float32) - rho4)/rho4,'Position (cm)', 'Density Fractional Residuals', 'Density Residuals at Time ' + str(time4) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxDensityResTime='+str(time4)+'.png', -1, 1)
MP.MakeFigure4(xList,(np.array(d5, dtype = np.float32) - rho5)/rho5,'Position (cm)', 'Density Fractional Residuals', 'Density Residuals at Time ' + str(time5) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxDensityResTime='+str(time5)+'.png', -0.75, 0.75)

MP.MakeFigure4(xList,(np.array(v1, dtype = np.float32) - vel1)/vel1,'Position (cm)', 'Velocity Fractional Residuals', 'Velocity Residuals at Time ' + str(time1) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxVelocityResTime='+str(time1)+'.png', -2, 2)
MP.MakeFigure4(xList,(np.array(v2, dtype = np.float32) - vel2)/vel2,'Position (cm)', 'Velocity Fractional Residuals', 'Velocity Residuals at Time ' + str(time2) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxVelocityResTime='+str(time2)+'.png', -2, 2)
MP.MakeFigure4(xList,(np.array(v3, dtype = np.float32) - vel3)/vel3,'Position (cm)', 'Velocity Fractional Residuals', 'Velocity Residuals at Time ' + str(time3) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxVelocityResTime='+str(time3)+'.png', -2, 2)
MP.MakeFigure4(xList,(np.array(v4, dtype = np.float32) - vel4)/vel4,'Position (cm)', 'Velocity Fractional Residuals', 'Velocity Residuals at Time ' + str(time4) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxVelocityResTime='+str(time4)+'.png', -2, 2)
MP.MakeFigure4(xList,(np.array(v5, dtype = np.float32) - vel5)/vel5,'Position (cm)', 'Velocity Fractional Residuals', 'Velocity Residuals at Time ' + str(time5) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxVelocityResTime='+str(time5)+'.png', -2, 2)

MP.MakeFigure4(xList,(np.array(P1, dtype = np.float32) - press1)/press1,'Position (cm)', 'Pressure Fractional Residuals', 'Pressure Residuals at Time ' + str(time1) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxPressureResTime='+str(time1)+'.png', -2, 2)
MP.MakeFigure4(xList,(np.array(P2, dtype = np.float32) - press2)/press2,'Position (cm)', 'Pressure Fractional Residuals', 'Pressure Residuals at Time ' + str(time2) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxPressureResTime='+str(time2)+'.png', -2, 2)
MP.MakeFigure4(xList,(np.array(P3, dtype = np.float32) - press3)/press3,'Position (cm)', 'Pressure Fractional Residuals', 'Pressure Residuals at Time ' + str(time3) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxPressureResTime='+str(time3)+'.png', -1, 1)
MP.MakeFigure4(xList,(np.array(P4, dtype = np.float32) - press4)/press4,'Position (cm)', 'Pressure Fractional Residuals', 'Pressure Residuals at Time ' + str(time4) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxPressureResTime='+str(time4)+'.png', -2, 2)
MP.MakeFigure4(xList,(np.array(P5, dtype = np.float32) - press5)/press5,'Position (cm)', 'Pressure Fractional Residuals', 'Pressure Residuals at Time ' + str(time5) + ' sec', ['(RK Approximate - Fortran Exact)/(Fortran Exact)'], 'FortranRKApproxPressureResTime='+str(time5)+'.png', -2, 2)


