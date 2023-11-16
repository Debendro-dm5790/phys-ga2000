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

fileList = []
savedTimes = np.array([time1,time2,time3,time4,time5], dtype = np.float32)

tList = np.zeros(timePoints, dtype = np.float32)
timeCounter = 0

u1, u2, u3 = Init.setAndGetInitialVals(gamma, numPoints+2, v0, rhoL, rhoR, PressL, PressR)

deltaX = np.float32(xList[1]-xList[0])
currentTime = 0
    
'''
Save Initial Density Data
'''
filename = 'DensityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (cm)', 'Mass Density (g per cubic centimeter)'])
Save.SaveDataAsCSVFile(xList, u1[1:len(u1)-1], filename, fields)
fileList.append(filename)
print(filename)
            
'''
Save Initial Velocity Data
'''
filename = 'VelocityDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (cm)', 'Velocity (cm per second)'])
Save.SaveDataAsCSVFile(xList, u2[1:len(u1)-1]/u1[1:len(u1)-1], filename, fields)
fileList.append(filename)
print(filename)
            
'''
Save Initial Pressure Data
'''
filename = 'PressureDataAtTime=' + str(currentTime) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
fields = np.array(['Position (cm)', 'Pressure (erg per cubic centimeter)'])
Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[1:len(u1)-1] - 0.5*(u2[1:len(u1)-1]**2/u1[1:len(u1)-1])), filename, fields)
fileList.append(filename)
print(filename)

for t in range(len(tList)):
    timeStep, F1_HLL, F2_HLL, F3_HLL = Evolv.computeTimeStepandFHLL(deltaX, gamma, numPoints+1, timeFrac, u1, u2, u3)
    DivF1, DivF2, DivF3 = Evolv.computeFluxDivergence(deltaX, numPoints, F1_HLL, F2_HLL, F3_HLL)
        
    if currentTime < savedTimes[1] and savedTimes[1] < currentTime + timeStep:
        timeStep = savedTimes[1] - currentTime
            
        u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
        currentTime += timeStep
            
        '''
        Save Density Data
        '''
        filename = 'DensityDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, u1[1:len(u1)-1], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'VelocityDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
        Save.SaveDataAsCSVFile(xList, u2[1:len(u1)-1]/u1[1:len(u1)-1], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'PressureDataAtTime=' + str(savedTimes[1]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[1:len(u1)-1] - 0.5*(u2[1:len(u1)-1]**2/u1[1:len(u1)-1])), filename, fields)
        fileList.append(filename)
        print(filename)
    elif currentTime < savedTimes[2] and savedTimes[2] < currentTime + timeStep:
        timeStep = savedTimes[2] - currentTime
        
        u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
        currentTime += timeStep
        '''
        Save Density Data
        '''
        filename = 'DensityDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, u1[1:len(u1)-1], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'VelocityDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
        Save.SaveDataAsCSVFile(xList, u2[1:len(u1)-1]/u1[1:len(u1)-1], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'PressureDataAtTime=' + str(savedTimes[2]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[1:len(u1)-1] - 0.5*(u2[1:len(u1)-1]**2/u1[1:len(u1)-1])), filename, fields)
        fileList.append(filename)
        print(filename)
    elif currentTime < savedTimes[3] and savedTimes[3] < currentTime + timeStep:
        timeStep = savedTimes[3] - currentTime
            
        u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
        currentTime += timeStep
        '''
        Save Density Data
        '''
        filename = 'DensityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, u1[1:len(u1)-1], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'VelocityDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
        Save.SaveDataAsCSVFile(xList, u2[1:len(u1)-1]/u1[1:len(u1)-1], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'PressureDataAtTime=' + str(savedTimes[3]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[1:len(u1)-1] - 0.5*(u2[1:len(u1)-1]**2/u1[1:len(u1)-1])), filename, fields)
        fileList.append(filename)
        print(filename)
    elif currentTime < savedTimes[4] and savedTimes[4] < currentTime + timeStep:
        timeStep = savedTimes[4] - currentTime
            
        u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
        currentTime += timeStep
            
        '''
        Save Density Data
        '''
        filename = 'DensityDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Mass Density (kg per cubic meter)'])
        Save.SaveDataAsCSVFile(xList, u1[1:len(u1)-1], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Velocity Data
        '''
        filename = 'VelocityDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Velocity (meters per second)'])
        Save.SaveDataAsCSVFile(xList, u2[1:len(u1)-1]/u1[1:len(u1)-1], filename, fields)
        fileList.append(filename)
        print(filename)
            
        '''
        Save Pressure Data
        '''
        filename = 'PressureDataAtTime=' + str(savedTimes[4]) + 'withSpatialGridSize=' + str(numPoints) + '.csv'
        fields = np.array(['Position (meters)', 'Pressure (Pascals)'])
        Save.SaveDataAsCSVFile(xList, (gamma - 1)*(u3[1:len(u1)-1] - 0.5*(u2[1:len(u1)-1]**2/u1[1:len(u1)-1])), filename, fields)
        fileList.append(filename)
        print(filename)
            
    u1, u2, u3 = Evolv.updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3)
    currentTime += timeStep
            
    timeCounter += 1
            
            
x, d1 = MP.ReadAndReturnData(fileList[0])
x, d2 = MP.ReadAndReturnData(fileList[3])
x, d3 = MP.ReadAndReturnData(fileList[6])
x, d4 = MP.ReadAndReturnData(fileList[9])
x, d5 = MP.ReadAndReturnData(fileList[12])
MP.MakeFigure(x, d1, d2, d3, d4, d5, 'Position (cm)', 'Density (grams per cubic centimeter)', 'Densities Over Time', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Densities.png')

x, v1 = MP.ReadAndReturnData(fileList[1])
x, v2 = MP.ReadAndReturnData(fileList[4])
x, v3 = MP.ReadAndReturnData(fileList[7])
x, v4 = MP.ReadAndReturnData(fileList[10])
x, v5 = MP.ReadAndReturnData(fileList[13])
MP.MakeFigure(x, v1, v2, v3, v4, v5, 'Position (cm)', 'Velocity (centimeters per second)', 'Velocities Over Time', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Velocities.png')

x, P1 = MP.ReadAndReturnData(fileList[2])
x, P2 = MP.ReadAndReturnData(fileList[5])
x, P3 = MP.ReadAndReturnData(fileList[8])
x, P4 = MP.ReadAndReturnData(fileList[11])
x, P5 = MP.ReadAndReturnData(fileList[14])
MP.MakeFigure(x, P1, P2, P3, P4, P5, 'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Pressures Over Time', [str(savedTimes[0]) + ' sec', str(savedTimes[1]) + ' sec', str(savedTimes[2]) + ' sec', str(savedTimes[3]) + ' sec', str(savedTimes[4]) + ' sec'], 'Pressures.png')

MP.MakeFigure3(xList,d1,rho1,'Position (cm)', 'Density (grams per cubic centimeter)', 'Comparing Densities at Time ' + str(time1) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonDensityTime='+str(time1)+'.png')
MP.MakeFigure3(xList,d2,rho2,'Position (cm)', 'Density (grams per cubic centimeter)', 'Comparing Densities at Time ' + str(time2) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonDensityTime='+str(time2)+'.png')
MP.MakeFigure3(xList,d3,rho3,'Position (cm)', 'Density (grams per cubic centimeter)', 'Comparing Densities at Time ' + str(time3) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonDensityTime='+str(time3)+'.png')
MP.MakeFigure3(xList,d4,rho4,'Position (cm)', 'Density (grams per cubic centimeter)', 'Comparing Densities at Time ' + str(time4) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonDensityTime='+str(time4)+'.png')
MP.MakeFigure3(xList,d5,rho5,'Position (cm)', 'Density (grams per cubic centimeter)', 'Comparing Densities at Time ' + str(time5) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonDensityTime='+str(time5)+'.png')

MP.MakeFigure3(xList,v1,vel1,'Position (cm)', 'Velocity (centimeters per second)', 'Comparing Velocities at Time ' + str(time1) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonVelocityTime='+str(time1)+'.png')
MP.MakeFigure3(xList,v2,vel2,'Position (cm)', 'Velocity (centimeters per second)', 'Comparing Velocities at Time ' + str(time2) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonVelocityTime='+str(time2)+'.png')
MP.MakeFigure3(xList,v3,vel3,'Position (cm)', 'Velocity (centimeters per second)', 'Comparing Velocities at Time ' + str(time3) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonVelocityTime='+str(time3)+'.png')
MP.MakeFigure3(xList,v4,vel4,'Position (cm)', 'Velocity (centimeters per second)', 'Comparing Velocities at Time ' + str(time4) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonVelocityTime='+str(time4)+'.png')
MP.MakeFigure3(xList,v5,vel5,'Position (cm)', 'Velocity (centimeters per second)', 'Comparing Velocities at Time ' + str(time5) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonVelocityTime='+str(time5)+'.png')

MP.MakeFigure3(xList,P1,press1,'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Comparing Pressures at Time ' + str(time1) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonPressureTime='+str(time1)+'.png')
MP.MakeFigure3(xList,P2,press2,'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Comparing Pressures at Time ' + str(time2) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonPressureTime='+str(time2)+'.png')
MP.MakeFigure3(xList,P3,press3,'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Comparing Pressures at Time ' + str(time3) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonPressureTime='+str(time3)+'.png')
MP.MakeFigure3(xList,P4,press4,'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Comparing Pressures at Time ' + str(time4) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonPressureTime='+str(time4)+'.png')
MP.MakeFigure3(xList,P5,press5,'Position (cm)', 'Pressure (ergs per cubic centimeter)', 'Comparing Pressures at Time ' + str(time5) + ' sec', ['Python Approximate', 'Fortran Exact'], 'FortranPythonPressureTime='+str(time5)+'.png')

MP.MakeFigure4(xList,np.array(d1, dtype = np.float32) - rho1,'Position (cm)', 'Density Residuals (grams per cubic centimeter)', 'Density Residuals at Time ' + str(time1) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonDensityResTime='+str(time1)+'.png')
MP.MakeFigure4(xList,np.array(d2, dtype = np.float32) - rho2,'Position (cm)', 'Density Residuals (grams per cubic centimeter)', 'Density Residuals at Time ' + str(time2) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonDensityResTime='+str(time2)+'.png')
MP.MakeFigure4(xList,np.array(d3, dtype = np.float32) - rho3,'Position (cm)', 'Density Residuals (grams per cubic centimeter)', 'Density Residuals at Time ' + str(time3) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonDensityResTime='+str(time3)+'.png')
MP.MakeFigure4(xList,np.array(d4, dtype = np.float32) - rho4,'Position (cm)', 'Density Residuals (grams per cubic centimeter)', 'Density Residuals at Time ' + str(time4) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonDensityResTime='+str(time4)+'.png')
MP.MakeFigure4(xList,np.array(d5, dtype = np.float32) - rho5,'Position (cm)', 'Density Residuals (grams per cubic centimeter)', 'Density Residuals at Time ' + str(time5) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonDensityResTime='+str(time5)+'.png')

MP.MakeFigure4(xList,np.array(v1, dtype = np.float32) - vel1,'Position (cm)', 'Velocity Residuals (centimeters per second)', 'Velocity Residuals at Time ' + str(time1) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonVelocityResTime='+str(time1)+'.png')
MP.MakeFigure4(xList,np.array(v2, dtype = np.float32) - vel2,'Position (cm)', 'Velocity Residuals (centimeters per second)', 'Velocity Residuals at Time ' + str(time2) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonVelocityResTime='+str(time2)+'.png')
MP.MakeFigure4(xList,np.array(v3, dtype = np.float32) - vel3,'Position (cm)', 'Velocity Residuals (centimeters per second)', 'Velocity Residuals at Time ' + str(time3) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonVelocityResTime='+str(time3)+'.png')
MP.MakeFigure4(xList,np.array(v4, dtype = np.float32) - vel4,'Position (cm)', 'Velocity Residuals (centimeters per second)', 'Velocity Residuals at Time ' + str(time4) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonVelocityResTime='+str(time4)+'.png')
MP.MakeFigure4(xList,np.array(v5, dtype = np.float32) - vel5,'Position (cm)', 'Velocity Residuals (centimeters per second)', 'Velocity Residuals at Time ' + str(time5) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonVelocityResTime='+str(time5)+'.png')

MP.MakeFigure4(xList,np.array(P1, dtype = np.float32) - press1,'Position (cm)', 'Pressure Residuals (ergs per cubic centimeter)', 'Pressure Residuals at Time ' + str(time1) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonPressureResTime='+str(time1)+'.png')
MP.MakeFigure4(xList,np.array(P2, dtype = np.float32) - press2,'Position (cm)', 'Pressure Residuals (ergs per cubic centimeter)', 'Pressure Residuals at Time ' + str(time2) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonPressureResTime='+str(time2)+'.png')
MP.MakeFigure4(xList,np.array(P3, dtype = np.float32) - press3,'Position (cm)', 'Pressure Residuals (ergs per cubic centimeter)', 'Pressure Residuals at Time ' + str(time3) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonPressureResTime='+str(time3)+'.png')
MP.MakeFigure4(xList,np.array(P4, dtype = np.float32) - press4,'Position (cm)', 'Pressure Residuals (ergs per cubic centimeter)', 'Pressure Residuals at Time ' + str(time4) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonPressureResTime='+str(time4)+'.png')
MP.MakeFigure4(xList,np.array(P5, dtype = np.float32) - press5,'Position (cm)', 'Pressure Residuals (ergs per cubic centimeter)', 'Pressure Residuals at Time ' + str(time5) + ' sec', ['Python Approximate - Fortran Exact'], 'FortranPythonPressureResTime='+str(time5)+'.png')



