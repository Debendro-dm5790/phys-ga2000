import numpy as np
import matplotlib.pyplot as plt

import Initialize as Init 
import Evolution as Evolv
import SaveData as Save
import MakePlots as MP



x, d1 = MP.ReadAndReturnData('DensityDataAtTime=0.08withSpatialGridSize=100.csv')
y, d2 = MP.ReadAndReturnData('DensityDataAtTime=0.08withSpatialGridSize=200.csv')
z, d3 = MP.ReadAndReturnData('DensityDataAtTime=0.08withSpatialGridSize=400.csv')

xlabel = 'Position (meters)'
ylabel = 'Density (Kg per Cubic Meter)'
title = 'Densities at Similar Times As Grid Size Varies'
saveName = 'DensityConverge.png'

MP.MakeFigure2(x, y, z, d1, d2, d3, xlabel, ylabel,title, ['0.08 sec and N = 100', '0.08 sec and N = 200', '0.08 sec and N = 400'], saveName)


x, d1 = MP.ReadAndReturnData('PressureDataAtTime=0.08withSpatialGridSize=100.csv')
y, d2 = MP.ReadAndReturnData('PressureDataAtTime=0.08withSpatialGridSize=200.csv')
z, d3 = MP.ReadAndReturnData('PressureDataAtTime=0.08withSpatialGridSize=400.csv')

xlabel = 'Position (meters)'
ylabel = 'Pressure (Pascals)'
title = 'Pressures at Similar Times As Grid Size Varies'
saveName = 'PressureConverge.png'

MP.MakeFigure2(x, y, z, d1, d2, d3, xlabel, ylabel,title, ['0.08 sec and N = 100', '0.08 sec and N = 200', '0.08 sec and N = 400'], saveName)

x, d1 = MP.ReadAndReturnData('VelocityDataAtTime=0.08withSpatialGridSize=100.csv')
y, d2 = MP.ReadAndReturnData('VelocityDataAtTime=0.08withSpatialGridSize=200.csv')
z, d3 = MP.ReadAndReturnData('VelocityDataAtTime=0.08withSpatialGridSize=400.csv')

xlabel = 'Position (meters)'
ylabel = 'Velocity (meters per second)'
title = 'Velocities at Similar Times As Grid Size Varies'
saveName = 'VelocityConverge.png'

MP.MakeFigure2(x, y, z, d1, d2, d3, xlabel, ylabel,title, ['0.08 sec and N = 100', '0.08 sec and N = 200', '0.08 sec and N = 400'], saveName)
