import numpy as np
import csv

'''
This function saves data as a .csv file. 

Parameters:
-------------

arr1:         numpy array of 32-but floating point numbers
              The first input data that will be placed in the first column in the 
              .csv file. Usually the independent variable such as spatial locations.
    
arr2:         numpy array of 32-but floating point numbers
              The second input data that will be placed in the second column in the 
              .csv file. Usually the dependent variable such as density, pressure, and velocity.
    
filename:     string
              The name of the .csv file we want to create and save. 
    
fields:       numpy array of strings
              length two array containing the names of the fields. Example: 'Position' and 'Pressure'
              The first (second) field name corresponds to the independent (dependent) variable.
'''

def SaveDataAsCSVFile(arr1, arr2, array2d, filename, fields):
    grid = np.zeros((len(arr1)+1,len(arr1)+1), dtype = np.float32)
    grid[0,0] = 0
    
    for i in range(len(arr1)):
        grid[i+1,0] = arr2[i] #first column labels
        grid[0,i+1] = arr1[i] #first row labels 
    
    grid[1:, 1:] = array2d 
        
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(grid)

