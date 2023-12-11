import numpy as np
import csv

def SaveDataAsCSVFile(arr1, arr2, filename, fields):
    '''
    This function saves data from arrays as a .csv file. 

    Parameters:
    -------------

    arr1:         numpy array of 32-bit floating point numbers
                  The first input data that will be placed in the first column in the 
                  .csv file. Usually the independent variable such as spatial locations.
    
    arr2:         numpy array of 32-bit floating point numbers
                  The second input data that will be placed in the second column in the 
                  .csv file. Usually the dependent variable such as density, pressure, and velocity.
    
    filename:     string
                  The name of the .csv file we want to create and save. 
    
    fields:       numpy array of strings
                  length two array containing the names of the fields. Example: 'Position' and 'Pressure'
                  The first (second) field name corresponds to the independent (dependent) variable.
    '''
    rows = np.zeros((len(arr1),2), dtype = np.float32)
    
    for i in range(len(arr1)):
        rows[i][0] = arr1[i]
        rows[i][1] = arr2[i]
        
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
        
def SaveDataAsCSVFile2D(arr1, arr2, array2d, filename, fields):
    '''
    This function saves data as a .csv file. 

    Parameters:
    -------------

    arr1:         numpy array of 32-bit floating point numbers
                  The independent variable such as x spatial locations.
    
    arr2:         numpy array of 32-bit floating point numbers
                  The independent variable such as y spatial locations.
              
    array2d:      numpy array of 32-bit floating point numbers
                  The values of a dependent variable such as density, x and y momentum densities,
                  and the energy
    
    filename:     string
                  The name of the .csv file we want to create and save. 
    
    fields:       numpy array of strings
                  length two array containing the names of the fields. Example: 'Position' and 'Pressure'
                  The first (second) field name corresponds to the independent (dependent) variable. For 
                  the two-dimensional problem, this array serves as an easy way to understand what data is 
                  saved in the file when the file is opened.
    '''
    grid = np.zeros((len(arr1)+1,len(arr1)+1), dtype = np.float32)
    grid[0,0] = 0
    
    for i in range(len(arr1)):
        '''
        The y-position values are stored in the first column and the x-position values
        are stored in the first row of the 2D array called grid
        '''
        grid[i+1,0] = arr2[i] 
        grid[0,i+1] = arr1[i] 
    
    grid[1:, 1:] = array2d 
    '''
    The grid of data will look like the following
    x,y = 0     x1      x2       x3        x4        x5      x6 .....
    
    y1   data   data   data     data     data      data     data ....
    
    
    y2   data   data   data     data     data      data     data ....
    
    
    y3   data   data   data     data     data      data     data ....
    
    
    y4   data   data   data     data     data      data     data ....
    
    
    y5   data   data   data     data     data      data     data ....
    .
    .
    .
    
    '''
        
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(grid)
        
