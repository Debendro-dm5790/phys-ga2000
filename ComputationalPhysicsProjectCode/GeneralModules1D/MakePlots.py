import numpy as np
import matplotlib.pyplot as plt
import csv

def ReadAndReturnData(filename):
    '''
    This functions reads and returns the data stored in the .csv file. The file contains 
    two columns, one for the independent variable and another for the dependent variable.

    Parameter:
    ------------
    filename:       string
                    The name of the .csv file that we want to read and extract data
                
    Returns:
    ----------
    arr1:           Python list of 32-bit floats
                    List containing the stored independent variable values.
    
    arr2:           Python list of 32-bit floats
                    List containing the stored dependent variable values.

    '''
    arr1 = []
    arr2 = []
    counter = 0
    
    with open(filename, mode = 'r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            if counter != 0 and counter != 1 and counter%2 == 0:
                arr1.append(np.float32(lines[0]))
                arr2.append(np.float32(lines[1]))
                
            counter += 1
                
    return arr1, arr2

def ReadAndReturnFortranData(filename):
    '''
    This function reads and returns position, density, pressure, and velocity data 
    along with the time from a .dat file generated from the Fortran code found in 
    https://cococubed.com/code_pages/exact_riemann.shtml. This code solves the Riemann problem
    exactly. 
    
    Parameters
    ----------
    filename :  string
                The name of the .csv file that we want to read and extract data

    Returns
    -------
    time :      32-bit floating point number
                The particular time
    
    x :         Numpy array of 32-bit floats
                Position grid
    
    rho :       Numpy array of 32-bit floats
                Array of densities over various positions
    
    press :     Numpy array of 32-bit floats
                Array of pressure over various positions
    
    vel :       Numpy array of 32-bit floats
                Array of velocities over various positions
                
    '''
    newLines = []
    
    with open(filename) as input_file:
        lines = input_file.readlines()
        for line in lines:
            newLine = line.strip().split()
            newLines.append( newLine )
      
    newLines = np.array(newLines)

    length = len(newLines) - 3

    time = np.float32(newLines[0][2])
    x = np.zeros(length, dtype = np.float32)
    rho = x.copy()
    press = x.copy()
    vel = x.copy()

    for i in range(3, len(newLines)):
        x[i-3] = float(newLines[i][1])
        rho[i-3] = float(newLines[i][2])
        press[i-3] = float(newLines[i][3])
        vel[i-3] = float(newLines[i][4])
        
    return time, x, rho, press, vel

def MakeFigure(arr1A, arr2A, arr2B, arr2C, arr2D, arr2E, xlabel, ylabel, title, legend, saveName):
    '''
    Function that makes and saves a plot of four kinds of dependent variables with a common
    independent variable. In other words, the size of the array of independent variables is 
    the same for all cases.  

    Parameters:
    -------------
    arr1A:          Python list or Numpy array of 32-bit floats
                    Python list or Numpy array containing the independent variable values

    arr2A:          Python list or Numpy array of 32-bit floats
                    Python list or Numpy array containing the stored dependent variable values.
                
    arr2B:          Python list or Numpy array of 32-bit floats
                    Python list or Numpy array containing the stored dependent variable values.
                
    arr2C:          Python list or Numpy array of 32-bit floats
                    Python list or Numpy array containing the stored dependent variable values.
                
    arr2D:          Python list or Numpy array of 32-bit floats
                    Python list or Numpy array containing the stored dependent variable values.
                
    arr2E:          Python list or Numpy array of 32-bit floats
                    Python list or Numpy array containing the stored dependent variable values.
                
    xlabel:         string
                    The independent variable label
                
    ylabel:         string
                    The dependent variable label
                
    title:          string
                    The plot title
                
    legend:         Python list of strings
                    The appropriate legend
                
    saveName:       String
                    The name of the .png file we want to save. 
'''
    
    plt.plot(arr1A , arr2A , '.')
    plt.plot(arr1A , arr2B , '.')
    plt.plot(arr1A , arr2C , '.')
    plt.plot(arr1A , arr2D , '.')
    plt.plot(arr1A , arr2E , '.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend)
    plt.savefig(saveName)
    plt.close()
    
def MakeFigure2(arr1A, arr1B, arr1C, arr2A, arr2B, arr2C, xlabel, ylabel, title, legend, saveName):
    '''
    Function that makes and saves a plot of three kinds of dependent variables with three different
    independent variable. The arrays of independent variables, however, have the same units (ex: position)

    Parameters
    ----------
    arr1A :         Numpy array or Python list of 32-bit floats
                    Python list or Numpy array of independent variable values
         
    arr1B :         Numpy array or Python list of 32-bit floats
                    Python list or Numpy array of independent variable values
        
    arr1C :         Numpy array or Python list of 32-bit floats
                    Python list or Numpy array of independent variable values
           
    arr2A :         Numpy array or Python list of 32-bit floats
                    Python list or Numpy array containing the stored dependent variable values.
                    
    arr2B :         Numpy array or Python list of 32-bit floats
                    Python list or Numpy array containing the stored dependent variable values.
                    
    arr2C :         Numpy array or Python list of 32-bit floats
                    Python list or Numpy array containing the stored dependent variable values.
    
    xlabel:         string
                    The independent variable label
                
    ylabel:         string
                    The dependent variable label
                
    title:          string
                    The plot title
                
    legend:         Python list of strings
                    The appropriate legend
                
    saveName:       String
                    The name of the .png file we want to save. 

    '''
    plt.plot(arr1A , arr2A , '.')
    plt.plot(arr1B , arr2B , '.')
    plt.plot(arr1C , arr2C , '.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend)
    plt.savefig(saveName)
    plt.close()
    
def MakeFigure3(array1, array2A, array2B, xlabel, ylabel, title, legend, saveName):
    '''
    A function that plots two arrays and/or lists on a single plot

    Parameters
    ----------
    array1 :        Numpy array or Python list of 32-bit floats
                    The independent variable. An example is position. 
                   
    array2A :       Numpy array or Python list of 32-bit floats
                    A dependent variable. 
                   
    array2B :       Numpy array or Python list of 32-bit floats
                    A dependent variable. 
                   
    xlabel:         string
                    The independent variable label
                
    ylabel:         string
                    The dependent variable label
                
    title:          string
                    The plot title
                
    legend:         Python list of strings
                    The appropriate legend
                
    saveName:       String
                    The name of the .png file we want to save. 

    '''
    plt.plot(array1, array2A, '.')
    plt.plot(array1, array2B, '.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend)
    plt.savefig(saveName)
    plt.close()
            
def MakeFigure4(array1, array2A, xlabel, ylabel, title, legend, saveName, ylimA, ylimB):
    '''
    A basic plotting function

    Parameters
    ----------
    array1 :        Numpy array or Python list of 32-bit floats
                    The independent variable. An example is position. 
                   
    array2A :       Numpy array or Python list of 32-bit floats
                    The dependent variable. 
                   
    xlabel:         string
                    The independent variable label
                
    ylabel:         string
                    The dependent variable label
                
    title:          string
                    The plot title
                
    legend:         Python list of strings
                    The appropriate legend
                
    saveName:       String
                    The name of the .png file we want to save. 
                    
    ylimA:          32-bit floating point number
                    The lower limit of the y-axis
                    
    ylimB:          32-bit floating point number
                    The upper limit of the y-axis

    '''
    plt.plot(array1, array2A, '.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend)
    plt.ylim(ylimA, ylimB)
    plt.savefig(saveName)
    plt.close()
    
