import numpy as np
import matplotlib.pyplot as plt
import csv

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
def ReadAndReturnData(filename):
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

'''
Function that makes and saves a plot of four kinds of dependent variables. 

Parameters:
-------------
arr1A:          numpy array of 32-bit floats
                numpy array of independent variable values

arr2A:          Python list of 32-bit floats
                List containing the stored dependent variable values.
                
arr2B:          Python list of 32-bit floats
                List containing the stored dependent variable values.
                
arr2C:          Python list of 32-bit floats
                List containing the stored dependent variable values.
                
arr2D:          Python list of 32-bit floats
                List containing the stored dependent variable values.
                
arr2E:          Python list of 32-bit floats
                List containing the stored dependent variable values.
                
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

def MakeFigure(arr1A, arr2A, arr2B, arr2C, arr2D, arr2E, xlabel, ylabel, title, legend, saveName):
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
    plt.plot(arr1A , arr2A , '.')
    plt.plot(arr1B , arr2B , '.')
    plt.plot(arr1C , arr2C , '.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend)
    plt.savefig(saveName)
    plt.close()
            
        

