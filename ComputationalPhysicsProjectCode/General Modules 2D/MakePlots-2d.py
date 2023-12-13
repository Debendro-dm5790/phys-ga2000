import numpy as np
import matplotlib.pyplot as plt
import csv

'''
This functions reads and returns the data stored in the .csv file. 

Parameter:
------------
filename:       string
                The name of the .csv file that we want to read and extract data
                
Returns:
----------
X:              Python list of 32-bit floats
                List containing the stored x variable values.
    
Y:              Python list of 32-bit floats
                List containing the stored y variable values.
                
Z:              Python list of 32-bit floats
                List containing the stored z variable values.              

'''
def ReadAndReturnData(filename):
    arr1 = []
    arr2 = []
    counter = 0
    
    with open(filename, mode = 'r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            if counter != 0 and counter != 1 and counter%2 == 0:
                for i in range(len(lines)):
                    arr1.append(np.float32(lines[i]))
                arr2.append(arr1)
                arr1 = []
            counter += 1
    nparr = np.array([np.array(xi) for xi in arr2]) 
    x = nparr[0,1:]
    y = nparr[1:,0]
    X, Y = np.meshgrid(x, y)
    Z = nparr[1:,1:]           
    return X, Y, Z

def MakeFigure(arr1A, arr2A, arr1B, arr2B, arr3B, arr4B, arr5B, xlabel, ylabel, zlabel, title, legend, saveName):
    '''
    Function that makes and saves a plot of five kinds of dependent variables with a common set
    of bipartite independent variables (ex: x and y). 

    Parameters:
    -------------
    arr1A:          Python list of 32-bit floats
                    List containing the x variable values

    arr2A:          Python list of 32-bit floats
                    List containing the stored y variable values.
                
    arr1B:          Python list of 32-bit floats
                    List containing the stored dependent variable values.
                
    arr2B:          Python list of 32-bit floats
                    List containing the stored dependent variable values.
                
    arr3B:          Python list of 32-bit floats
                    List containing the stored dependent variable values.
                
    arr4B:          Python list of 32-bit floats
                    List containing the stored dependent variable values.
                    
    arr5B:          Python list of 32-bit floats
                    List containing the stored dependent variable values.                
                
    xlabel:         string
                    The x variable label
                
    ylabel:         string
                    The y variable label
                    
    zlabel:         string
                    The z variable label                
                
    title:          string
                    The plot title
                
    legend:         Python list of strings
                    The appropriate legend
                
    saveName:       String
                    The name of the .png file we want to save. 
'''

    ax = plt.axes(projection='3d')
    ax.plot_surface(arr1A, arr2A, arr1B, label=legend[0])
    ax.plot_surface(arr1A, arr2A, arr2B, label=legend[1])
    ax.plot_surface(arr1A, arr2A, arr3B, label=legend[2])
    ax.plot_surface(arr1A, arr2A, arr4B, label=legend[3])
    ax.plot_surface(arr1A, arr2A, arr5B, label=legend[4])
    ax.set_xlabel(xlabel, labelpad=20)
    ax.set_ylabel(ylabel, labelpad=20)
    ax.set_zlabel(zlabel, labelpad=20)
    plt.title(title)
    #ax.legend()
    plt.savefig(saveName)
    plt.show()
    #plt.close()

def MakeFigure1(arr1A, arr2A, arr1B, xlabel, ylabel, zlabel, title, legend, saveName):
    '''
    Function that makes and saves a plot of a dependent variables with two
    independent variable. 

    Parameters:
    -------------
    arr1A:          Python list of 32-bit floats
                    List containing the x variable values

    arr2A:          Python list of 32-bit floats
                    List containing the stored y variable values.
                
    arr1B:          Python list of 32-bit floats
                    List containing the stored dependent variable values.               
                
    xlabel:         string
                    The x variable label
                
    ylabel:         string
                    The y variable label
                    
    zlabel:         string
                    The z variable label                
                
    title:          string
                    The plot title
                
    legend:         Python list of strings
                    The appropriate legend
                
    saveName:       String
                    The name of the .png file we want to save. 
'''

    ax = plt.axes(projection='3d')
    ax.plot_surface(arr1A, arr2A, arr1B, label=legend[0])
    ax.set_xlabel(xlabel, labelpad=20)
    ax.set_ylabel(ylabel, labelpad=20)
    ax.set_zlabel(zlabel, labelpad=20)
    plt.title(title)
    #ax.legend()
    plt.savefig('onefig'+saveName)
    plt.show()

