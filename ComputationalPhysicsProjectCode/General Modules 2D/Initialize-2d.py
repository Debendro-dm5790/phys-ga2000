import numpy as np

def createSpaceTimeGrid(length, numPoints, TimePoints, spaceDim): #assuming dx=dy and lengthx=lengthy
    '''
    A function that generates the space time grid. 
    
    Parameters:
    -----------
    length:      32-bit floating point number
                 the total distance spanned by the spatial grid
                 spatial grid will have values between 0 and length
             
    numPoints:   (even) integer
                 the total number of points in each (x and possibly y) spatial grid
                 evenness will be checked and an exception could be raised
               
    TimePoints:  integer 
                 the total number of points in the temporal grid
               
    spaceDim:    integer
                 the total number of spatial grids. We will demand that this is 1 or 2
    
    Returns:
    -----------
    x:           numpy array of single-precision floats. 
                 The x-direction spatial grid. Size is numPoints.
               
    y:           numpy array of single-precision floats or an empty list. 
                 The y-direction spatial grid. Size is 0 or numPoints
               
    t:           numpy array of single-precision floating-point zeros. 
                 Size is numPoints
    
    Comments:
    -----------
    The spacial coordinates of this grid will be known, but the temporal 
    coordinate values will be left unknown and will be developed by future
    functions that determine the appropriate times using the Courant-Friedrich-Levy
    condition. Thus the temporal coordinates will not be evenly spaced
    
    '''
    if numPoints%2 == 1:
        raise Exception('Number of spacial and temporal points should be even.')   
    if spaceDim == 2:
        x = np.linspace(0, length, numPoints, dtype = np.float32)
        y = np.linspace(0, length, numPoints, dtype = np.float32)
        t = np.zeros(TimePoints, dtype = np.float32)
        return x,y,t
    elif spaceDim == 1:
        x = np.linspace(0, length, numPoints, dtype = np.float32)
        y = []
        t = np.zeros(TimePoints, dtype = np.float32)
        return x,y,t
    else:
        raise Exception("Spacial dimensions must be an positive integer less than 3.")
   
   
def setInitialVals(task, g, numPoints, v0x, v0y, rhoL, rhoR, PressL, PressR): #shock tube in x-direction
    '''
    A function that sets and returns the initial values of constants and system parameters
    
    Parameters
    ----------
    task :      32-bit integer
                The number of the problem with specified initial and boundry conditions
     
    g:          (ideally) 32-bit floating point number
                This is the adiabatic constant of the system
                
    numPoints:  (even) integer
                the total number of points in each (u1, u2, u3, u4) parameter grid
                evenness will be checked and an exception could be raised
                SHOULD BE 4 MORE than the total number of points in each spatial grid
                direction in order to include the boundary for the TWO-DIMENSIONAL PROBLEM 
                
    v0x:        (ideally) 32-bit floating point number
                The initial x-velocity of the system. 
    
    v0y:        (ideally) 32-bit floating point number
                The initial y-velocity of the system. 
                
    rhoL:       (ideally) 32-bit floating point number
                The initial density of the left half of the system
                
    rhoR :      (ideally) 32-bit floating point number
                The initial density of the right half of the system
                
    PressL :    (ideally) 32-bit floating point number
                The initial pressure of the left half of the system
                
    PressR :    (ideally) 32-bit floating point number
                The initial pressure of the right half of the system
    
    Returns
    -------
    u1 :                    numpy array of 32-bit floats
                            Array of initial mass density values
                            
    u2 :                    numpy array of 32-bit floats
                            Array of initial x-momentum density values
    
    u3 :                    numpy array of 32-bit floats
                            Array of initial y-momentum density values
                            
    u4 :                    numpy array of 32-bit floats
                            Array of initial energy density values
    
    Comment
    --------
    We note that each of the returns is a two-dimensional array whose columns are associated
    with the x direction and whose rows are associated with the y direction as shown below. We 
    note that only the sections labelled as 'data' correspond to the elements of each returned array.
    
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
    if numPoints%2 == 1:
        raise Exception('Number of spacial and temporal points should be even.')
        
    gamma = np.float32(g)
    
    u1 = np.zeros((numPoints, numPoints), dtype = np.float32)
    u2 = u1.copy()
    u3 = u1.copy()
    u4 = u1.copy()
    
    u1[:, 0:np.int32((numPoints/2))] = rhoL
    u1[:, np.int32(numPoints/2): numPoints] = rhoR
    
    if(task == 3): #for task 3 only (high density around the center)
		#rectangle
        #u1[:,:] = rhoL
        #u1[np.int32((numPoints/2 - numPoints/20)):np.int32((numPoints/2 + numPoints/20))+1, \
        #np.int32((numPoints/2 - numPoints/20)):np.int32((numPoints/2 + numPoints/20))+1] = rhoR #the high rho
        
        #circle
        center = (numPoints // 2, numPoints // 2)
        radius = numPoints // 20
        xx, yy = np.meshgrid(np.arange(numPoints), np.arange(numPoints))
        circle_mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 < radius ** 2
        u1[:,:] = rhoL
        u1[circle_mask] = rhoR
    
    u2 = v0x*u1
    u3 = v0y*u1
    
    if(task == 2): #for task 2 only (different Vx at top and bottom)
        u2[0:np.int32((numPoints/2)), :] = v0x
        u2[np.int32(numPoints/2): numPoints, :] = -v0x
        u2 = u2*u1
    
    u4[:, 0:np.int32((numPoints/2))] = np.float32(PressL/(gamma - 1))
    u4[:, np.int32(numPoints/2): numPoints] = np.float32(PressR/(gamma - 1))
    
    u4 = u4 + (u2**2/(2*u1)) + (u3**2/(2*u1))
    
    return u1, u2, u3, u4

