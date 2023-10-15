import numpy as np

def speedOfSound(index, gamma, u1, u2, u3):
    '''
    A function that computes and returns the speed of sound
    
    Parameters
    ----------
    index :             32-bit integer
                        The index of the u1, u2, u3 array. Needed to compute the sound speed
                    
    gamma :             32-bit floating point number
                        Adiabatic constant 
                        
    u1:                 numpy array of 32-bit floats
                        numpy array of mass densities
                        
    u2:                 numpy array of 32-bit floats
                        numpy array of momentum densities
                        
    u3:                 numpy array of 32-bit floats
                        numpy array of energy densities

    Returns
    -------
    The expression for the speed of sound

    '''
    return np.sqrt(gamma*(gamma - 1)*((u3[index]/u1[index]) - 0.5*(u2[index]**2/u1[index]**2)))

def computeTimeStepandFHLL(deltaX, gamma, numInterfacePoints, n, u1, u2, u3):
    '''
    A function that computes the appropriate time step that satisfies the 
    Courant-Friedrich-Levy condition and the approximate fluxes at the interfaces.
     
    Parameters
    -----------
    deltaX:             32-bit floating point number
                        The spatial grid spacing
                        
    gamma:              32-bit floating point number
                        The adiabatic index
                        
    numInterfacePoints: 32-bit integer
                        the number of interfaces in the spatial grid of cells.
                        
    n:                  32-bit integer
                        how much one wants to divide the minimum time step by
                        
    u1:                 numpy array of 32-bit floats
                        numpy array of mass densities
                        
    u2:                 numpy array of 32-bit floats
                        numpy array of momentum densities
                        
    u3:                 numpy array of 32-bit floats
                        numpy array of energy densities
    
    Returns
    --------
    timeStepCandidate:        32-bit floating point number
                              A candidate for the time step that satisfies 
                              Courant-Friedrich-Levy condition. It is one-n-th of the 
                              minimum of all such times that each satisfy the 
                              Courant-Friedrich-Levy condition
                              
    F1_HLL:                   numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the mass density.
                              Calculated with the Harten-Lax-van Leer approximation.
                              
    F2_HLL:                   numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the momentum density.
                              Calculated with the Harten-Lax-van Leer approximation.
    
    F3_HLL:                   numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the energy density.
                              Calculated with the Harten-Lax-van Leer approximation.                          
    '''
    timeStepCandidate = np.float32(0)
    
    F1 = u2
    F2 = ((gamma - 1)*u3) - (0.5*(gamma - 3)*(u2**2/u1))
    F3 = (gamma*((u2*u3)/(u1))) - 0.5*((gamma - 1)*(u2**3/u1**2))
    
    F1_HLL = np.zeros(numInterfacePoints, dtype = np.float32)
    F2_HLL = F1_HLL.copy()
    F3_HLL = F1_HLL.copy()
    
    for i in range(numInterfacePoints):
        lambdaPlusLeft = (u2[i]/u1[i]) + speedOfSound(i, gamma, u1, u2, u3)
        lambdaMinusLeft = (u2[i]/u1[i]) - speedOfSound(i, gamma, u1, u2, u3)
        lambdaPlusRight = (u2[i+1]/u1[i+1]) + speedOfSound(i+1, gamma, u1, u2, u3)
        lambdaMinusRight = (u2[i+1]/u1[i+1]) - speedOfSound(i+1, gamma, u1, u2, u3)
        
        alphaPlus = max([0, lambdaPlusLeft, lambdaPlusRight])
        alphaMinus = max([0, -1*lambdaMinusLeft, -1*lambdaMinusRight])
        
        possibleTimeStep = np.float32(deltaX/max(alphaPlus, alphaMinus))
        
        if i == 0:
            timeStepCandidate = possibleTimeStep
        else: 
            if possibleTimeStep < timeStepCandidate:
                timeStepCandidate = possibleTimeStep

        F1_HLL[i] = ((alphaPlus*F1[i]) + (alphaMinus*F1[i+1]) - (alphaPlus*alphaMinus*(u1[i+1] - u1[i])))/(alphaPlus + alphaMinus)
        F2_HLL[i] = ((alphaPlus*F2[i]) + (alphaMinus*F2[i+1]) - (alphaPlus*alphaMinus*(u2[i+1] - u2[i])))/(alphaPlus + alphaMinus)
        F3_HLL[i] = ((alphaPlus*F3[i]) + (alphaMinus*F3[i+1]) - (alphaPlus*alphaMinus*(u3[i+1] - u3[i])))/(alphaPlus + alphaMinus)
        
    timeStepCandidate = timeStepCandidate/n
    
    return timeStepCandidate, F1_HLL, F2_HLL, F3_HLL

def computeFluxDivergence(deltaX, numPoints, F1_HLL, F2_HLL, F3_HLL):
    '''
    Parameters
    ----------
    deltaX :                  32-bit floating point number
                              The spatial grid spacing
       
    numPoints :               32-bit integer
                              The number points in the spatial grid.
    
    F1_HLL:                   numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the mass density.
                              Calculated with the Harten-Lax-van Leer approximation.
                              
    F2_HLL:                   numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the momentum density.
                              Calculated with the Harten-Lax-van Leer approximation.
    
    F3_HLL:                   numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the energy density.
                              Calculated with the Harten-Lax-van Leer approximation.  

    Returns
    -------
    DivF1 :                   numpy array of 32-bit floating point numbers
                              The change in interface flux (left interface - right interface)
                              associated with the mass density divided by the 
                              spatial grid spacing
                              
    DivF2 :                   numpy array of 32-bit floating point numbers
                              The change in interface flux (left interface - right interface)
                              associated with the momentum density divided by the 
                              spatial grid spacing
        
    DivF3 :                   numpy array of 32-bit floating point numbers
                              The change in interface flux (left interface - right interface)
                              associated with the energy density divided by the 
                              spatial grid spacing

    '''
    DivF1 = np.zeros(numPoints, dtype = np.float32)
    DivF2 = DivF1.copy()
    DivF3 = DivF1.copy()
    
    for i in range(numPoints):
        DivF1[i] = -1*(F1_HLL[i+1] - F1_HLL[i])/deltaX
        DivF2[i] = -1*(F2_HLL[i+1] - F2_HLL[i])/deltaX
        DivF3[i] = -1*(F3_HLL[i+1] - F3_HLL[i])/deltaX
        
    return DivF1, DivF2, DivF3
        
def updateQuantities(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3):
    '''
    
    Parameters
    ----------
    timeStep :                   32-bit floating point number
                                 An appropriate timestep that does not violate the 
                                 Courant-Friedrich-Levy condition.
                                 
    numPoints :                  32-bit integer
                                 The number of points in the spatial grid.
   
    u1:                          numpy array of 32-bit floats
                                 numpy array of mass densities
                        
    u2:                          numpy array of 32-bit floats
                                 numpy array of momentum densities
                        
    u3:                          numpy array of 32-bit floats
                                 numpy array of energy densities
   
    DivF1 :                      numpy array of 32-bit floating point numbers
                                 The change in interface flux (left interface - right interface)
                                 associated with the mass density divided by the 
                                 spatial grid spacing
                              
    DivF2 :                      numpy array of 32-bit floating point numbers
                                 The change in interface flux (left interface - right interface)
                                 associated with the momentum density divided by the 
                                 spatial grid spacing
        
    DivF3 :                      numpy array of 32-bit floating point numbers
                                 The change in interface flux (left interface - right interface)
                                 associated with the energy density divided by the 
                                 spatial grid spacing

    Returns
    -------
    u1:                          numpy array of 32-bit floats
                                 numpy array of updated mass densities
                        
    u2:                          numpy array of 32-bit floats
                                 numpy array of updated momentum densities
                        
    u3:                          numpy array of 32-bit floats
                                 numpy array of updated energy densities

    '''
    for i in range(1, numPoints + 1):
        u1[i] = u1[i] + timeStep*DivF1[i-1]
        u2[i] = u2[i] + timeStep*DivF2[i-1]
        u3[i] = u3[i] + timeStep*DivF3[i-1]
        
    u1[0] = u1[1]
    u2[0] = u2[1]
    u3[0] = u3[1]
    
    u1[numPoints + 1] = u1[numPoints]
    u2[numPoints + 1] = u2[numPoints]
    u3[numPoints + 1] = u3[numPoints]
        
    return u1, u2, u3
