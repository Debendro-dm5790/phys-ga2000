import numpy as np
import math

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

def minmod(x,y,z):
    '''
    This function returns the minmod of three input parameters x,y and z. The minmod 
    is defined as 0.25 |sgn(x) + sgn(y)| * (sgn(x) + sgn(z)) * min(|x|, |y|, |z|)

    Parameters
    ----------
    x :       32-bit floating-point number
              An input parameter
        
    y :       32-bit floating-point number
              An input parameter
        
    z :       32-bit floating-point number
              An input parameter

    Returns
    -------
    The minmod = 0.25 |sgn(x) + sgn(y)| * (sgn(x) + sgn(z)) * min(|x|, |y|, |z|)

    '''
    sgn_x = 0
    sgn_y = 0
    sgn_z = 0
    
    if x < 0:
        sgn_x = -1
    elif x > 0:
        sgn_x = 1
    else:
        sgn_x = 0
        
    if y < 0:
        sgn_y = -1
    elif y > 0:
        sgn_y = 1
    else:
        sgn_y = 0
        
    if z < 0:
        sgn_z = -1
    elif z > 0:
        sgn_z = 1
    else:
        sgn_z = 0
        
    min_xyz = min([np.abs(x), np.abs(y), np.abs(z)])
    
    return 0.25*np.abs(sgn_x + sgn_y)*(sgn_x + sgn_z)*min_xyz

def computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numInterfacePoints, n, u1, u2, u3):
    '''

    Parameters
    ----------
    deltaX:                 32-bit floating point number
                            The spatial grid spacing
                        
    gamma:                  32-bit floating point number
                            The adiabatic index
    
    theta :                 32-bit floating point number
                            A parameter between 1 and 2
    
    numInterfacePoints:     32-bit integer
                            the number of interfaces in the spatial grid of cells.
        
    n:                      32-bit integer
                            how much one wants to divide the minimum time step by
                        
    u1:                     numpy array of 32-bit floats
                            numpy array of mass densities
                        
    u2:                     numpy array of 32-bit floats
                            numpy array of momentum densities
                        
    u3:                     numpy array of 32-bit floats
                            numpy array of energy densities

    Returns
    -------
    timeStepCandidate:      32-bit floating point number
                            A candidate for the time step that satisfies 
                            Courant-Friedrich-Levy condition. It is one-n-th of the 
                            minimum of all such times that each satisfy the 
                            Courant-Friedrich-Levy condition
    
    F1_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the mass density.
                              Calculated with the Harten-Lax-van Leer approximation, but
                              the left and right fluxes were calculated with the 3rd order
                              Runge-Kutta method.
                              
    F2_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the momentum density.
                              Calculated with the Harten-Lax-van Leer approximation, but
                              the left and right fluxes were calculated with the 3rd order
                              Runge-Kutta method.
    
    F3_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the energy density.
                              Calculated with the Harten-Lax-van Leer approximation, but
                              the left and right fluxes were calculated with the 3rd order
                              Runge-Kutta method.

    '''
    C1 = u1
    C2 = u2/u1
    C3 = (gamma - 1)*(u3 - 0.5*(u2**2/u1))
    
    LeftOfInterfaceRho_C1 = np.zeros(numInterfacePoints, dtype = np.float32)
    LeftOfInterfaceVel_C2 = LeftOfInterfaceRho_C1.copy()
    LeftOfInterfacePress_C3 = LeftOfInterfaceRho_C1.copy()
    RightOfInterfaceRho_C1 = LeftOfInterfaceRho_C1.copy()
    RightOfInterfaceVel_C2 = LeftOfInterfaceRho_C1.copy()
    RightOfInterfacePress_C3 = LeftOfInterfaceRho_C1.copy()
        
    for i in range(numInterfacePoints):
        LeftOfInterfaceRho_C1[i] = C1[i+1] + 0.5*minmod(theta*(C1[i+1] - C1[i]), 0.5*(C1[i+2] - C1[i]), theta*(C1[i+2] - C1[i+1]))
        LeftOfInterfaceVel_C2[i] = C2[i+1] + 0.5*minmod(theta*(C2[i+1] - C2[i]), 0.5*(C2[i+2] - C2[i]), theta*(C2[i+2] - C2[i+1]))
        LeftOfInterfacePress_C3[i] = C3[i+1] + 0.5*minmod(theta*(C3[i+1] - C3[i]), 0.5*(C3[i+2] - C3[i]), theta*(C3[i+2] - C3[i+1]))
        
        RightOfInterfaceRho_C1[i] = C1[i+2] + 0.5*minmod(theta*(C1[i+2] - C1[i+1]), 0.5*(C1[i+3] - C1[i+1]), theta*(C1[i+3] - C1[i+2]))
        RightOfInterfaceVel_C2[i] = C2[i+2] + 0.5*minmod(theta*(C2[i+2] - C2[i+1]), 0.5*(C2[i+3] - C2[i+1]), theta*(C2[i+3] - C2[i+2]))
        RightOfInterfacePress_C3[i] = C3[i+2] + 0.5*minmod(theta*(C3[i+2] - C3[i+1]), 0.5*(C3[i+3] - C3[i+1]), theta*(C3[i+3] - C3[i+2]))
    
    LeftOfInterfaceFlux_1 = LeftOfInterfaceRho_C1*LeftOfInterfaceVel_C2
    LeftOfInterfaceFlux_2 = LeftOfInterfaceRho_C1*LeftOfInterfaceVel_C2**2 + LeftOfInterfacePress_C3
    LeftOfInterfaceFlux_3 = 0.5*LeftOfInterfaceRho_C1*LeftOfInterfaceVel_C2**3 + (gamma/(gamma - 1))*(LeftOfInterfaceVel_C2*LeftOfInterfacePress_C3)
    
    RightOfInterfaceFlux_1 = RightOfInterfaceRho_C1*RightOfInterfaceVel_C2
    RightOfInterfaceFlux_2 = RightOfInterfaceRho_C1*RightOfInterfaceVel_C2**2 + RightOfInterfacePress_C3
    RightOfInterfaceFlux_3 = 0.5*RightOfInterfaceRho_C1*RightOfInterfaceVel_C2**3 + (gamma/(gamma - 1))*(RightOfInterfaceVel_C2*RightOfInterfacePress_C3)
    
    F1_HLL_RungeKutta = np.zeros(numInterfacePoints, dtype = np.float32)
    F2_HLL_RungeKutta = F1_HLL_RungeKutta.copy()
    F3_HLL_RungeKutta = F1_HLL_RungeKutta.copy()
    errorIndex = []
    
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
                
        F1_HLL_RungeKutta[i] = ((alphaPlus*LeftOfInterfaceFlux_1[i]) + (alphaMinus*RightOfInterfaceFlux_1[i]) - (alphaPlus*alphaMinus*(u1[i+2] - u1[i+1])))/(alphaPlus + alphaMinus)
        F2_HLL_RungeKutta[i] = ((alphaPlus*LeftOfInterfaceFlux_2[i]) + (alphaMinus*RightOfInterfaceFlux_2[i]) - (alphaPlus*alphaMinus*(u2[i+2] - u2[i+1])))/(alphaPlus + alphaMinus)
        F3_HLL_RungeKutta[i] = ((alphaPlus*LeftOfInterfaceFlux_3[i]) + (alphaMinus*RightOfInterfaceFlux_3[i]) - (alphaPlus*alphaMinus*(u3[i+2] - u3[i+1])))/(alphaPlus + alphaMinus)
    
    timeStepCandidate = timeStepCandidate/n
    
    return timeStepCandidate, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta

def computeFluxDivergenceRungeKutta(deltaX, numPoints, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta):
    '''
    Parameters
    ----------
    deltaX :                  32-bit floating point number
                              The spatial grid spacing
       
    numPoints :               32-bit integer
                              The number points in the spatial grid.
    
    F1_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the mass density.
                              Calculated with the Harten-Lax-van Leer approximation, but
                              the left and right fluxes were calculated with the 3rd order
                              Runge-Kutta method.
                              
    F2_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the momentum density.
                              Calculated with the Harten-Lax-van Leer approximation, but
                              the left and right fluxes were calculated with the 3rd order
                              Runge-Kutta method.
    
    F3_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the energy density.
                              Calculated with the Harten-Lax-van Leer approximation, but
                              the left and right fluxes were calculated with the 3rd order
                              Runge-Kutta method.

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
        DivF1[i] = -1*(F1_HLL_RungeKutta[i+1] - F1_HLL_RungeKutta[i])/deltaX
        DivF2[i] = -1*(F2_HLL_RungeKutta[i+1] - F2_HLL_RungeKutta[i])/deltaX
        DivF3[i] = -1*(F3_HLL_RungeKutta[i+1] - F3_HLL_RungeKutta[i])/deltaX
        
    return DivF1, DivF2, DivF3

def updateQuantitiesRungeKutta(timeStep, numPoints, u1, u2, u3, DivF1, DivF2, DivF3):
    '''
    Parameters
    ----------
    timeStep :              32-bit floating point number
                            An appropriate timestep that does not violate the Courant-Friedrich-Levy condition.
                            
    numPoints :             32-bit integer
                            The number of points in the spatial grid
        
    u1:                     numpy array of 32-bit floats
                            numpy array of mass densities
                        
    u2:                     numpy array of 32-bit floats
                            numpy array of momentum densities
                        
    u3:                     numpy array of 32-bit floats
                            numpy array of energy densities
                            
    DivF1 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the mass density divided by the 
                            spatial grid spacing
                              
    DivF2 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the momentum density divided by the 
                            spatial grid spacing
        
    DivF3 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the energy density divided by the 
                            spatial grid spacing

    Returns
    -------
    u1:                     numpy array of 32-bit floats
                            numpy array of updated mass densities
                        
    u2:                     numpy array of 32-bit floats
                            numpy array of updated momentum densities
                        
    u3:                     numpy array of 32-bit floats
                            numpy array of updated energy densities

    '''
    u1_1 = np.zeros(len(range(2, numPoints + 2)), dtype = np.float32)
    u1_2 = u1_1.copy()
    u2_1 = u1_1.copy()
    u2_2 = u1_1.copy()
    u3_1 = u1_1.copy()
    u3_2 = u1_1.copy()
    
    u1_1 = u1[2: numPoints + 2] + timeStep*DivF1
    u1_2 = 0.75*u1[2: numPoints + 2] + 0.25*u1_1 + 0.25*timeStep*DivF1
    u1[2: numPoints + 2] = (1/3)*u1[2: numPoints + 2] + (2/3)*u1_2 + (2/3)*timeStep*DivF1
    
    u2_1 = u2[2: numPoints + 2] + timeStep*DivF2
    u2_2 = 0.75*u2[2: numPoints + 2] + 0.25*u2_1 + 0.25*timeStep*DivF2
    u2[2: numPoints + 2] = (1/3)*u2[2: numPoints + 2] + (2/3)*u2_2 + (2/3)*timeStep*DivF2
    
    u3_1 = u3[2: numPoints + 2] + timeStep*DivF3
    u3_2 = 0.75*u3[2: numPoints + 2] + 0.25*u3_1 + 0.25*timeStep*DivF3
    u3[2: numPoints + 2] = (1/3)*u3[2: numPoints + 2] + (2/3)*u3_2 + (2/3)*timeStep*DivF3
        
    u1[1] = u1[2]
    u2[1] = u2[2]
    u3[1] = u3[2]
    
    u1[0] = u1[2]
    u2[0] = u2[2]
    u3[0] = u3[2]
    
    for i in range(numPoints + 2, len(u1)):
        u1[i] = u1[numPoints + 1]
        u2[i] = u2[numPoints + 1]
        u3[i] = u3[numPoints + 1]
        
    return u1, u2, u3
    
    