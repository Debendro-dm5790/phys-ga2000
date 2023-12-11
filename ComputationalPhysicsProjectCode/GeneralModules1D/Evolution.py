import numpy as np
import math

def speedOfSound(gamma, u1, u2, u3, start, end):
    '''
    A function that computes and returns the speed of sound
    
    Parameters
    ----------
    gamma :             32-bit floating point number
                        Adiabatic constant 
                        
    u1:                 numpy array of 32-bit floats
                        numpy array of mass densities
                        
    u2:                 numpy array of 32-bit floats
                        numpy array of momentum densities
                        
    u3:                 numpy array of 32-bit floats
                        numpy array of energy densities
                        
    start:              32-bit integer
                        The starting index of the arrays used to compute the sound speed.
                        
    end:                32-bit integer
                        The (ending index + 1) of the arrays used to compute the sound speed.

    Returns
    -------
    An array of sound speeds.

    '''
    return np.sqrt(gamma*(gamma - 1)*((u3[start:end]/u1[start:end]) - 0.5*(u2[start:end]**2/u1[start:end]**2)))

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
    
    lambdaPlusLeft = (u2[0:numInterfacePoints]/u1[0:numInterfacePoints]) + speedOfSound(gamma, u1, u2, u3, 0, numInterfacePoints)
    lambdaMinusLeft = (u2[0:numInterfacePoints]/u1[0:numInterfacePoints]) - speedOfSound(gamma, u1, u2, u3, 0, numInterfacePoints)
    lambdaPlusRight = (u2[1:numInterfacePoints+1]/u1[1:numInterfacePoints+1]) + speedOfSound(gamma, u1, u2, u3, 1, numInterfacePoints+1)
    lambdaMinusRight = (u2[1:numInterfacePoints+1]/u1[1:numInterfacePoints+1]) - speedOfSound(gamma, u1, u2, u3, 1, numInterfacePoints+1)
    
    zeros = np.zeros(len(lambdaMinusLeft), dtype = np.float32)
    
    alphaPlus = np.maximum(np.maximum(zeros, lambdaPlusLeft), lambdaPlusRight)
    alphaMinus = np.maximum(np.maximum(zeros, -1*lambdaMinusLeft), -1*lambdaMinusRight)
    
    possibleTimeStep = np.float32(deltaX/np.maximum(alphaPlus, alphaMinus))
    timeStepCandidate = np.min(possibleTimeStep)/n
    
    F1_HLL = ((alphaPlus*F1[0:numInterfacePoints]) + (alphaMinus*F1[1:numInterfacePoints+1]) - (alphaPlus*alphaMinus*(u1[1:numInterfacePoints+1] - u1[0:numInterfacePoints])))/(alphaPlus + alphaMinus)
    F2_HLL = ((alphaPlus*F2[0:numInterfacePoints]) + (alphaMinus*F2[1:numInterfacePoints+1]) - (alphaPlus*alphaMinus*(u2[1:numInterfacePoints+1] - u2[0:numInterfacePoints])))/(alphaPlus + alphaMinus)
    F3_HLL = ((alphaPlus*F3[0:numInterfacePoints]) + (alphaMinus*F3[1:numInterfacePoints+1]) - (alphaPlus*alphaMinus*(u3[1:numInterfacePoints+1] - u3[0:numInterfacePoints])))/(alphaPlus + alphaMinus)
        
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
    
    DivF1 = -1*(F1_HLL[1:numPoints+1] - F1_HLL[0:numPoints])/deltaX
    DivF2 = -1*(F2_HLL[1:numPoints+1] - F2_HLL[0:numPoints])/deltaX
    DivF3 = -1*(F3_HLL[1:numPoints+1] - F3_HLL[0:numPoints])/deltaX
    
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
    
    u1[1:numPoints + 1] = u1[1:numPoints + 1] + timeStep*DivF1[0:numPoints]
    u2[1:numPoints + 1] = u2[1:numPoints + 1] + timeStep*DivF2[0:numPoints]
    u3[1:numPoints + 1] = u3[1:numPoints + 1] + timeStep*DivF3[0:numPoints]
        
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
    x :       Numpy Array of 32-bit floating-point numbers
              An input parameter
        
    y :       Numpy Array of 32-bit floating-point numbers
              An input parameter
        
    z :       Numpy Array of 32-bit floating-point numbers
              An input parameter

    Returns
    -------
    The minmod = 0.25 |sgn(x) + sgn(y)| * (sgn(x) + sgn(z)) * min(|x|, |y|, |z|)

    '''
    min_xyz = np.minimum(np.minimum(np.abs(x), np.abs(y)), np.abs(z))
    
    return 0.25*np.abs(np.sign(x) + np.sign(y))*(np.sign(x) + np.sign(z))*min_xyz

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
    
    F1_HLL_RungeKutta:      numpy array of 32-bit floating point numbers
                            Interface fluxes associated with the mass density.
                            Calculated with the Harten-Lax-van Leer approximation, but
                            the left and right fluxes were calculated with the 3rd order
                            Runge-Kutta method.
                              
    F2_HLL_RungeKutta:      numpy array of 32-bit floating point numbers
                            Interface fluxes associated with the momentum density.
                            Calculated with the Harten-Lax-van Leer approximation, but
                            the left and right fluxes were calculated with the 3rd order
                            Runge-Kutta method.
    
    F3_HLL_RungeKutta:      numpy array of 32-bit floating point numbers
                            Interface fluxes associated with the energy density.
                            Calculated with the Harten-Lax-van Leer approximation, but
                            the left and right fluxes were calculated with the 3rd order
                            Runge-Kutta method.

    '''
    timeStepCandidate = np.float32(0)
    
    C1 = u1
    C2 = u2/u1
    C3 = (gamma - 1)*(u3 - 0.5*(u2**2/u1))
    
    LeftOfInterfaceRho_C1 = np.zeros(numInterfacePoints, dtype = np.float32)
    LeftOfInterfaceVel_C2 = LeftOfInterfaceRho_C1.copy()
    LeftOfInterfacePress_C3 = LeftOfInterfaceRho_C1.copy()
    RightOfInterfaceRho_C1 = LeftOfInterfaceRho_C1.copy()
    RightOfInterfaceVel_C2 = LeftOfInterfaceRho_C1.copy()
    RightOfInterfacePress_C3 = LeftOfInterfaceRho_C1.copy()
    
    LeftOfInterfaceRho_C1[0:numInterfacePoints] = C1[1:numInterfacePoints+1] + 0.5*minmod(theta*(C1[1:numInterfacePoints+1] - C1[0:numInterfacePoints]), 0.5*(C1[2:numInterfacePoints+2] - C1[0:numInterfacePoints]), theta*(C1[2:numInterfacePoints+2] - C1[1:numInterfacePoints+1]))
    LeftOfInterfaceVel_C2[0:numInterfacePoints] = C2[1:numInterfacePoints+1] + 0.5*minmod(theta*(C2[1:numInterfacePoints+1] - C2[0:numInterfacePoints]), 0.5*(C2[2:numInterfacePoints+2] - C2[0:numInterfacePoints]), theta*(C2[2:numInterfacePoints+2] - C2[1:numInterfacePoints+1]))
    LeftOfInterfacePress_C3[0:numInterfacePoints] = C3[1:numInterfacePoints+1] + 0.5*minmod(theta*(C3[1:numInterfacePoints+1] - C3[0:numInterfacePoints]), 0.5*(C3[2:numInterfacePoints+2] - C3[0:numInterfacePoints]), theta*(C3[2:numInterfacePoints+2] - C3[1:numInterfacePoints+1]))
        
    RightOfInterfaceRho_C1[0:numInterfacePoints] = C1[2:numInterfacePoints+2] + 0.5*minmod(theta*(C1[2:numInterfacePoints+2] - C1[1:numInterfacePoints+1]), 0.5*(C1[3:numInterfacePoints+3] - C1[1:numInterfacePoints+1]), theta*(C1[3:numInterfacePoints+3] - C1[2:numInterfacePoints+2]))
    RightOfInterfaceVel_C2[0:numInterfacePoints] = C2[2:numInterfacePoints+2] + 0.5*minmod(theta*(C2[2:numInterfacePoints+2] - C2[1:numInterfacePoints+1]), 0.5*(C2[3:numInterfacePoints+3] - C2[1:numInterfacePoints+1]), theta*(C2[3:numInterfacePoints+3] - C2[2:numInterfacePoints+2]))
    RightOfInterfacePress_C3[0:numInterfacePoints] = C3[2:numInterfacePoints+2] + 0.5*minmod(theta*(C3[2:numInterfacePoints+2] - C3[1:numInterfacePoints+1]), 0.5*(C3[3:numInterfacePoints+3] - C3[1:numInterfacePoints+1]), theta*(C3[3:numInterfacePoints+3] - C3[2:numInterfacePoints+2]))

    LeftOfInterfaceFlux_1 = LeftOfInterfaceRho_C1*LeftOfInterfaceVel_C2
    LeftOfInterfaceFlux_2 = LeftOfInterfaceRho_C1*LeftOfInterfaceVel_C2**2 + LeftOfInterfacePress_C3
    LeftOfInterfaceFlux_3 = 0.5*LeftOfInterfaceRho_C1*LeftOfInterfaceVel_C2**3 + (gamma/(gamma - 1))*(LeftOfInterfaceVel_C2*LeftOfInterfacePress_C3)
    
    RightOfInterfaceFlux_1 = RightOfInterfaceRho_C1*RightOfInterfaceVel_C2
    RightOfInterfaceFlux_2 = RightOfInterfaceRho_C1*RightOfInterfaceVel_C2**2 + RightOfInterfacePress_C3
    RightOfInterfaceFlux_3 = 0.5*RightOfInterfaceRho_C1*RightOfInterfaceVel_C2**3 + (gamma/(gamma - 1))*(RightOfInterfaceVel_C2*RightOfInterfacePress_C3)
    
    F1_HLL_RungeKutta = np.zeros(numInterfacePoints, dtype = np.float32)
    F2_HLL_RungeKutta = F1_HLL_RungeKutta.copy()
    F3_HLL_RungeKutta = F1_HLL_RungeKutta.copy()
    
    lambdaPlusLeft = (u2[1:numInterfacePoints+1]/u1[1:numInterfacePoints+1]) + speedOfSound(gamma, u1, u2, u3, 1, numInterfacePoints+1)
    lambdaMinusLeft = (u2[1:numInterfacePoints+1]/u1[1:numInterfacePoints+1]) - speedOfSound(gamma, u1, u2, u3, 1, numInterfacePoints+1)
    lambdaPlusRight = (u2[2:numInterfacePoints+2]/u1[2:numInterfacePoints+2]) + speedOfSound(gamma, u1, u2, u3, 2, numInterfacePoints+2)
    lambdaMinusRight = (u2[2:numInterfacePoints+2]/u1[2:numInterfacePoints+2]) - speedOfSound(gamma, u1, u2, u3, 2, numInterfacePoints+2)
    
    zeros = np.zeros(len(lambdaMinusLeft), dtype = np.float32)
    
    alphaPlus = np.maximum(np.maximum(zeros, lambdaPlusLeft), lambdaPlusRight)
    alphaMinus = np.maximum(np.maximum(zeros, -1*lambdaMinusLeft), -1*lambdaMinusRight)
    
    possibleTimeStep = np.float32(deltaX/np.maximum(alphaPlus, alphaMinus))
    timeStepCandidate = np.min(possibleTimeStep)/n
    
    F1_HLL_RungeKutta[0:numInterfacePoints] = ((alphaPlus*LeftOfInterfaceFlux_1[0:numInterfacePoints]) + (alphaMinus*RightOfInterfaceFlux_1[0:numInterfacePoints]) - (alphaPlus*alphaMinus*(u1[2:numInterfacePoints+2] - u1[1:numInterfacePoints+1])))/(alphaPlus + alphaMinus)
    F2_HLL_RungeKutta[0:numInterfacePoints] = ((alphaPlus*LeftOfInterfaceFlux_2[0:numInterfacePoints]) + (alphaMinus*RightOfInterfaceFlux_2[0:numInterfacePoints]) - (alphaPlus*alphaMinus*(u2[2:numInterfacePoints+2] - u2[1:numInterfacePoints+1])))/(alphaPlus + alphaMinus)
    F3_HLL_RungeKutta[0:numInterfacePoints] = ((alphaPlus*LeftOfInterfaceFlux_3[0:numInterfacePoints]) + (alphaMinus*RightOfInterfaceFlux_3[0:numInterfacePoints]) - (alphaPlus*alphaMinus*(u3[2:numInterfacePoints+2] - u3[1:numInterfacePoints+1])))/(alphaPlus + alphaMinus)
    
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
    
    DivF1[0:numPoints] = -1*(F1_HLL_RungeKutta[1:numPoints+1] - F1_HLL_RungeKutta[0:numPoints])/deltaX
    DivF2[0:numPoints] = -1*(F2_HLL_RungeKutta[1:numPoints+1] - F2_HLL_RungeKutta[0:numPoints])/deltaX
    DivF3[0:numPoints] = -1*(F3_HLL_RungeKutta[1:numPoints+1] - F3_HLL_RungeKutta[0:numPoints])/deltaX
        
    return DivF1, DivF2, DivF3

def computeAndReturnU_1(numPoints, u1, u2, u3, timeStep, DivF1, DivF2, DivF3):
    '''
    Parameters
    -----------
    numPoints :             32-bit integer
                            The number of points in the spatial grid
                            
    u1:                     numpy array of 32-bit floats
                            numpy array of mass densities
                        
    u2:                     numpy array of 32-bit floats
                            numpy array of momentum densities
                        
    u3:                     numpy array of 32-bit floats
                            numpy array of energy densities
                            
    timeStep :              32-bit floating point number
                            An appropriate timestep that does not violate the Courant-Friedrich-Levy condition.
                            
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
                              
    Returns:
    --------
    u1_1:                   numpy array of 32-bit floats
                            numpy array of higher order mass densities from U(1)
                        
    u2_1:                   numpy array of 32-bit floats
                            numpy array of higher order momentum densities from U(1)
                        
    u3_1:                   numpy array of 32-bit floats
                            numpy array of higher order energy densities from U(1)
    '''
    u1_1 = np.zeros(len(u1), dtype = np.float32)
    u2_1 = u1_1.copy()
    u3_1 = u1_1.copy()
    
    u1_1[2: numPoints + 2] = u1[2: numPoints + 2] + timeStep*DivF1
    u2_1[2: numPoints + 2] = u2[2: numPoints + 2] + timeStep*DivF2
    u3_1[2: numPoints + 2] = u3[2: numPoints + 2] + timeStep*DivF3
    
    u1_1[1] = u1_1[2]
    u2_1[1] = u2_1[2]
    u3_1[1] = u3_1[2]
    
    u1_1[0] = u1_1[2]
    u2_1[0] = u2_1[2]
    u3_1[0] = u3_1[2]
    
    u1_1[numPoints + 2: len(u1)] = u1_1[numPoints + 1]
    u2_1[numPoints + 2: len(u1)] = u2_1[numPoints + 1]
    u3_1[numPoints + 2: len(u1)] = u3_1[numPoints + 1]
        
    return u1_1, u2_1, u3_1

def computeAndReturnU_2(numPoints, u1, u2, u3, u1_1, u2_1, u3_1, timeStep, DivF1_1, DivF2_1, DivF3_1):
    '''
    Parameters
    -----------
    numPoints :             32-bit integer
                            The number of points in the spatial grid
        
    u1:                     numpy array of 32-bit floats
                            numpy array of mass densities
                        
    u2:                     numpy array of 32-bit floats
                            numpy array of momentum densities
                        
    u3:                     numpy array of 32-bit floats
                            numpy array of energy densities
                            
    u1_1:                   numpy array of 32-bit floats
                            numpy array of higher order mass densities from U(1)
                        
    u2_1:                   numpy array of 32-bit floats
                            numpy array of higher order momentum densities from U(1)
                        
    u3_1:                   numpy array of 32-bit floats
                            numpy array of higher order energy densities from U(1)
                            
    timeStep :              32-bit floating point number
                            An appropriate timestep that does not violate the Courant-Friedrich-Levy condition.
                            
    DivF1_1 :               numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the higher order mass density divided by the 
                            spatial grid spacing
                              
    DivF2_1 :               numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the higher order momentum density divided by the 
                            spatial grid spacing
        
    DivF3_1 :               numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the higher order energy density divided by the 
                            spatial grid spacing

    Returns
    -------
    u1_2:                   numpy array of 32-bit floats
                            numpy array of higher order mass densities related to U(2)
                        
    u2_2:                   numpy array of 32-bit floats
                            numpy array of higher order momentum densities related to U(2)
                        
    u3_2:                   numpy array of 32-bit floats
                            numpy array of higher order energy densities related to U(2)
                            
    '''
    u1_2 = np.zeros(len(u1), dtype = np.float32)
    u2_2 = u1_1.copy()
    u3_2 = u1_1.copy()
    
    u1_2[2: numPoints + 2] = 0.75*u1[2: numPoints + 2] + 0.25*u1_1[2: numPoints + 2] + 0.25*timeStep*DivF1_1
    u2_2[2: numPoints + 2] = 0.75*u2[2: numPoints + 2] + 0.25*u2_1[2: numPoints + 2] + 0.25*timeStep*DivF2_1
    u3_2[2: numPoints + 2] = 0.75*u3[2: numPoints + 2] + 0.25*u3_1[2: numPoints + 2] + 0.25*timeStep*DivF3_1
    
    u1_2[1] = u1_2[2]
    u2_2[1] = u2_2[2]
    u3_2[1] = u3_2[2]
    
    u1_2[0] = u1_2[2]
    u2_2[0] = u2_2[2]
    u3_2[0] = u3_2[2]
    
    u1_2[numPoints + 2: len(u1)] = u1_2[numPoints + 1]
    u2_2[numPoints + 2: len(u1)] = u2_2[numPoints + 1]
    u3_2[numPoints + 2: len(u1)] = u3_2[numPoints + 1]
        
    return u1_2, u2_2, u3_2

def updateQuantitiesRungeKutta(numPoints, u1, u2, u3, u1_2, u2_2, u3_2, timeStep, DivF1_2, DivF2_2, DivF3_2):
    '''
    Parameters
    -----------
    numPoints :             32-bit integer
                            The number of points in the spatial grid
        
    u1:                     numpy array of 32-bit floats
                            numpy array of mass densities
                        
    u2:                     numpy array of 32-bit floats
                            numpy array of momentum densities
                        
    u3:                     numpy array of 32-bit floats
                            numpy array of energy densities
                            
    u1_2:                   numpy array of 32-bit floats
                            numpy array of higher order mass densities from U(2)
                        
    u2_2:                   numpy array of 32-bit floats
                            numpy array of higher order momentum densities from U(2)
                        
    u3_2:                   numpy array of 32-bit floats
                            numpy array of higher order energy densities from U(2)
                            
    timeStep :              32-bit floating point number
                            An appropriate timestep that does not violate the Courant-Friedrich-Levy condition.
                            
    DivF1_2 :               numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the higher order mass density divided by the 
                            spatial grid spacing
                              
    DivF2_2 :               numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the higher order momentum density divided by the 
                            spatial grid spacing
        
    DivF3_2 :               numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the higher order energy density divided by the 
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

    u1[2: numPoints + 2] = (1/3)*u1[2: numPoints + 2] + (2/3)*u1_2[2: numPoints + 2] + (2/3)*timeStep*DivF1_2
    u2[2: numPoints + 2] = (1/3)*u2[2: numPoints + 2] + (2/3)*u2_2[2: numPoints + 2] + (2/3)*timeStep*DivF2_2
    u3[2: numPoints + 2] = (1/3)*u3[2: numPoints + 2] + (2/3)*u3_2[2: numPoints + 2] + (2/3)*timeStep*DivF3_2
        
    u1[1] = u1[2]
    u2[1] = u2[2]
    u3[1] = u3[2]
    
    u1[0] = u1[2]
    u2[0] = u2[2]
    u3[0] = u3[2]
    
    u1[numPoints + 2: len(u1)] = u1[numPoints + 1]
    u2[numPoints + 2: len(u1)] = u2[numPoints + 1]
    u3[numPoints + 2: len(u1)] = u3[numPoints + 1]
    
    return u1, u2, u3

def doEntireRungeKuttaTimeStep(numPoints, u1, u2, u3, timeStep, DivF1, DivF2, DivF3, deltaX, gamma, theta, timeFrac):
    '''
    Parameters
    -----------
    numPoints :               32-bit integer
                              The number of points in the spatial grid
        
    u1:                       numpy array of 32-bit floats
                              numpy array of mass densities
                        
    u2:                       numpy array of 32-bit floats
                              numpy array of momentum densities
                        
    u3:                       numpy array of 32-bit floats
                              numpy array of energy densities
                            
    timeStep :                32-bit floating point number
                              An appropriate timestep that does not violate the Courant-Friedrich-Levy condition.
                            
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
                              
    deltaX :                  32-bit floating point number
                              The spatial grid spacing
                              
    gamma:                    32-bit floating point number
                              The adiabatic index
    
    theta :                   32-bit floating point number
                              A parameter between 1 and 2
                              
    timeFrac:                 32-bit integer
                              how much one wants to divide the minimum time step by
                              
    Returns
    -------
    u1:                     numpy array of 32-bit floats
                            numpy array of updated mass densities
                        
    u2:                     numpy array of 32-bit floats
                            numpy array of updated momentum densities
                        
    u3:                     numpy array of 32-bit floats
                            numpy array of updated energy densities
    '''
    u1_1, u2_1, u3_1 = computeAndReturnU_1(numPoints, u1, u2, u3, timeStep, DivF1, DivF2, DivF3)
        
    timeStepCandidate, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1 = computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_1, u2_1, u3_1)
    DivF1_1, DivF2_1, DivF3_1 = computeFluxDivergenceRungeKutta(deltaX, numPoints, F1_HLL_RungeKutta_1, F2_HLL_RungeKutta_1, F3_HLL_RungeKutta_1) 
        
    u1_2, u2_2, u3_2 = computeAndReturnU_2(numPoints, u1, u2, u3, u1_1, u2_1, u3_1, timeStep, DivF1_1, DivF2_1, DivF3_1)
        
    timeStepCandidate2, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2 = computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numPoints+1, timeFrac, u1_2, u2_2, u3_2)
    DivF1_2, DivF2_2, DivF3_2 = computeFluxDivergenceRungeKutta(deltaX, numPoints, F1_HLL_RungeKutta_2, F2_HLL_RungeKutta_2, F3_HLL_RungeKutta_2)
        
    u1, u2, u3 = updateQuantitiesRungeKutta(numPoints, u1, u2, u3, u1_2, u2_2, u3_2, timeStep, DivF1_2, DivF2_2, DivF3_2)
      
    return u1, u2, u3