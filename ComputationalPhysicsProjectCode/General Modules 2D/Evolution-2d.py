import numpy as np

def speedOfSound(gamma, u1, u2, u3, u4): 
    '''
    A function that computes and returns the speed of sound
    
    Parameters
    ----------
    gamma :             32-bit floating point number
                        Adiabatic constant 
                        
    u1 :                    numpy array of 32-bit floats
                            Array of mass density values
                            
    u2 :                    numpy array of 32-bit floats
                            Array of x-momentum density values
    
    u3 :                    numpy array of 32-bit floats
                            Array of y-momentum density values
                            
    u4 :                    numpy array of 32-bit floats
                            Array of energy density values
    
    Returns
    -------
    numpy array of 32-bit floats
    Array of the speed of sound
    
    '''
    epsilon = 1e-6
    u1 = np.maximum(epsilon, abs(u1))
    c = gamma*(gamma - 1)*((u4/u1) - 0.5*(u2**2/u1**2) \
    - 0.5*(u3**2/u1**2)) 
    return np.sqrt(c)

def minmod(x, y, z):
    '''
    This function returns the minmod of three input parameters x, y, and z. The minmod 
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
    sgn_x = np.sign(x)
    sgn_y = np.sign(y)
    sgn_z = np.sign(z)
    
    min_xyz = np.min(np.abs([x, y, z]))
    
    return 0.25 * np.abs(sgn_x + sgn_y) * (sgn_x + sgn_z) * min_xyz
    
def minmod_array(x, y, z):
    '''
    This function returns the minmod of three input parameters x, y, and z. The minmod 
    is defined as 0.25 |sgn(x) + sgn(y)| * (sgn(x) + sgn(z)) * min(|x|, |y|, |z|)
    
    Parameters
    ----------
    x :       NumPy array of 32-bit floating-point numbers
              An input parameter
        
    y :       NumPy array of 32-bit floating-point numbers
              An input parameter
        
    z :       NumPy array of 32-bit floating-point numbers
              An input parameter
    
    Returns
    -------
    The minmod = 0.25 |sgn(x) + sgn(y)| * (sgn(x) + sgn(z)) * min(|x|, |y|, |z|)
    
    '''
    sgn_x = np.sign(x)
    sgn_y = np.sign(y)
    sgn_z = np.sign(z)
    
    min_xyz = np.min(np.abs(np.stack([x, y, z], axis=-1)), axis=-1)
    
    return 0.25 * np.abs(sgn_x + sgn_y) * (sgn_x + sgn_z) * min_xyz    
    
def computeTimeStepandFLLRungeKutta(deltaX, gamma, theta, numInterfacePoints, n, u1, u2, u3, u4):
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
                        
    u1 :                    numpy array of 32-bit floats
                            Array of mass density values
                            
    u2 :                    numpy array of 32-bit floats
                            Array of x-momentum density values
    
    u3 :                    numpy array of 32-bit floats
                            Array of y-momentum density values
                            
    u4 :                    numpy array of 32-bit floats
                            Array of energy density values
    
    Returns
    -------
    timeStepCandidate:      32-bit floating point number
                            A candidate for the time step that satisfies 
                            Courant-Friedrich-Levy condition. It is one-n-th of the 
                            minimum of all such times that each satisfy the 
                            Courant-Friedrich-Levy condition
    
    F1_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the x-mass density.
                              Calculated with the 3rd order
                              Runge-Kutta method.
                              
    F2_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the x-momentum density.
                              Calculated with the 3rd order
                              Runge-Kutta method.
                              
    F3_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the x-momentum & y-momentum.
                              Calculated with the 3rd order
                              Runge-Kutta method.                          
    
    F4_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the x-energy density.
                              Calculated with the 3rd order
                              Runge-Kutta method.
                              
    G1_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the y-mass density.
                              Calculated with the 3rd order
                              Runge-Kutta method.
                              
    G2_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the y-momentum & x-momentum.
                              Calculated with the 3rd order
                              Runge-Kutta method.
                              
    G3_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the y-momentum density.
                              Calculated with the 3rd order
                              Runge-Kutta method.                          
    
    G4_HLL_RungeKutta:        numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the y-energy density.
                              Calculated with the 3rd order
                              Runge-Kutta method.                          
    
    '''
    
    C1 = u1
    C2 = u2/u1
    C3 = u3/u1
    C4 = (gamma - 1)*(u4 - 0.5*(u2**2/u1) - 0.5*(u3**2/u1))
    
    LeftOfInterfaceRho_C1 = C1[:-3, 1:-2] + 0.5 * minmod_array(
        theta * (C1[:-3, 1:-2] - C1[:-3, :-3]),
        0.5 * (C1[:-3, 2:-1] - C1[:-3, :-3]),
        theta * (C1[:-3, 2:-1] - C1[:-3, 1:-2]),
    )

    LeftOfInterfaceVelx_C2 = C2[:-3, 1:-2] + 0.5 * minmod_array(
        theta * (C2[:-3, 1:-2] - C2[:-3, :-3]),
        0.5 * (C2[:-3, 2:-1] - C2[:-3, :-3]),
        theta * (C2[:-3, 2:-1] - C2[:-3, 1:-2]),
    )

    LeftOfInterfaceVely_C3 = C3[:-3, 1:-2] + 0.5 * minmod_array(
        theta * (C3[:-3, 1:-2] - C3[:-3, :-3]),
        0.5 * (C3[:-3, 2:-1] - C3[:-3, :-3]),
        theta * (C3[:-3, 2:-1] - C3[:-3, 1:-2]),
    )

    LeftOfInterfacePress_C4 = C4[:-3, 1:-2] + 0.5 * minmod_array(
        theta * (C4[:-3, 1:-2] - C4[:-3, :-3]),
        0.5 * (C4[:-3, 2:-1] - C4[:-3, :-3]),
        theta * (C4[:-3, 2:-1] - C4[:-3, 1:-2]),
    )
    
    RightOfInterfaceRho_C1 = C1[:-3, 2:-1] + 0.5 * minmod_array(
        theta * (C1[:-3, 2:-1] - C1[:-3, 1:-2]),
        0.5 * (C1[:-3, 3:] - C1[:-3, 1:-2]),
        theta * (C1[:-3, 3:] - C1[:-3, 2:-1]),
    )

    RightOfInterfaceVelx_C2 = C2[:-3, 2:-1] + 0.5 * minmod_array(
        theta * (C2[:-3, 2:-1] - C2[:-3, 1:-2]),
        0.5 * (C2[:-3, 3:] - C2[:-3, 1:-2]),
        theta * (C2[:-3, 3:] - C2[:-3, 2:-1]),
    )

    RightOfInterfaceVely_C3 = C3[:-3, 2:-1] + 0.5 * minmod_array(
        theta * (C3[:-3, 2:-1] - C3[:-3, 1:-2]),
        0.5 * (C3[:-3, 3:] - C3[:-3, 1:-2]),
        theta * (C3[:-3, 3:] - C3[:-3, 2:-1]),
    )

    RightOfInterfacePress_C4 = C4[:-3, 2:-1] + 0.5 * minmod_array(
        theta * (C4[:-3, 2:-1] - C4[:-3, 1:-2]),
        0.5 * (C4[:-3, 3:] - C4[:-3, 1:-2]),
        theta * (C4[:-3, 3:] - C4[:-3, 2:-1]),
    )
    
    BottomOfInterfaceRho_C1 = C1[1:-2, :-3] + 0.5 * minmod_array(
        theta * (C1[1:-2, :-3] - C1[:-3, :-3]),
        0.5 * (C1[2:-1, :-3] - C1[:-3, :-3]),
        theta * (C1[2:-1, :-3] - C1[1:-2, :-3]),
    )

    BottomOfInterfaceVelx_C2 = C2[1:-2, :-3] + 0.5 * minmod_array(
        theta * (C2[1:-2, :-3] - C2[:-3, :-3]),
        0.5 * (C2[2:-1, :-3] - C2[:-3, :-3]),
        theta * (C2[2:-1, :-3] - C2[1:-2, :-3]),
    )

    BottomOfInterfaceVely_C3 = C3[1:-2, :-3] + 0.5 * minmod_array(
        theta * (C3[1:-2, :-3] - C3[:-3, :-3]),
        0.5 * (C3[2:-1, :-3] - C3[:-3, :-3]),
        theta * (C3[2:-1, :-3] - C3[1:-2, :-3]),
    )

    BottomOfInterfacePress_C4 = C4[1:-2, :-3] + 0.5 * minmod_array(
        theta * (C4[1:-2, :-3] - C4[:-3, :-3]),
        0.5 * (C4[2:-1, :-3] - C4[:-3, :-3]),
        theta * (C4[2:-1, :-3] - C4[1:-2, :-3]),
    )
    
    TopOfInterfaceRho_C1 = C1[2:-1, :-3] + 0.5 * minmod_array(
        theta * (C1[2:-1, :-3] - C1[1:-2, :-3]),
        0.5 * (C1[3:, :-3] - C1[1:-2, :-3]),
        theta * (C1[3:, :-3] - C1[2:-1, :-3]),
    )

    TopOfInterfaceVelx_C2 = C2[2:-1, :-3] + 0.5 * minmod_array(
        theta * (C2[2:-1, :-3] - C2[1:-2, :-3]),
        0.5 * (C2[3:, :-3] - C2[1:-2, :-3]),
        theta * (C2[3:, :-3] - C2[2:-1, :-3]),
    )

    TopOfInterfaceVely_C3 = C3[2:-1, :-3] + 0.5 * minmod_array(
        theta * (C3[2:-1, :-3] - C3[1:-2, :-3]),
        0.5 * (C3[3:, :-3] - C3[1:-2, :-3]),
        theta * (C3[3:, :-3] - C3[2:-1, :-3]),
    )

    TopOfInterfacePress_C4 = C4[2:-1, :-3] + 0.5 * minmod_array(
        theta * (C4[2:-1, :-3] - C4[1:-2, :-3]),
        0.5 * (C4[3:, :-3] - C4[1:-2, :-3]),
        theta * (C4[3:, :-3] - C4[2:-1, :-3]),
    )

    LeftOfInterfaceFlux_1 = LeftOfInterfaceRho_C1*LeftOfInterfaceVelx_C2
    LeftOfInterfaceFlux_2 = LeftOfInterfaceRho_C1*LeftOfInterfaceVelx_C2**2 + LeftOfInterfacePress_C4
    LeftOfInterfaceFlux_3 = LeftOfInterfaceRho_C1*LeftOfInterfaceVelx_C2*LeftOfInterfaceVely_C3
    LeftOfInterfaceFlux_4 = 0.5*LeftOfInterfaceRho_C1*LeftOfInterfaceVelx_C2**3 + (gamma/(gamma - 1))*(LeftOfInterfaceVelx_C2*LeftOfInterfacePress_C4) \
                            + 0.5*LeftOfInterfaceRho_C1*LeftOfInterfaceVelx_C2*LeftOfInterfaceVely_C3**2
                            
    RightOfInterfaceFlux_1 = RightOfInterfaceRho_C1*RightOfInterfaceVelx_C2
    RightOfInterfaceFlux_2 = RightOfInterfaceRho_C1*RightOfInterfaceVelx_C2**2 + RightOfInterfacePress_C4
    RightOfInterfaceFlux_3 = RightOfInterfaceRho_C1*RightOfInterfaceVelx_C2*RightOfInterfaceVely_C3
    RightOfInterfaceFlux_4 = 0.5*RightOfInterfaceRho_C1*RightOfInterfaceVelx_C2**3 + (gamma/(gamma - 1))*(RightOfInterfaceVelx_C2*RightOfInterfacePress_C4) \
                            + 0.5*RightOfInterfaceRho_C1*RightOfInterfaceVelx_C2*RightOfInterfaceVely_C3**2 
                            
    BottomOfInterfaceFlux_1 = BottomOfInterfaceRho_C1*BottomOfInterfaceVely_C3
    BottomOfInterfaceFlux_2 = BottomOfInterfaceRho_C1*BottomOfInterfaceVelx_C2*BottomOfInterfaceVely_C3
    BottomOfInterfaceFlux_3 = BottomOfInterfaceRho_C1*BottomOfInterfaceVely_C3**2 + BottomOfInterfacePress_C4
    BottomOfInterfaceFlux_4 = 0.5*BottomOfInterfaceRho_C1*BottomOfInterfaceVely_C3**3 + (gamma/(gamma - 1))*(BottomOfInterfaceVely_C3*BottomOfInterfacePress_C4) \
                            + 0.5*BottomOfInterfaceRho_C1*BottomOfInterfaceVely_C3*BottomOfInterfaceVelx_C2**2 
                            
    TopOfInterfaceFlux_1 = TopOfInterfaceRho_C1*TopOfInterfaceVely_C3
    TopOfInterfaceFlux_2 = TopOfInterfaceRho_C1*TopOfInterfaceVelx_C2*TopOfInterfaceVely_C3
    TopOfInterfaceFlux_3 = TopOfInterfaceRho_C1*TopOfInterfaceVely_C3**2 + TopOfInterfacePress_C4
    TopOfInterfaceFlux_4 = 0.5*TopOfInterfaceRho_C1*TopOfInterfaceVely_C3**3 + (gamma/(gamma - 1))*(TopOfInterfaceVely_C3*TopOfInterfacePress_C4) \
                          + 0.5*TopOfInterfaceRho_C1*TopOfInterfaceVely_C3*TopOfInterfaceVelx_C2**2                        
    
    epsilon = 1e-6
    #u1 = np.maximum(epsilon, abs(u1))
    speedOfSoundlocal = speedOfSound(gamma, u1, u2, u3, u4)
    
    lambdaPlusLeft = (u2[:-3, 1:-2]/u1[:-3, 1:-2]) + speedOfSoundlocal[:-3, 1:-2]
    lambdaMinusLeft = (u2[:-3, 1:-2]/u1[:-3, 1:-2]) - speedOfSoundlocal[:-3, 1:-2]
    lambdaPlusRight = (u2[:-3, 2:-1]/u1[:-3, 2:-1]) + speedOfSoundlocal[:-3, 2:-1]
    lambdaMinusRight = (u2[:-3, 2:-1]/u1[:-3, 2:-1]) - speedOfSoundlocal[:-3, 2:-1]
    
    lambdaPlusBottom = (u3[1:-2, :-3]/u1[1:-2, :-3]) + speedOfSoundlocal[1:-2, :-3]
    lambdaMinusBottom = (u3[1:-2, :-3]/u1[1:-2, :-3]) - speedOfSoundlocal[1:-2, :-3]
    lambdaPlusTop = (u3[2:-1, :-3]/u1[2:-1, :-3]) + speedOfSoundlocal[2:-1, :-3]
    lambdaMinusTop = (u3[2:-1, :-3]/u1[2:-1, :-3]) - speedOfSoundlocal[2:-1, :-3]
    
    alphaPlus = np.maximum(epsilon, lambdaPlusLeft, lambdaPlusRight)
    alphaMinus = np.maximum(epsilon, -1*lambdaMinusLeft, -1*lambdaMinusRight)
    betaPlus = np.maximum(epsilon, lambdaPlusBottom, lambdaPlusTop)
    betaMinus = np.maximum(epsilon, -1*lambdaMinusBottom, -1*lambdaMinusTop)
    alphamax = np.maximum(alphaPlus, alphaMinus)
    possibleTimeStep = (deltaX/np.maximum(alphamax, betaPlus, betaMinus))
    timeStepCandidate = possibleTimeStep.min()
    
    F1_HLL_RungeKutta = ((alphaPlus*LeftOfInterfaceFlux_1) + (alphaMinus*RightOfInterfaceFlux_1) - (alphaPlus*alphaMinus*(u1[:-3, 2:-1] - u1[:-3, 1:-2])))/(alphaPlus + alphaMinus)
    F2_HLL_RungeKutta = ((alphaPlus*LeftOfInterfaceFlux_2) + (alphaMinus*RightOfInterfaceFlux_2) - (alphaPlus*alphaMinus*(u2[:-3, 2:-1] - u2[:-3, 1:-2])))/(alphaPlus + alphaMinus)
    F3_HLL_RungeKutta = ((alphaPlus*LeftOfInterfaceFlux_3) + (alphaMinus*RightOfInterfaceFlux_3) - (alphaPlus*alphaMinus*(u3[:-3, 2:-1] - u3[:-3, 1:-2])))/(alphaPlus + alphaMinus)
    F4_HLL_RungeKutta = ((alphaPlus*LeftOfInterfaceFlux_4) + (alphaMinus*RightOfInterfaceFlux_4) - (alphaPlus*alphaMinus*(u4[:-3, 2:-1] - u4[:-3, 1:-2])))/(alphaPlus + alphaMinus)
    
    G1_HLL_RungeKutta = ((betaPlus*BottomOfInterfaceFlux_1) + (betaMinus*TopOfInterfaceFlux_1) - (betaPlus*betaMinus*(u1[2:-1, :-3] - u1[1:-2, :-3])))/(betaPlus + betaMinus)
    G2_HLL_RungeKutta = ((betaPlus*BottomOfInterfaceFlux_2) + (betaMinus*TopOfInterfaceFlux_2) - (betaPlus*betaMinus*(u2[2:-1, :-3] - u2[1:-2, :-3])))/(betaPlus + betaMinus)
    G3_HLL_RungeKutta = ((betaPlus*BottomOfInterfaceFlux_3) + (betaMinus*TopOfInterfaceFlux_3) - (betaPlus*betaMinus*(u3[2:-1, :-3] - u3[1:-2, :-3])))/(betaPlus + betaMinus)
    G4_HLL_RungeKutta = ((betaPlus*BottomOfInterfaceFlux_4) + (betaMinus*TopOfInterfaceFlux_4) - (betaPlus*betaMinus*(u4[2:-1, :-3] - u4[1:-2, :-3])))/(betaPlus + betaMinus)
    
    timeStepCandidate = timeStepCandidate/n
    
    return timeStepCandidate, F1_HLL_RungeKutta, F2_HLL_RungeKutta, F3_HLL_RungeKutta, F4_HLL_RungeKutta,\
           G1_HLL_RungeKutta, G2_HLL_RungeKutta, G3_HLL_RungeKutta, G4_HLL_RungeKutta
    

def computeFluxDivergence(deltaX, numPoints, F1_HLL, F2_HLL, F3_HLL, F4_HLL, G1_HLL, G2_HLL, G3_HLL, G4_HLL):
    '''
    Parameters
    ----------
    deltaX :                  32-bit floating point number
                              The spatial grid spacing
       
    numPoints :               32-bit integer
                              The number points in the spatial grid.
    
    F1_HLL :                  numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the x-mass density.
                              Calculated with the 3rd order
                              Runge-Kutta method.
                              
    F2_HLL :                  numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the x-momentum density.
                              Calculated with the 3rd order
                              Runge-Kutta method.
                              
    F3_HLL :                  numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the x-momentum & y-momentum.
                              Calculated with the 3rd order
                              Runge-Kutta method.                          
    		             
    F4_HLL :                  numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the x-energy density.
                              Calculated with the 3rd order
                              Runge-Kutta method.
                              
    G1_HLL :                  numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the y-mass density.
                              Calculated with the 3rd order
                              Runge-Kutta method.
                              
    G2_HLL :                  numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the y-momentum & x-momentum.
                              Calculated with the 3rd order
                              Runge-Kutta method.
                              
    G3_HLL :                  numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the y-momentum density.
                              Calculated with the 3rd order
                              Runge-Kutta method.                          
    		             
    G4_HLL :                  numpy array of 32-bit floating point numbers
                              Interface fluxes associated with the y-energy density.
                              Calculated with the 3rd order
                              Runge-Kutta method.     
    
    Returns
    -------
    DivF1 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-mass density divided by the 
                            spatial grid spacing
                              
    DivF2 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-momentum density divided by the 
                            spatial grid spacing
                            
    DivF3 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x&y-momentum density divided by the 
                            spatial grid spacing                        
        
    DivF4 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-energy density divided by the 
                            spatial grid spacing
                            
    DivG1 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-mass density divided by the 
                            spatial grid spacing
                              
    DivG2 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y&x-momentum density divided by the 
                            spatial grid spacing
                            
    DivG3 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-momentum density divided by the 
                            spatial grid spacing                        
        
    DivG4 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-energy density divided by the 
                            spatial grid spacing
    
    '''
    DivF1 = -1*(F1_HLL[:-1, 1:] - F1_HLL[:-1, :-1])/deltaX
    DivF2 = -1*(F2_HLL[:-1, 1:] - F2_HLL[:-1, :-1])/deltaX
    DivF3 = -1*(F3_HLL[:-1, 1:] - F3_HLL[:-1, :-1])/deltaX 
    DivF4 = -1*(F4_HLL[:-1, 1:] - F4_HLL[:-1, :-1])/deltaX 
    DivG1 = -1*(G1_HLL[1:, :-1] - G1_HLL[:-1, :-1])/deltaX 
    DivG2 = -1*(G2_HLL[1:, :-1] - G2_HLL[:-1, :-1])/deltaX 
    DivG3 = -1*(G3_HLL[1:, :-1] - G3_HLL[:-1, :-1])/deltaX
    DivG4 = -1*(G4_HLL[1:, :-1] - G4_HLL[:-1, :-1])/deltaX
        
    return DivF1, DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4

def updateQuantitiesRungeKutta_order1(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, DivF1, DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4):
    '''
    Parameters
    ----------
    timeStep :              32-bit floating point number
                            An appropriate timestep that does not violate the Courant-Friedrich-Levy condition.
                            
    numPoints :             32-bit integer
                            The number of points in the spatial grid
    
    PeriodicX :             logical value of True if there are periodic 
                            boundry conditions in the x-direction and False otherwise 
                            
    PeriodicY :             logical value of True if there are periodic 
                            boundry conditions in the y-direction and False otherwise                        
        
    u1 :                    numpy array of 32-bit floats
                            Array of mass density values
                            
    u2 :                    numpy array of 32-bit floats
                            Array of x-momentum density values
    
    u3 :                    numpy array of 32-bit floats
                            Array of y-momentum density values
                            
    u4 :                    numpy array of 32-bit floats
                            Array of energy density values
                            
    DivF1 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-mass density divided by the 
                            spatial grid spacing
                              
    DivF2 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-momentum density divided by the 
                            spatial grid spacing
                            
    DivF3 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x&y-momentum density divided by the 
                            spatial grid spacing                        
        
    DivF4 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-energy density divided by the 
                            spatial grid spacing
                            
    DivG1 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-mass density divided by the 
                            spatial grid spacing
                              
    DivG2 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y&x-momentum density divided by the 
                            spatial grid spacing
                            
    DivG3 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-momentum density divided by the 
                            spatial grid spacing                        
        
    DivG4 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-energy density divided by the 
                            spatial grid spacing                        
    
    Returns
    -------
    u1_1 :                  numpy array of 32-bit floats
                            Array of updated mass density values to the first order
                            
    u2_1 :                  numpy array of 32-bit floats
                            Array of updated x-momentum density values to the first order
    
    u3_1 :                  numpy array of 32-bit floats
                            Array of updated y-momentum density values to the first order
                            
    u4_1 :                  numpy array of 32-bit floats
                            Array of updated energy density values to the first order
    
    '''
    u1_1 = np.ones((len(u1), len(u1)), dtype = np.float32)
    u2_1 = u1_1.copy()
    u3_1 = u1_1.copy()
    u4_1 = u1_1.copy()
      
    u1_1[2: numPoints + 2, 2: numPoints + 2] = u1[2: numPoints + 2, 2: numPoints + 2] + timeStep*(DivF1 + DivG1)
    u2_1[2: numPoints + 2, 2: numPoints + 2] = u2[2: numPoints + 2, 2: numPoints + 2] + timeStep*(DivF2 + DivG2)
    u3_1[2: numPoints + 2, 2: numPoints + 2] = u3[2: numPoints + 2, 2: numPoints + 2] + timeStep*(DivF3 + DivG3)    
    u4_1[2: numPoints + 2, 2: numPoints + 2] = u4[2: numPoints + 2, 2: numPoints + 2] + timeStep*(DivF4 + DivG4)
    
    if (PeriodicX==True):
        u1_1[:,0] = u1_1[:,numPoints]
        u2_1[:,0] = u2_1[:,numPoints]
        u3_1[:,0] = u3_1[:,numPoints]
        u4_1[:,0] = u4_1[:,numPoints]
        
        u1_1[:,1] = u1_1[:,numPoints + 1]
        u2_1[:,1] = u2_1[:,numPoints + 1]
        u3_1[:,1] = u3_1[:,numPoints + 1]
        u4_1[:,1] = u4_1[:,numPoints + 1]
        
        u1_1[:,numPoints + 2] = u1_1[:,2]
        u2_1[:,numPoints + 2] = u2_1[:,2]
        u3_1[:,numPoints + 2] = u3_1[:,2]
        u4_1[:,numPoints + 2] = u4_1[:,2]
        
        u1_1[:,numPoints + 3] = u1_1[:,3]
        u2_1[:,numPoints + 3] = u2_1[:,3]
        u3_1[:,numPoints + 3] = u3_1[:,3]
        u4_1[:,numPoints + 3] = u4_1[:,3]
        
    else:	
        u1_1[:,0] = u1_1[:,2]
        u2_1[:,0] = u2_1[:,2]
        u3_1[:,0] = u3_1[:,2]
        u4_1[:,0] = u4_1[:,2] 
    
        u1_1[:,1] = u1_1[:,0]
        u2_1[:,1] = u2_1[:,0]
        u3_1[:,1] = u3_1[:,0]
        u4_1[:,1] = u4_1[:,0]
        
        for i in range(numPoints + 2, len(u1_1[0])):
            u1_1[:,i] = u1_1[:,numPoints + 1]
            u2_1[:,i] = u2_1[:,numPoints + 1]
            u3_1[:,i] = u3_1[:,numPoints + 1]
            u4_1[:,i] = u4_1[:,numPoints + 1]
    
    if (PeriodicY==True):
        u1_1[0,:] = u1_1[numPoints,:]
        u2_1[0,:] = u2_1[numPoints,:]
        u3_1[0,:] = u3_1[numPoints,:]
        u4_1[0,:] = u4_1[numPoints,:]
        
        u1_1[1,:] = u1_1[numPoints + 1,:]
        u2_1[1,:] = u2_1[numPoints + 1,:]
        u3_1[1,:] = u3_1[numPoints + 1,:]
        u4_1[1,:] = u4_1[numPoints + 1,:]
        
        u1_1[numPoints + 2,:] = u1_1[2,:]
        u2_1[numPoints + 2,:] = u2_1[2,:]
        u3_1[numPoints + 2,:] = u3_1[2,:]
        u4_1[numPoints + 2,:] = u4_1[2,:]
        
        u1_1[numPoints + 3,:] = u1_1[3,:]
        u2_1[numPoints + 3,:] = u2_1[3,:]
        u3_1[numPoints + 3,:] = u3_1[3,:]
        u4_1[numPoints + 3,:] = u4_1[3,:]
        
    else: 	
        u1_1[0,:] = u1_1[2,:]
        u2_1[0,:] = u2_1[2,:]
        u3_1[0,:] = u3_1[2,:]
        u4_1[0,:] = u4_1[2,:]
    
        u1_1[1,:] = u1_1[0,:]
        u2_1[1,:] = u2_1[0,:]
        u3_1[1,:] = u3_1[0,:]
        u4_1[1,:] = u4_1[0,:]
        
        for i in range(numPoints + 2, len(u1_1[0])):
            u1_1[i,:] = u1_1[numPoints + 1,:]
            u2_1[i,:] = u2_1[numPoints + 1,:]
            u3_1[i,:] = u3_1[numPoints + 1,:]
            u4_1[i,:] = u4_1[numPoints + 1,:]

    return u1_1, u2_1, u3_1, u4_1
    
def updateQuantitiesRungeKutta_order2(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_1, u2_1, u3_1, u4_1, DivF1, DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4):
    '''
    Parameters
    ----------
    timeStep :              32-bit floating point number
                            An appropriate timestep that does not violate the Courant-Friedrich-Levy condition.
                            
    numPoints :             32-bit integer
                            The number of points in the spatial grid
    
    PeriodicX :             logical value of True if there are periodic 
                            boundry conditions in the x-direction and False otherwise 
                            
    PeriodicY :             logical value of True if there are periodic 
                            boundry conditions in the y-direction and False otherwise                        
        
    u1 :                    numpy array of 32-bit floats
                            Array of mass density values
                            
    u2 :                    numpy array of 32-bit floats
                            Array of x-momentum density values
    
    u3 :                    numpy array of 32-bit floats
                            Array of y-momentum density values
                            
    u4 :                    numpy array of 32-bit floats
                            Array of energy density values
                            
    u1_1 :                  numpy array of 32-bit floats
                            Array of mass density values to the first order
                            
    u2_1 :                  numpy array of 32-bit floats
                            Array of x-momentum density values to the first order
    
    u3_1 :                  numpy array of 32-bit floats
                            Array of y-momentum density values to the first order
                            
    u4_1 :                  numpy array of 32-bit floats
                            Array of energy density values to the first order                        
                            
    DivF1 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-mass density divided by the 
                            spatial grid spacing
                              
    DivF2 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-momentum density divided by the 
                            spatial grid spacing
                            
    DivF3 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x&y-momentum density divided by the 
                            spatial grid spacing                        
        
    DivF4 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-energy density divided by the 
                            spatial grid spacing
                            
    DivG1 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-mass density divided by the 
                            spatial grid spacing
                              
    DivG2 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y&x-momentum density divided by the 
                            spatial grid spacing
                            
    DivG3 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-momentum density divided by the 
                            spatial grid spacing                        
        
    DivG4 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-energy density divided by the 
                            spatial grid spacing                        
    
    Returns
    -------
    u1_2 :                  numpy array of 32-bit floats
                            Array of updated mass density values to the second order
                           
    u2_2 :                  numpy array of 32-bit floats
                            Array of updated x-momentum density values to the second order
	   
    u3_2 :                  numpy array of 32-bit floats
                            Array of updated y-momentum density values to the second order
                           
    u4_2 :                  numpy array of 32-bit floats
                            Array of updated energy density values to the second order
    
    '''
    u1_2 = np.ones((len(u1), len(u1)), dtype = np.float32)
    u2_2 = u1_2.copy()
    u3_2 = u1_2.copy()
    u4_2 = u1_2.copy()
    
    u1_2[2: numPoints + 2, 2: numPoints + 2] = 0.75*u1[2: numPoints + 2, 2: numPoints + 2] + 0.25*u1_1[2: numPoints + 2, 2: numPoints + 2] + 0.25*timeStep*(DivF1 + DivG1)
    u2_2[2: numPoints + 2, 2: numPoints + 2] = 0.75*u2[2: numPoints + 2, 2: numPoints + 2] + 0.25*u2_1[2: numPoints + 2, 2: numPoints + 2] + 0.25*timeStep*(DivF2 + DivG2)
    u3_2[2: numPoints + 2, 2: numPoints + 2] = 0.75*u3[2: numPoints + 2, 2: numPoints + 2] + 0.25*u3_1[2: numPoints + 2, 2: numPoints + 2] + 0.25*timeStep*(DivF3 + DivG3)
    u4_2[2: numPoints + 2, 2: numPoints + 2] = 0.75*u4[2: numPoints + 2, 2: numPoints + 2] + 0.25*u4_1[2: numPoints + 2, 2: numPoints + 2] + 0.25*timeStep*(DivF4 + DivG4)
    
    if (PeriodicX==True):
        u1_2[:,0] = u1_2[:,numPoints]
        u2_2[:,0] = u2_2[:,numPoints]
        u3_2[:,0] = u3_2[:,numPoints]
        u4_2[:,0] = u4_2[:,numPoints]
        
        u1_2[:,1] = u1_2[:,numPoints + 1]
        u2_2[:,1] = u2_2[:,numPoints + 1]
        u3_2[:,1] = u3_2[:,numPoints + 1]
        u4_2[:,1] = u4_2[:,numPoints + 1]
        
        u1_2[:,numPoints + 2] = u1_2[:,2]
        u2_2[:,numPoints + 2] = u2_2[:,2]
        u3_2[:,numPoints + 2] = u3_2[:,2]
        u4_2[:,numPoints + 2] = u4_2[:,2]
        
        u1_2[:,numPoints + 3] = u1_2[:,3]
        u2_2[:,numPoints + 3] = u2_2[:,3]
        u3_2[:,numPoints + 3] = u3_2[:,3]
        u4_2[:,numPoints + 3] = u4_2[:,3]
        
    else:	
        u1_2[:,0] = u1_2[:,2]
        u2_2[:,0] = u2_2[:,2]
        u3_2[:,0] = u3_2[:,2]
        u4_2[:,0] = u4_2[:,2] 
    
        u1_2[:,1] = u1_2[:,0]
        u2_2[:,1] = u2_2[:,0]
        u3_2[:,1] = u3_2[:,0]
        u4_2[:,1] = u4_2[:,0]
        
        for i in range(numPoints + 2, len(u1_2[0])):
            u1_2[:,i] = u1_2[:,numPoints + 1]
            u2_2[:,i] = u2_2[:,numPoints + 1]
            u3_2[:,i] = u3_2[:,numPoints + 1]
            u4_2[:,i] = u4_2[:,numPoints + 1]
            
    if (PeriodicY==True):
        u1_2[0,:] = u1_2[numPoints,:]
        u2_2[0,:] = u2_2[numPoints,:]
        u3_2[0,:] = u3_2[numPoints,:]
        u4_2[0,:] = u4_2[numPoints,:]
        
        u1_2[1,:] = u1_2[numPoints + 1,:]
        u2_2[1,:] = u2_2[numPoints + 1,:]
        u3_2[1,:] = u3_2[numPoints + 1,:]
        u4_2[1,:] = u4_2[numPoints + 1,:]
        
        u1_2[numPoints + 2,:] = u1_2[2,:]
        u2_2[numPoints + 2,:] = u2_2[2,:]
        u3_2[numPoints + 2,:] = u3_2[2,:]
        u4_2[numPoints + 2,:] = u4_2[2,:]
        
        u1_2[numPoints + 3,:] = u1_2[3,:]
        u2_2[numPoints + 3,:] = u2_2[3,:]
        u3_2[numPoints + 3,:] = u3_2[3,:]
        u4_2[numPoints + 3,:] = u4_2[3,:]
        
    else: 	
        u1_2[0,:] = u1_2[2,:]
        u2_2[0,:] = u2_2[2,:]
        u3_2[0,:] = u3_2[2,:]
        u4_2[0,:] = u4_2[2,:]
    
        u1_2[1,:] = u1_2[0,:]
        u2_2[1,:] = u2_2[0,:]
        u3_2[1,:] = u3_2[0,:]
        u4_2[1,:] = u4_2[0,:]
        
        for i in range(numPoints + 2, len(u1_2[0])):
            u1_2[i,:] = u1_2[numPoints + 1,:]
            u2_2[i,:] = u2_2[numPoints + 1,:]
            u3_2[i,:] = u3_2[numPoints + 1,:]
            u4_2[i,:] = u4_2[numPoints + 1,:]        

    return u1_2, u2_2, u3_2, u4_2

def updateQuantitiesRungeKutta(task, timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, u1_2, u2_2, u3_2, u4_2, DivF1, DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4):
    '''
    Parameters
    ----------
    task :                  32-bit integer
                            The number of the problem with specified initial and boundry conditions
                            
    timeStep :              32-bit floating point number
                            An appropriate timestep that does not violate the Courant-Friedrich-Levy condition.
                            
    numPoints :             32-bit integer
                            The number of points in the spatial grid
    
    PeriodicX :             logical value of True if there are periodic 
                            boundry conditions in the x-direction and False otherwise 
                            
    PeriodicY :             logical value of True if there are periodic 
                            boundry conditions in the y-direction and False otherwise                        
        
    u1 :                    numpy array of 32-bit floats
                            Array of mass density values
                            
    u2 :                    numpy array of 32-bit floats
                            Array of x-momentum density values
    
    u3 :                    numpy array of 32-bit floats
                            Array of y-momentum density values
                            
    u4 :                    numpy array of 32-bit floats
                            Array of energy density values
                            
    u1_2 :                  numpy array of 32-bit floats
                            Array of mass density values to the second order
                           
    u2_2 :                  numpy array of 32-bit floats
                            Array of x-momentum density values to the second order
	   
    u3_2 :                  numpy array of 32-bit floats
                            Array of y-momentum density values to the second order
                           
    u4_2 :                  numpy array of 32-bit floats
                            Array of energy density values to the second order                        
                            
    DivF1 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-mass density divided by the 
                            spatial grid spacing
                              
    DivF2 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-momentum density divided by the 
                            spatial grid spacing
                            
    DivF3 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x&y-momentum density divided by the 
                            spatial grid spacing                        
        
    DivF4 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (left interface - right interface)
                            associated with the x-energy density divided by the 
                            spatial grid spacing
                            
    DivG1 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-mass density divided by the 
                            spatial grid spacing
                              
    DivG2 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y&x-momentum density divided by the 
                            spatial grid spacing
                            
    DivG3 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-momentum density divided by the 
                            spatial grid spacing                        
        
    DivG4 :                 numpy array of 32-bit floating point numbers
                            The change in interface flux (below interface - above interface)
                            associated with the y-energy density divided by the 
                            spatial grid spacing                        
    
    Returns
    -------
    u1 :                    numpy array of 32-bit floats
                            Array of updated mass density values to the third order
                            
    u2 :                    numpy array of 32-bit floats
                            Array of updated x-momentum density values to the third order
    
    u3 :                    numpy array of 32-bit floats
                            Array of updated y-momentum density values to the third order
                            
    u4 :                    numpy array of 32-bit floats
                            Array of updated energy density values to the third order
    
    '''
    u1[2: numPoints + 2, 2: numPoints + 2] = (1/3)*u1[2: numPoints + 2, 2: numPoints + 2] + (2/3)*u1_2[2: numPoints + 2, 2: numPoints + 2] + (2/3)*timeStep*(DivF1 + DivG1)
    u2[2: numPoints + 2, 2: numPoints + 2] = (1/3)*u2[2: numPoints + 2, 2: numPoints + 2] + (2/3)*u2_2[2: numPoints + 2, 2: numPoints + 2] + (2/3)*timeStep*(DivF2 + DivG2)
    u3[2: numPoints + 2, 2: numPoints + 2] = (1/3)*u3[2: numPoints + 2, 2: numPoints + 2] + (2/3)*u3_2[2: numPoints + 2, 2: numPoints + 2] + (2/3)*timeStep*(DivF3 + DivG3)
    u4[2: numPoints + 2, 2: numPoints + 2] = (1/3)*u4[2: numPoints + 2, 2: numPoints + 2] + (2/3)*u4_2[2: numPoints + 2, 2: numPoints + 2] + (2/3)*timeStep*(DivF4 + DivG4)

    if(task == 3):
        q1 = u1[int((numPoints + 2)/2)+1: numPoints + 2, int((numPoints + 2)/2)+1: numPoints + 2]
        q2 = u2[int((numPoints + 2)/2)+1: numPoints + 2, int((numPoints + 2)/2)+1: numPoints + 2]
        q3 = u3[int((numPoints + 2)/2)+1: numPoints + 2, int((numPoints + 2)/2)+1: numPoints + 2]
        q4 = u4[int((numPoints + 2)/2)+1: numPoints + 2, int((numPoints + 2)/2)+1: numPoints + 2]
        
        #reversed_array_columns = original_array[:, ::-1]
        u1[int((numPoints + 2)/2)+1: numPoints + 2, 2: int((numPoints + 2)/2)+1] = q1[:, ::-1] 
        u2[int((numPoints + 2)/2)+1: numPoints + 2, 2: int((numPoints + 2)/2)+1] = -q2[:, ::-1] 
        u3[int((numPoints + 2)/2)+1: numPoints + 2, 2: int((numPoints + 2)/2)+1] = q3[:, ::-1] 
        u4[int((numPoints + 2)/2)+1: numPoints + 2, 2: int((numPoints + 2)/2)+1] = q4[:, ::-1] 
        
        u1[2: int((numPoints + 2)/2)+1, int((numPoints + 2)/2)+1: numPoints + 2] = np.flipud(q1)
        u2[2: int((numPoints + 2)/2)+1, int((numPoints + 2)/2)+1: numPoints + 2] = np.flipud(q2) 
        u3[2: int((numPoints + 2)/2)+1, int((numPoints + 2)/2)+1: numPoints + 2] = -np.flipud(q3) 
        u4[2: int((numPoints + 2)/2)+1, int((numPoints + 2)/2)+1: numPoints + 2] = np.flipud(q4)
        
        u1[2: int((numPoints + 2)/2)+1, 2: int((numPoints + 2)/2)+1] = np.flipud(q1)[:, ::-1]
        u2[2: int((numPoints + 2)/2)+1, 2: int((numPoints + 2)/2)+1] = -np.flipud(q2)[:, ::-1]
        u3[2: int((numPoints + 2)/2)+1, 2: int((numPoints + 2)/2)+1] = -np.flipud(q3)[:, ::-1]
        u4[2: int((numPoints + 2)/2)+1, 2: int((numPoints + 2)/2)+1] = np.flipud(q4)[:, ::-1]
    
    
    if (PeriodicY==True):
        u1[0,:] = u1[numPoints,:]
        u2[0,:] = u2[numPoints,:]
        u3[0,:] = u3[numPoints,:]
        u4[0,:] = u4[numPoints,:]
        
        u1[1,:] = u1[numPoints + 1,:]
        u2[1,:] = u2[numPoints + 1,:]
        u3[1,:] = u3[numPoints + 1,:]
        u4[1,:] = u4[numPoints + 1,:]
        
        u1[numPoints + 2,:] = u1[2,:]
        u2[numPoints + 2,:] = u2[2,:]
        u3[numPoints + 2,:] = u3[2,:]
        u4[numPoints + 2,:] = u4[2,:]
        
        u1[numPoints + 3,:] = u1[3,:]
        u2[numPoints + 3,:] = u2[3,:]
        u3[numPoints + 3,:] = u3[3,:]
        u4[numPoints + 3,:] = u4[3,:]
        
    else: 	
        u1[0,:] = u1[2,:]
        u2[0,:] = u2[2,:]
        u3[0,:] = u3[2,:]
        u4[0,:] = u4[2,:]
    
        u1[1,:] = u1[0,:]
        u2[1,:] = u2[0,:]
        u3[1,:] = u3[0,:]
        u4[1,:] = u4[0,:]
        
        for i in range(numPoints + 2, len(u1[0])):
            u1[i,:] = u1[numPoints + 1,:]
            u2[i,:] = u2[numPoints + 1,:]
            u3[i,:] = u3[numPoints + 1,:]
            u4[i,:] = u4[numPoints + 1,:]
    
    if (PeriodicX==True):
        u1[:,0] = u1[:,numPoints]
        u2[:,0] = u2[:,numPoints]
        u3[:,0] = u3[:,numPoints]
        u4[:,0] = u4[:,numPoints]
        
        u1[:,1] = u1[:,numPoints + 1]
        u2[:,1] = u2[:,numPoints + 1]
        u3[:,1] = u3[:,numPoints + 1]
        u4[:,1] = u4[:,numPoints + 1]
        
        u1[:,numPoints + 2] = u1[:,2]
        u2[:,numPoints + 2] = u2[:,2]
        u3[:,numPoints + 2] = u3[:,2]
        u4[:,numPoints + 2] = u4[:,2]
        
        u1[:,numPoints + 3] = u1[:,3]
        u2[:,numPoints + 3] = u2[:,3]
        u3[:,numPoints + 3] = u3[:,3]
        u4[:,numPoints + 3] = u4[:,3]
        
    else:	
        u1[:,0] = u1[:,2]
        u2[:,0] = u2[:,2]
        u3[:,0] = u3[:,2]
        u4[:,0] = u4[:,2] 
    
        u1[:,1] = u1[:,0]
        u2[:,1] = u2[:,0]
        u3[:,1] = u3[:,0]
        u4[:,1] = u4[:,0]
        
        for i in range(numPoints + 2, len(u1[0])):
            u1[:,i] = u1[:,numPoints + 1]
            u2[:,i] = u2[:,numPoints + 1]
            u3[:,i] = u3[:,numPoints + 1]
            u4[:,i] = u4[:,numPoints + 1]

    return u1, u2, u3, u4    
        


