import numpy as np

def speedOfSound(index1, index2, gamma, u1, u2, u3, u4): 
    '''
    	A function that computes and returns the speed of sound
    
    Parameters
    ----------
    index1 :            32-bit integer
                        The first index of the u1, u2, u3, u4 array. Needed to compute the sound speed
                        
    index2 :            32-bit integer
                        The second index of the u1, u2, u3, u4 array. Needed to compute the sound speed                    
                    
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
    The expression for the speed of sound
    
    '''

    return np.sqrt(gamma*(gamma - 1)*((u4[index1, index2]/u1[index1, index2]) - 0.5*(u2[index1, index2]**2/u1[index1, index2]**2) \
    - 0.5*(u3[index1, index2]**2/u1[index1, index2]**2))) # this is c

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
    
    LeftOfInterfaceRho_C1 = np.zeros((numInterfacePoints, numInterfacePoints), dtype = np.float32)
    LeftOfInterfaceVelx_C2 = LeftOfInterfaceRho_C1.copy()
    LeftOfInterfaceVely_C3 = LeftOfInterfaceRho_C1.copy()
    LeftOfInterfacePress_C4 = LeftOfInterfaceRho_C1.copy()
    
    RightOfInterfaceRho_C1 = LeftOfInterfaceRho_C1.copy()
    RightOfInterfaceVelx_C2 = LeftOfInterfaceRho_C1.copy()
    RightOfInterfaceVely_C3 = LeftOfInterfaceRho_C1.copy()
    RightOfInterfacePress_C4 = LeftOfInterfaceRho_C1.copy()
    
    BottomOfInterfaceRho_C1 = LeftOfInterfaceRho_C1.copy()
    BottomOfInterfaceVelx_C2 = LeftOfInterfaceRho_C1.copy()
    BottomOfInterfaceVely_C3 = LeftOfInterfaceRho_C1.copy()
    BottomOfInterfacePress_C4 = LeftOfInterfaceRho_C1.copy()
    
    TopOfInterfaceRho_C1 = LeftOfInterfaceRho_C1.copy()
    TopOfInterfaceVelx_C2 = LeftOfInterfaceRho_C1.copy()
    TopOfInterfaceVely_C3 = LeftOfInterfaceRho_C1.copy()
    TopOfInterfacePress_C4 = LeftOfInterfaceRho_C1.copy()
    
        
    for i in range(numInterfacePoints):
        for j in range(numInterfacePoints):
            LeftOfInterfaceRho_C1[i,j] = C1[i,j+1] + 0.5*minmod(theta*(C1[i,j+1] - C1[i,j]), 0.5*(C1[i,j+2] - C1[i,j]), theta*(C1[i,j+2] - C1[i,j+1]))
            LeftOfInterfaceVelx_C2[i,j] = C2[i,j+1] + 0.5*minmod(theta*(C2[i,j+1] - C2[i,j]), 0.5*(C2[i,j+2] - C2[i,j]), theta*(C2[i,j+2] - C2[i,j+1]))
            LeftOfInterfaceVely_C3[i,j] = C3[i,j+1] + 0.5*minmod(theta*(C3[i,j+1] - C3[i,j]), 0.5*(C3[i,j+2] - C3[i,j]), theta*(C3[i,j+2] - C3[i,j+1]))
            LeftOfInterfacePress_C4[i,j] = C4[i,j+1] + 0.5*minmod(theta*(C4[i,j+1] - C4[i,j]), 0.5*(C4[i,j+2] - C4[i,j]), theta*(C4[i,j+2] - C4[i,j+1]))
            
            RightOfInterfaceRho_C1[i,j] = C1[i,j+2] + 0.5*minmod(theta*(C1[i,j+2] - C1[i,j+1]), 0.5*(C1[i,j+3] - C1[i,j+1]), theta*(C1[i,j+3] - C1[i,j+2]))
            RightOfInterfaceVelx_C2[i,j] = C2[i,j+2] + 0.5*minmod(theta*(C2[i,j+2] - C2[i,j+1]), 0.5*(C2[i,j+3] - C2[i,j+1]), theta*(C2[i,j+3] - C2[i,j+2]))
            RightOfInterfaceVely_C3[i,j] = C3[i,j+2] + 0.5*minmod(theta*(C3[i,j+2] - C3[i,j+1]), 0.5*(C3[i,j+3] - C3[i,j+1]), theta*(C3[i,j+3] - C3[i,j+2]))
            RightOfInterfacePress_C4[i,j] = C4[i,j+2] + 0.5*minmod(theta*(C4[i,j+2] - C4[i,j+1]), 0.5*(C4[i,j+3] - C4[i,j+1]), theta*(C4[i,j+3] - C4[i,j+2]))
            
            BottomOfInterfaceRho_C1[i,j] = C1[i+1,j] + 0.5*minmod(theta*(C1[i+1,j] - C1[i,j]), 0.5*(C1[i+2,j] - C1[i,j]), theta*(C1[i+2,j] - C1[i+1,j]))
            BottomOfInterfaceVelx_C2[i,j] = C2[i+1,j] + 0.5*minmod(theta*(C2[i+1,j] - C2[i,j]), 0.5*(C2[i+2,j] - C2[i,j]), theta*(C2[i+2,j] - C2[i+1,j]))
            BottomOfInterfaceVely_C3[i,j] = C3[i+1,j] + 0.5*minmod(theta*(C3[i+1,j] - C3[i,j]), 0.5*(C3[i+2,j] - C3[i,j]), theta*(C3[i+2,j] - C3[i+1,j]))
            BottomOfInterfacePress_C4[i,j] = C4[i+1,j] + 0.5*minmod(theta*(C4[i+1,j] - C4[i,j]), 0.5*(C4[i+2,j] - C4[i,j]), theta*(C4[i+2,j] - C4[i+1,j]))
            
            TopOfInterfaceRho_C1[i,j] = C1[i+2,j] + 0.5*minmod(theta*(C1[i+2,j] - C1[i+1,j]), 0.5*(C1[i+3,j] - C1[i+1,j]), theta*(C1[i+3,j] - C1[i+2,j]))
            TopOfInterfaceVelx_C2[i,j] = C2[i+2,j] + 0.5*minmod(theta*(C2[i+2,j] - C2[i+1,j]), 0.5*(C2[i+3,j] - C2[i+1,j]), theta*(C2[i+3,j] - C2[i+2,j]))
            TopOfInterfaceVely_C3[i,j] = C3[i+2,j] + 0.5*minmod(theta*(C3[i+2,j] - C3[i+1,j]), 0.5*(C3[i+3,j] - C3[i+1,j]), theta*(C3[i+3,j] - C3[i+2,j]))
            TopOfInterfacePress_C4[i,j] = C4[i+2,j] + 0.5*minmod(theta*(C4[i+2,j] - C4[i+1,j]), 0.5*(C4[i+3,j] - C4[i+1,j]), theta*(C4[i+3,j] - C4[i+2,j]))
    
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
      
    F1_HLL_RungeKutta = np.zeros((numInterfacePoints, numInterfacePoints), dtype = np.float32)
    F2_HLL_RungeKutta = F1_HLL_RungeKutta.copy()
    F3_HLL_RungeKutta = F1_HLL_RungeKutta.copy()
    F4_HLL_RungeKutta = F1_HLL_RungeKutta.copy()
    G1_HLL_RungeKutta = F1_HLL_RungeKutta.copy()
    G2_HLL_RungeKutta = F1_HLL_RungeKutta.copy()
    G3_HLL_RungeKutta = F1_HLL_RungeKutta.copy()
    G4_HLL_RungeKutta = F1_HLL_RungeKutta.copy()
    
    for i in range(numInterfacePoints):
        for j in range(numInterfacePoints):
            lambdaPlusLeft = (u2[i,j]/u1[i,j]) + speedOfSound(i, j, gamma, u1, u2, u3, u4)
            lambdaMinusLeft = (u2[i,j]/u1[i,j]) - speedOfSound(i, j, gamma, u1, u2, u3, u4)
            lambdaPlusRight = (u2[i,j+1]/u1[i,j+1]) + speedOfSound(i, j, gamma, u1, u2, u3, u4)
            lambdaMinusRight = (u2[i,j+1]/u1[i,j+1]) - speedOfSound(i, j, gamma, u1, u2, u3, u4)
            
            lambdaPlusBottom = (u3[i,j]/u1[i,j]) + speedOfSound(i, j, gamma, u1, u2, u3, u4)
            lambdaMinusBottom = (u3[i,j]/u1[i,j]) - speedOfSound(i, j, gamma, u1, u2, u3, u4)
            lambdaPlusTop = (u3[i+1,j]/u1[i+1,j]) + speedOfSound(i, j, gamma, u1, u2, u3, u4)
            lambdaMinusTop = (u3[i+1,j]/u1[i+1,j]) - speedOfSound(i, j, gamma, u1, u2, u3, u4)
            
            alphaPlus = max([0, lambdaPlusLeft, lambdaPlusRight])
            alphaMinus = max([0, -1*lambdaMinusLeft, -1*lambdaMinusRight])
            
            betaPlus = max([0, lambdaPlusBottom, lambdaPlusTop])
            betaMinus = max([0, -1*lambdaMinusBottom, -1*lambdaMinusTop])
            
            if (alphaPlus==0) and (alphaMinus==0) and (betaPlus==0) and (betaMinus==0):		
                print (lambdaPlusLeft, lambdaMinusLeft, lambdaPlusRight, lambdaMinusRight)
                print (lambdaPlusBottom, lambdaMinusBottom, lambdaPlusTop, lambdaMinusTop)
                #print (u2[i,j],u1[i,j], speedOfSound(i, j, gamma, u1, u2, u3, u4))
                raise Exception("error, division by zero")
            
            possibleTimeStep = np.float32(deltaX/max(alphaPlus, alphaMinus, betaPlus, betaMinus))
            
            
            if (i == 0) and (j == 0):
                timeStepCandidate = possibleTimeStep
            else: 
                if possibleTimeStep < timeStepCandidate:
                    timeStepCandidate = possibleTimeStep
                    
            F1_HLL_RungeKutta[i,j] = ((alphaPlus*LeftOfInterfaceFlux_1[i,j]) + (alphaMinus*RightOfInterfaceFlux_1[i,j]) - (alphaPlus*alphaMinus*(u1[i,j+2] - u1[i,j+1])))/(alphaPlus + alphaMinus)
            F2_HLL_RungeKutta[i,j] = ((alphaPlus*LeftOfInterfaceFlux_2[i,j]) + (alphaMinus*RightOfInterfaceFlux_2[i,j]) - (alphaPlus*alphaMinus*(u2[i,j+2] - u2[i,j+1])))/(alphaPlus + alphaMinus)
            F3_HLL_RungeKutta[i,j] = ((alphaPlus*LeftOfInterfaceFlux_3[i,j]) + (alphaMinus*RightOfInterfaceFlux_3[i,j]) - (alphaPlus*alphaMinus*(u3[i,j+2] - u3[i,j+1])))/(alphaPlus + alphaMinus)
            F4_HLL_RungeKutta[i,j] = ((alphaPlus*LeftOfInterfaceFlux_4[i,j]) + (alphaMinus*RightOfInterfaceFlux_4[i,j]) - (alphaPlus*alphaMinus*(u4[i,j+2] - u4[i,j+1])))/(alphaPlus + alphaMinus)
            
            G1_HLL_RungeKutta[i,j] = ((betaPlus*BottomOfInterfaceFlux_1[i,j]) + (betaMinus*TopOfInterfaceFlux_1[i,j]) - (betaPlus*betaMinus*(u1[i+2,j] - u1[i+1,j])))/(betaPlus + betaMinus)
            G2_HLL_RungeKutta[i,j] = ((betaPlus*BottomOfInterfaceFlux_2[i,j]) + (betaMinus*TopOfInterfaceFlux_2[i,j]) - (betaPlus*betaMinus*(u2[i+2,j] - u2[i+1,j])))/(betaPlus + betaMinus)
            G3_HLL_RungeKutta[i,j] = ((betaPlus*BottomOfInterfaceFlux_3[i,j]) + (betaMinus*TopOfInterfaceFlux_3[i,j]) - (betaPlus*betaMinus*(u3[i+2,j] - u3[i+1,j])))/(betaPlus + betaMinus)
            G4_HLL_RungeKutta[i,j] = ((betaPlus*BottomOfInterfaceFlux_4[i,j]) + (betaMinus*TopOfInterfaceFlux_4[i,j]) - (betaPlus*betaMinus*(u4[i+2,j] - u4[i+1,j])))/(betaPlus + betaMinus)
    
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
    DivF1 = np.zeros((numPoints, numPoints), dtype = np.float32)
    DivF2 = DivF1.copy()
    DivF3 = DivF1.copy()
    DivF4 = DivF1.copy()
    DivG1 = DivF1.copy()
    DivG2 = DivF1.copy()
    DivG3 = DivF1.copy()
    DivG4 = DivF1.copy()
    
    for i in range(numPoints):
        for j in range(numPoints):
            DivF1[i, j] = -1*(F1_HLL[i, j+1] - F1_HLL[i, j])/deltaX
            DivF2[i, j] = -1*(F2_HLL[i, j+1] - F2_HLL[i, j])/deltaX
            DivF3[i, j] = -1*(F3_HLL[i, j+1] - F3_HLL[i, j])/deltaX 
            DivF4[i, j] = -1*(F4_HLL[i, j+1] - F4_HLL[i, j])/deltaX 
            DivG1[i, j] = -1*(G1_HLL[i+1, j] - G1_HLL[i, j])/deltaX 
            DivG2[i, j] = -1*(G2_HLL[i+1, j] - G2_HLL[i, j])/deltaX 
            DivG3[i, j] = -1*(G3_HLL[i+1, j] - G3_HLL[i, j])/deltaX
            DivG4[i, j] = -1*(G4_HLL[i+1, j] - G4_HLL[i, j])/deltaX
        
    return DivF1, DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4
    
def updateQuantitiesRungeKutta(timeStep, numPoints, PeriodicX, PeriodicY, u1, u2, u3, u4, DivF1, DivF2, DivF3, DivF4, DivG1, DivG2, DivG3, DivG4):
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
    u1 :                    numpy array of 32-bit floats
                            Array of updated mass density values
                            
    u2 :                    numpy array of 32-bit floats
                            Array of updated x-momentum density values
    
    u3 :                    numpy array of 32-bit floats
                            Array of updated y-momentum density values
                            
    u4 :                    numpy array of 32-bit floats
                            Array of updated energy density values
    
    '''
	
    u1_1 = np.zeros((len(range(2, numPoints + 2)),len(range(2, numPoints + 2))), dtype = np.float32)
    u1_2 = u1_1.copy()
    u2_1 = u1_1.copy()
    u2_2 = u1_1.copy()
    u3_1 = u1_1.copy()
    u3_2 = u1_1.copy()
    u4_1 = u1_1.copy()
    u4_2 = u1_1.copy()
    
    u1_1 = u1[2: numPoints + 2, 2: numPoints + 2] + timeStep*(DivF1 + DivG1)
    u1_2 = 0.75*u1[2: numPoints + 2, 2: numPoints + 2] + 0.25*u1_1 + 0.25*timeStep*(DivF1 + DivG1)
    u1[2: numPoints + 2, 2: numPoints + 2] = (1/3)*u1[2: numPoints + 2, 2: numPoints + 2] + (2/3)*u1_2 + (2/3)*timeStep*(DivF1 + DivG1)
    
    u2_1 = u2[2: numPoints + 2, 2: numPoints + 2] + timeStep*(DivF2 + DivG2)
    u2_2 = 0.75*u2[2: numPoints + 2, 2: numPoints + 2] + 0.25*u2_1 + 0.25*timeStep*(DivF2 + DivG2)
    u2[2: numPoints + 2, 2: numPoints + 2] = (1/3)*u2[2: numPoints + 2, 2: numPoints + 2] + (2/3)*u2_2 + (2/3)*timeStep*(DivF2 + DivG2)
    
    u3_1 = u3[2: numPoints + 2, 2: numPoints + 2] + timeStep*(DivF3 + DivG3)
    u3_2 = 0.75*u3[2: numPoints + 2, 2: numPoints + 2] + 0.25*u3_1 + 0.25*timeStep*(DivF3 + DivG3)
    u3[2: numPoints + 2, 2: numPoints + 2] = (1/3)*u3[2: numPoints + 2, 2: numPoints + 2] + (2/3)*u3_2 + (2/3)*timeStep*(DivF3 + DivG3)
    
    u4_1 = u4[2: numPoints + 2, 2: numPoints + 2] + timeStep*(DivF4 + DivG4)
    u4_2 = 0.75*u4[2: numPoints + 2, 2: numPoints + 2] + 0.25*u4_1 + 0.25*timeStep*(DivF4 + DivG4)
    u4[2: numPoints + 2, 2: numPoints + 2] = (1/3)*u4[2: numPoints + 2, 2: numPoints + 2] + (2/3)*u4_2 + (2/3)*timeStep*(DivF4 + DivG4)
    
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
        


