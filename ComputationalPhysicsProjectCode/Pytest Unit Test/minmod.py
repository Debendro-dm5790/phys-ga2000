import numpy as np

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

