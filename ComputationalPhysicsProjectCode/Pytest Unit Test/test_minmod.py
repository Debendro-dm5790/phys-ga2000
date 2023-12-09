import pytest

import numpy as np
import minmod

def test_minmod():
    '''
    Here we test a case when all three inputs are positive
    '''
    output1 = minmod.minmod(1,1,1)
    assert(np.abs(output1 - 1) < 1e-8)
    
    '''
    Here we test a case when the first two terms, x and y, differ in sign.
    Minmod's |sgn(x) + sgn(y)| should be 0 here
    '''
    output2 = minmod.minmod(1,-2,1)
    assert(np.abs(output2 - 0) < 1e-8)
    
    '''
    Here we test a case when the last two terms, y and z, differ in sign.
    Minmod's (sgn(x) + sgn(z)) should be 0 here
    '''
    output3 = minmod.minmod(-5,-1,6)
    assert(np.abs(output3 - 0) < 1e-8)
    
    '''
    Here we test a case when all three inputs are negative
    '''
    output4 = minmod.minmod(-1,-1,-1)
    assert(np.abs(output4 + 1) < 1e-8)