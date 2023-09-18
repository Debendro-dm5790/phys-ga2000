# -*- coding: utf-8 -*-
import pytest
import numpy as np
import quadratic

def test_quadratic():
    # Check the case from the problem
    arr = quadratic.quadratic(a=0.001, b=1000., c=0.001)
    assert (np.abs(arr[0] - (- 1.e-6)) < 1.e-10)
    assert (np.abs(arr[1] - (- 0.999999999999e+6)) < 1.e-10)
    
    # Check a related case to the problem
    arr2 = quadratic.quadratic(a=0.001, b=-1000., c=0.001)
    assert (np.abs(arr2[0] - (0.999999999999e+6)) < 1.e-10)
    assert (np.abs(arr2[1] - (1.e-6)) < 1.e-10)
    
    # Check a simpler case (note it requires the + solution first)
    arr3 = quadratic.quadratic(a=1., b=8., c=12.)
    assert (np.abs(arr3[0] - (- 2.)) < 1.e-10)
    assert (np.abs(arr3[1] - (- 6)) < 1.e-10)

