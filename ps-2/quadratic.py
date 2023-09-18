# -*- coding: utf-8 -*-
"""
Quadratic Root Finder Using the Second Method
"""
import numpy as np

def quadratic(a,b,c):
  root1 = (-1*b + np.sqrt(b**2 - 4*a*c))/(2*a)
  root2 = (-1*b - np.sqrt(b**2 - 4*a*c))/(2*a)
  return [root1, root2]

