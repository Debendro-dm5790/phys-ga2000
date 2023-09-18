# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:15:51 2023

@author: mooke
"""

import numpy as np

def quadraticStandard(a,b,c):
  root1 = (-1*b + np.sqrt(b**2 - 4*a*c))/(2*a)
  root2 = (-1*b - np.sqrt(b**2 - 4*a*c))/(2*a)
  return [root1, root2]

def quadraticNew(a,b,c):
  root1 = (2*c)/(-1*b - np.sqrt(b**2 - 4*a*c))
  root2 = (2*c)/(-1*b + np.sqrt(b**2 - 4*a*c))
  return [root1, root2]

expectedRootPlus = -1e-6
expectedRootMinus = -1e6

standardRootPlus, standardRootMinus = quadraticStandard(a=0.001,b=1000,c=0.001)
equivRootPlus, equivRootMinus = quadraticNew(a=0.001,b=1000,c=0.001)

percentErrorStandardRootPlus = np.abs(((standardRootPlus - expectedRootPlus)/expectedRootPlus)*100)
percentErrorStandardRootMinus = np.abs(((standardRootMinus - expectedRootMinus)/expectedRootMinus)*100)
percentErrorEquivRootPlus = np.abs(((equivRootPlus - expectedRootPlus)/expectedRootPlus)*100)
percentErrorEquivRootMinus = np.abs(((equivRootMinus - expectedRootMinus)/expectedRootMinus)*100)

print("The roots obtained by the standard formula are: ")
print(standardRootPlus, standardRootMinus)
print("and the corresponding percent errors are: ")
print(str(percentErrorStandardRootPlus) + " % and " + str(percentErrorStandardRootMinus) + ' %')
print('')
print("The roots obtained by the equivalent formula are: ")
print(equivRootPlus, equivRootMinus)
print("and the corresponding percent errors are: ")
print(str(percentErrorEquivRootPlus) + " % and " + str(percentErrorEquivRootMinus) + ' %')

absoluteErrorStandardRootPlus = np.abs(standardRootPlus - expectedRootPlus)
absoluteErrorStandardRootMinus = np.abs(standardRootMinus - expectedRootMinus)
absoluteErrorEquivRootPlus = np.abs(equivRootPlus - expectedRootPlus)
absoluteErrorEquivRootMinus = np.abs(equivRootMinus - expectedRootMinus)

print('')

print("The roots obtained by the standard formula are: ")
print(standardRootPlus, standardRootMinus)
print("and the corresponding absolute errors are: ")
print(str(absoluteErrorStandardRootPlus) + " and " + str(absoluteErrorStandardRootMinus))
print('')
print("The roots obtained by the equivalent formula are: ")
print(equivRootPlus, equivRootMinus)
print("and the corresponding absolute errors are: ")
print(str(absoluteErrorEquivRootPlus) + " and " + str(absoluteErrorEquivRootMinus))
