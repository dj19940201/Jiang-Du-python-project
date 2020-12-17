import numpy as np
from scipy.optimize import minimize
import os
# references are [1]https://github.com/keit0222/optimization-evaluation  [2]https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# https://github.com/keit0222/optimization-evaluation/blob/master/opteval/benchmark_func.py
# Goldstein-Price benchmark function 
def GoldsteinPrice(variables):
    tmp1 = (1+np.power(variables[0]+variables[1]+1,2)*(19-14*variables[0]+3*np.power(variables[0],2)-14*variables[1]+6*variables[0]*variables[1]+3*np.power(variables[1],2)))
    tmp2 = (30+(np.power(2*variables[0]-3*variables[1],2)*(18-32*variables[0]+12*np.power(variables[0],2)+48*variables[1]-36*variables[0]*variables[1]+27*np.power(variables[1],2))))
    return tmp1*tmp2

#start point
x0 = np.array([50, 20])

##### Class Schaffer function N.4 #####
def SchafferN4(variables):
    tmp1 = np.power(np.cos(np.sin(np.absolute(np.power(variables[0],2)-np.power(variables[1],2)))),2)-0.5
    tmp2 = np.power(1+0.001*(np.power(variables[0],2)+np.power(variables[1],2)),2)
    return 0.5+tmp1/tmp2




print ('results for Goldstein-Price function')

print ('Nelder-Mead algorithm')
res=minimize(GoldsteinPrice, x0, method='Nelder-Mead',
               options={'xatol': 1e-8, 'disp': True})

print ('Powell algorithm')
res=minimize(GoldsteinPrice, x0, method='Powell',
               options={ 'disp': True})

print('Broyden–Fletcher–Goldfarb–Shanno algorithm')
res=minimize(GoldsteinPrice, x0, method='BFGS',
               options={ 'disp': True})



print ('results for Five-well potential benchmark function')

print ('Nelder-Mead algorithm')
res=minimize(SchafferN4, x0, method='Nelder-Mead',
               options={'xatol': 1e-8, 'disp': True})

print ('Powell algorithm')
res=minimize(SchafferN4, x0, method='Powell',
               options={ 'disp': True})

print('Broyden–Fletcher–Goldfarb–Shanno algorithm')
res=minimize(SchafferN4, x0, method='BFGS',
               options={ 'disp': True})

