import matplotlib.pyplot as plt
import numpy as np

'''
Implement a N x N grid in the region -2 <= x <= 2 and -2 <= y <= 2
'''
N = 600
x = np.float32(np.linspace(start = -2, stop = 2, num = N, dtype = np.float32))
y = np.float32(np.linspace(start = -2, stop = 2, num = N, dtype = np.float32))

'''
Initialize arrays cList, the array of all c values which are just the 
grid points x + iy, and boolean array, which will contain 1 (0) if the c 
value is (is not) in the Mandelbrot set. 
'''
cList = np.array([])
booleanArray = np.array([])

for i in x:
  for k in y:
    cList = np.append(cList, i + 1j*k)
    booleanArray = np.append(booleanArray, True)
    
'''
Define magLimit, which is the maximum allowed limit of the magnitude after
each iteration.
'''
magLimit = 2

'''
Initialize zlist, the array of all resulting complex numbers after each
iteration of z = z^2 + c. This array will get updated.
'''
zList = np.float32(np.zeros(len(cList)))

'''
Begin the iterations
'''

for i in range(100):
  zList = zList*zList + cList
  '''
  The elements of rightWrongMag are 0 (1) if the transformation of c has 
  magnitude greater than (less than or equal to) the magnitude limit 2.
  '''
  rightWrongMag = np.absolute(zList) <= magLimit
  
  '''
  Transformations whose magnitudes are greater than 2 must not be considered
  in future iterations. This is done by setting the correspong z-values and 
  c-values to 0 so that any future iterations of z = z^2 + c on these values
  have no effect and by setting the corresponding boolean value in 
  booleanArray to 0.
  '''
  for j in range(len(zList)):
    if rightWrongMag[j] == 0:
      zList[j] = np.float32(0)
      cList[j] = np.float32(0)
      booleanArray[j] = False
  
'''
The first N elements of booleanArray before reshaping correspong to 
the x = -2 and the last N elements correspond to x = +2. Therefore, after
reshaping the boolean array, the first element is an array of N elements 
corresponding to x = -2 with -2 < y < 2 and the last element is an array 
of N elements corresponding to x = 2 with -2 < y < 2. We can think of 
the reshaped array pictorally as as a rectangle whose left horizontal 
side goes from negative x (i.e. the top) to positive x (i.e. the bottom)
and whose upper vertical side goes from negative y (i.e. the left) 
to positive y.
'''
booleanArray = booleanArray.reshape(N,N)

'''
Collect the c's, i.e. x+iy, that are and are not in the Mandelbrot set 
'''
xin = np.array([])
yin = np.array([])
xout = np.array([])
yout = np.array([])

for i in range(N):
    for j in range(N):
        if booleanArray[i][j] == 0:
            xout = np.append(xout, x[i])
            yout = np.append(yout, y[j])
        else:
            xin = np.append(xin, x[i])
            yin = np.append(yin, y[j])


plt.plot(xin, yin, '.', color = 'b')
plt.plot(xout, yout, '.', color = 'y')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel('x, Real Part')
plt.ylabel('y, Imaginary Part')
plt.title('The Mandelbrot Set')
plt.legend(['In Set', 'Not In Set'], loc = 'lower right')
plt.savefig('Mandelbrot.png')
plt.show()

'''
Make a fancier density plot with Pyplot's pcolormesh and zoom into the picture
'''
plt.pcolormesh(x,y,booleanArray.T, shading = 'auto')
plt.xlim(-2,1)
plt.ylim(-1.25,1.25)
plt.xlabel('x, Real Part')
plt.ylabel('y, Imaginary Part')
plt.title('The Mandelbrot Set')
plt.savefig('Mandelbrot Density Plot.png')
plt.show()
