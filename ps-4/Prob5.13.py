import numpy as np
import matplotlib.pyplot as plt
import math as m

'''
We define a function called H(n,x) which dermines the value of the n-th Hermite 
polynomial at x. We test to see whether out coded Hermite polynomials match the 
actual Hermite polynomials for the case n = 5. They indeed do and we store this
verification in a .png file called HermiteTest.png 
'''

def H(n,x):
    H_previous_previous = 1
    H_previous = 2*x
    H_current = 0
    
    if n == 0:
        H_current = 1
    elif n == 1:
        H_current = 2*x
    else:
        for i in range(n-1):
            H_current = 2*x*H_previous - 2*(i+1)*H_previous_previous
            H_previous_previous = H_previous
            H_previous = H_current
         
    return H_current

x_test = np.linspace(-5,5,100)
H_5_actual = 32*x_test**5 - 160*x_test**3 + 120*x_test
H_5_coded = H(5,x_test)

    
plt.plot(x_test, H_5_actual)
plt.plot(x_test, H_5_coded)
plt.xlabel('x')
plt.ylabel('H_5(x)')
plt.title('Comparing Actual H_5 with coded H_5')
plt.legend(['H_5_actual', 'H_5_coded'], loc = 'lower right')
plt.savefig('HermiteTest.png')
plt.show()

'''
Using the coded Hermite functions we plot the first four wavefunctions of the Harmonic 
osscillator in the range -4 < x < 4 and the 31st wavefunction in the range
-10 < x < 10. These correspond to n = 0, 1, 2, 3, and 30. We save the plots as .png
files.
'''

xList = np.linspace(-4,4,500)
n = [0,1,2,3,30]
wavefunction0 = (1/np.sqrt(2**n[0]*m.factorial(n[0])*np.sqrt(np.pi)))*np.exp(-1*0.5*xList**2)*H(n[0], xList)
wavefunction1 = (1/np.sqrt(2**n[1]*m.factorial(n[1])*np.sqrt(np.pi)))*np.exp(-1*0.5*xList**2)*H(n[1], xList)
wavefunction2 = (1/np.sqrt(2**n[2]*m.factorial(n[2])*np.sqrt(np.pi)))*np.exp(-1*0.5*xList**2)*H(n[2], xList)
wavefunction3 = (1/np.sqrt(2**n[3]*m.factorial(n[3])*np.sqrt(np.pi)))*np.exp(-1*0.5*xList**2)*H(n[3], xList)

plt.plot(xList, wavefunction0)
plt.plot(xList, wavefunction1)
plt.plot(xList, wavefunction2)
plt.plot(xList, wavefunction3)
plt.xlabel('Position')
plt.ylabel('Wavefunction Value')
plt.title('The First Four Wavefunctions of the Harmonic Oscillator')
plt.legend(['n = 0', 'n = 1', 'n = 2', 'n = 3'])
plt.savefig('FirstFourWavefunctions.png')
plt.show()

xList2 = np.linspace(-10,10,500)

wavefunction30 = (1/np.sqrt(2**n[4]*m.factorial(n[4])*np.sqrt(np.pi)))*np.exp(-1*0.5*xList2**2)*H(n[4], xList2)

plt.plot(xList2, wavefunction30)
plt.xlabel('Position')
plt.ylabel('Wavefunction Value')
plt.title('The n = 30 Wavefunction of the Harmonic Oscillator')
plt.savefig('Wavefunction30.png')
plt.show()


