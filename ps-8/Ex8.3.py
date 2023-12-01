import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

def pend(vars, t, sigma, b, r):
    x,y,z = vars
    f = [sigma*(y - x), r*x - y - x*z, x*y - b*z]
    return f
   
sigma = 10
r = 28
b = 8/3

f0 = [0,1,0]

t = np.linspace(0, 50, 250)

sol = odeint(pend, f0, t, args=(sigma, b, r))

plt.plot(t, sol[:, 1], label = 'y(t)')
plt.legend(loc='best')
plt.xlabel('Time (seconds)')
plt.ylabel('Function Value of y(t)')
plt.title('Plot of Function y(t)')
plt.grid()
plt.savefig('YPlot.png')
plt.show()

plt.plot(sol[:, 0], sol[:, 2], label = 'z(x)')
plt.legend(loc='best')
plt.xlabel('Function Value of X(t)')
plt.ylabel('Function Value of Z(t)')
plt.title('Plot of Function z(x)')
plt.grid()
plt.savefig('StrangeAttractor.png')
plt.show()