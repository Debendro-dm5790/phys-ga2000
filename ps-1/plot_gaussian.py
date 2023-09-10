import numpy as np
import matplotlib.pyplot as plt

'''
This codes creates and plots a normalized Gaussian with mean 0
and standard deviation 3 over the range [-10,10]. The plot is
saved as an .png file called gaussian.png
'''

'''
Define integer constants mu and sigma representing the mean and standard deviation,
respectively.
'''
mu = 0
sigma = 3

'''
Create the data list and the density of probabilities which each contain 1000 elements. It can be shown that the gaussian plots of single-precision and double-precision dataList are basically identical and so it is better to make dataList an array of float32 and not waste memory.  
'''
dataList = np.linspace(start = -10, stop = 10, num = 1000, dtype = np.float32)
DoP = (1/(sigma*np.sqrt(2*np.pi)))*np.e**(-0.5*((dataList - mu)/sigma)**2)

'''
Plot the data with the axes labeled and title given. Save the plot as a .png file before showing it; otherwise, a blank file will be saved.
'''
plt.plot(dataList, DoP)
plt.xlabel("Data Values")
plt.xlim(-10,10)
plt.ylabel("Density of Probability")
plt.title("Gaussian Distribution with Mean 0 and Standard Deviation 3")
plt.savefig("gaussian.png.")
plt.show()

