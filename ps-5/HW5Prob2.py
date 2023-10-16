import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
We first use Python's pandas module to read the data file signal.data and convert 
it into a .txt file using the to_csv() method. Yes this is possible! We then open the 
text file signal.txt and fill the time and signal values into the Python list time and 
signal. We then convert these lists into numpy arrays called time and signal. We then plot
the signal and save the plot as a .png file called SignalPlot.png.
'''
df = pd.read_csv('signal.dat', delimiter='\t')
df.to_csv('signal.txt', index=None)

time = [] 
signal = [] 

counter = 0
for line in open('signal.txt', 'r'): 
    lines = [i for i in line.split()] 
    
    if counter != 0:
        time.append(np.float32(lines[1]))
        signal.append(np.float32(lines[3]))
        
    counter += 1
    
time = np.array(time, dtype = np.float32)
signal = np.array(signal, dtype = np.float32)

plt.plot(time, signal, '.')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Signal Plot over Time')
plt.savefig('SignalPlot.png')
plt.show()

'''
We then implement singular value decomposition to determine the best cubic fit to the signal
data. We first re-express the time data by subtracting from each time value the mean and divide
that by the standard deviation of the time data. The adjusted time values are stored in a 
numpy array called tp. We then implement a design matrix A whose columns are the  functions
1, tp, tp^2, and tp^3 and perform singular value decomposition to determine the coefficients in 
front of 1, tp, tp^2, and tp^3. We also print out the singular values and the condition number,
which the largest singular value divided by the smallest one.  
'''
tp = (time - time.mean())/time.std()

A = np.zeros((len(time), 4))
A[:, 0] = 1.
A[:, 1] = (tp)
A[:, 2] = (tp)**2
A[:, 3] = (tp)**3

(U,D,VT) = np.linalg.svd(A, full_matrices = False)
A_inv = VT.transpose().dot(np.diag(1./D)).dot(U.transpose())
C = A_inv.dot(signal)

print('Coefficients of 1, t, t^2, and t^3 of best fit are')
print(C)
print('The singular values are')
print(D)
print('The condition number is')
print(max(D)/min(D))

'''
We then superimpose the cubic model on the original signal data and plot the residuals. The 
residual plot shows a sinusoidal pattern instead of a randomized scattering of residuals, suggesting
that the cubic model is poor. We also computed the chi-square statistic and found it to be about 
4582.7. A large chi-square suggests that the model is poor. 
'''

signal_model = A.dot(C)

plt.plot(time, signal, '.')
plt.plot(time, signal_model, '.')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Signal Plot over Time')
plt.legend(['Orignal Signal Data', 'Cubic Model'])
plt.savefig('SignalCubicModelPlot.png')
plt.show()

plt.plot(time, signal - signal_model, '.', color = 'red')
plt.xlabel('Time')
plt.ylabel('Residual for Cubic Model: Actual - Model')
plt.title('Signal Residuals')
plt.savefig('SignalCubicModelResidualPlot.png')
plt.show()

chiSquare1 = sum((signal - signal_model)**2/signal_model)
print('Chi-square is ' + str(chiSquare1))

'''
We then implement a polynomial model of degree 24 by redefining the design matrix and using the 
same adjusted time values. We plot the original signal and the new model. From this plot, we see
that the model roughly captures the ossilating behavior of the signal. From the residual plot
for this new model, we see that the residuals have no pattern, a sign that the model is not as 
poor as the previous one. Superimposing the cubic residuals with the new residuals, we see that the 
new model's residuals are usually less than those of the cubic model. We also determine the 
condition number, which is approximately 6,267,712,864.4. Although this condition number is large
it does not exceed the dynamic range of the 32-bit floating point numbers. Thus the condition 
number is OK. We also compute and print the chi-square statistic, which is about 3720.87. Although
it is larger, the statistic for this model is less than that of the cubic model. '
'''

Anew = np.zeros((len(tp), 25))
Anew[:, 0] = 1.
Anew[:, 1] = (tp)
Anew[:, 2] = (tp)**2
Anew[:, 3] = (tp)**3
Anew[:, 4] = (tp)**4
Anew[:, 5] = (tp)**5
Anew[:, 6] = (tp)**6
Anew[:, 7] = (tp)**7
Anew[:, 8] = (tp)**8
Anew[:, 9] = (tp)**9
Anew[:, 10] = (tp)**10
Anew[:, 11] = (tp)**11
Anew[:, 12] = (tp)**12
Anew[:, 13] = (tp)**13
Anew[:, 14] = (tp)**14
Anew[:, 15] = (tp)**15
Anew[:, 16] = (tp)**16
Anew[:, 17] = (tp)**17
Anew[:, 18] = (tp)**18
Anew[:, 19] = (tp)**19
Anew[:, 20] = (tp)**20
Anew[:, 21] = (tp)**21
Anew[:, 22] = (tp)**22
Anew[:, 23] = (tp)**23
Anew[:, 24] = (tp)**24


(U2,D2,V2T) = np.linalg.svd(Anew, full_matrices = False)
Anew_inv = V2T.transpose().dot(np.diag(1./D2)).dot(U2.transpose())
C2 = Anew_inv.dot(signal)

print('Coefficients of  best fit are')
print(C2)
print('The singular values are')
print(D2)
print('The condition number is')
print(max(D2)/min(D2))

signal_model2 = Anew.dot(C2)

plt.plot(time, signal, '.')
plt.plot(time, signal_model2, '.')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Signal Plot over Time')
plt.legend(['Orignal Signal Data', 'Degree 24 Polynomial Model'])
plt.savefig('SignalDegree24ModelPlot.png')
plt.show()

plt.plot(time, signal - signal_model2, '.', color = 'red')
plt.xlabel('Time')
plt.ylabel('Residual for Degree 24 Model: Actual - Model')
plt.title('Signal Residuals')
plt.savefig('SignalDegree24ModelResidualPlot.png')
plt.show()

chiSquare2 = sum((signal - signal_model2)**2/signal_model)
print('Chi-square is ' + str(chiSquare2))

plt.plot(time, signal - signal_model, '.', color = 'red')
plt.plot(time, signal - signal_model2, '.', color = 'blue')
plt.xlabel('Time')
plt.ylabel('Comparison of the Residuals')
plt.title('Signal Residuals')
plt.legend(['Cubic Residuals', 'Degree 24 Residuals'])
plt.savefig('SignalResidualComparisonPlot.png')
plt.show()




