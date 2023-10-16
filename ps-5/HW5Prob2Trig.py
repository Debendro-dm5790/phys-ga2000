import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

'''
We now determine and subtract off any linear trend by using singular value decomposition
'''
A = np.zeros((len(time), 2))
A[:, 0] = 1.
A[:, 1] = time
(U, D, VT) = np.linalg.svd(A, full_matrices=False)
Ainv = VT.transpose().dot(np.diag(1. / D)).dot(U.transpose())
C = Ainv.dot(signal)
signal_lin_model = A.dot(C) 

plt.plot(time, signal, '.')
plt.plot(time, signal_lin_model, '.')
plt.xlabel('Time')
plt.ylabel('Signal and Model Values')
plt.legend(['Original Signal', 'Linear Model'])
plt.title('Determining Any Linear Trend in the Signal')
plt.savefig('SignalLinearTrend.png')
plt.show()

NoLinTrend = signal - signal_lin_model
plt.plot(time, NoLinTrend, '.')
plt.xlabel('Time')
plt.ylabel('Flat Signal Values')
plt.title('The Signal without the linear Trend')
plt.savefig('SignalNoLinearTrend.png')
plt.show()


'''
To determine the Fourier frequencies in the signal without the linear trend, we perform 
a fast Fourier transform. The frequencies returned are not omega, they are f. The frequency
magnitudes are between 0 and 0.5 Hertz, assuming time is measured in seconds. Because the 
frequencies are bounded, there is an overall period to the signal, which can be seen by just
looking at the original signal data.

We also make a plot of the frequencies versus the FFT amplitudes. We see that the plot is symmetric. To
determine the trigonometric model, we take the absolute values of the frequencies obtained from
the FFT and create a design matrix of sines and cosines with all of the frequencies. We perform 
singular value decomposition on the design matrix to determine the model. We superimpose this
model onto the orginal signal and see almost perfect agreement. This reveals that the FFT was 
done correctly
'''

X = np.fft.fft(NoLinTrend)
freqs = np.fft.fftfreq(len(NoLinTrend))

print(len(freqs))

plt.stem(freqs, np.abs(X), 'b')
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.title('FFT of Signal')
plt.savefig('FFTofSignal.png')
plt.show()

freqs = np.abs(freqs)
freqs = np.unique(freqs)

print(freqs)

A2 = np.zeros((len(time), 2 + 2*len(freqs)))
A2[:, 0] = 1.
A2[:, 1] = time

counter = 0
for i in range(2,2 + 2*len(freqs),2):
    A2[:, i] = np.cos(2*np.pi*freqs[counter]*time)
    A2[:, i+1] = np.sin(2*np.pi*freqs[counter]*time)
    counter += 1
    
(U2, D2, VT2) = np.linalg.svd(A2, full_matrices=False)
A2inv = VT2.transpose().dot(np.diag(1. / D2)).dot(U2.transpose())
C2 = A2inv.dot(signal)
ym = A2.dot(C2) 

plt.plot(time, signal, '.')
plt.plot(time, ym, '.')
plt.xlabel('Time')
plt.ylabel('Signal and Model Values')
plt.legend(['Original Signal', 'Fourier Model'])
plt.title('Fourier Model of the Signal')
plt.savefig('SignalFourier.png')
plt.show()

plt.plot(time, signal - ym, '.')
plt.xlabel('Time')
plt.ylabel('Residual Values: Actual - Model')
plt.title('Residuals for the Fourier Model of the Signal')
plt.savefig('SignalFourierResiduals.png')
plt.show()

