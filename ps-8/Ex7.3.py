import numpy as np
from scipy.fft import rfft, ifft, fftfreq
import matplotlib.pyplot as plt
import pandas as pd


def readAndDetermineFFT(signalTextFile, saveName1, plt1Title, saveName2, plt2Title, saveName3, plt3Title):
    signal = pd.read_csv(signalTextFile).to_numpy()
    signal = signal.reshape(len(signal))
    dataNum = np.array(range(0,len(signal)))
    samplingRate = np.float32(44100)
    
    
    plt.plot(dataNum, signal)
    plt.xlabel('Signal Count')
    plt.ylabel('Signal Values')
    plt.title(plt1Title)
    plt.savefig(saveName1)
    plt.show() 
    plt.close()
    plt.clf()
    
    FFT = rfft(signal)
    FFT_mag = np.abs(FFT)[0:10000]
    freq = fftfreq(len(signal),1/samplingRate)
    max_k = np.argmax(FFT_mag)
    
    plt.plot(FFT_mag)
    plt.xlabel('Discrete Fourier Transform Mode k Index')
    plt.ylabel('Magnitude of Fourier Coefficient $|c_k|$')
    plt.title(plt2Title)
    plt.savefig(saveName2)
    plt.show()
    plt.close()
    
    plt.plot(freq[0:10000], FFT_mag)
    plt.xlabel('Frequency Hertz')
    plt.ylabel('Magnitude of Fourier Coefficient $|c_k|$')
    plt.title(plt3Title)
    plt.savefig(saveName3)
    plt.show()
    plt.close()
    
    return max_k*samplingRate/len(signal)
    

freqPiano = readAndDetermineFFT('piano.txt', 'xpiano.png', 'Piano Signal', 'xDFTPiano.png','Discrete Fourier Transform of Piano Signal', 'DFTPianoFreq.png', 'Discrete Fourier Transform of Piano Signal')    
freqTrumpet = readAndDetermineFFT('trumpet.txt', 'xtrumpet.png', 'Trumpet Signal', 'xDFTTrumpet.png','Discrete Fourier Transform of Trumpet Signal', 'DFTTrumpetFreq.png', 'Discrete Fourier Transform of Trumpet Signal')    

'''
We see that the piano signal's DFT contains one major peak at relatively small frequency
while the trumpet signal's DFT contains several major peaks for many frequencies, many of 
which are larger than the frequency of the piano spectrum's major peak. The piano produces
a low sound while the trumpet produces a higher sound due to its broader range of larger
frequencies. 
'''
print('The main peak of the piano signal corresponds to a frequency of ' + str(freqPiano) + ' Hertz.')
print('The main peak of the trumpet signal corresponds to a frequency of ' + str(freqTrumpet) + ' Hertz.')

'''
The piano signal's main FFT peak corresponds to a frequency of 524.8 Hz, which corresponds
to the C5 note (523.25 Hz). The trumpet signal's main peak corresponds to a frequency of 
1043.9 Hz, or the C6 note (1046.50 Hz), but we see that the prior peak has a frequency near
the C5 note. Information on musical notes and their frequencies were taken from
https://pages.mtu.edu/~suits/notefreqs.html
'''