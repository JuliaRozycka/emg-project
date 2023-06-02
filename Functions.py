import numpy as np
import matplotlib.pyplot as plt
from Feature import Feature
from scipy.fft import rfft, rfftfreq

def RootMeanSquare(data):
    fRMS = np.sqrt(np.mean(np.power(data, 2)))
    return fRMS
def MeanAbsoluteValue(data):
    fMAV = np.mean(np.abs(data))
    return fMAV

def Integrated(data):
    fIEMG = np.sum(np.abs(data))
    return fIEMG

def Variance(data):
    fVAR = np.var(data)
    return fVAR

def SimpleSquareIntegral(data):
    fSSI = np.sum(np.power(data, 2))
    return fSSI

def WillisonAmplitude(data):
    fWAMP=np.sum(np.abs(np.diff(data)) > 0.01)
    return fWAMP

def WaveformLength(data):
    fWL=np.sum(np.abs(np.diff(data)))
    return fWL

def FrequencyFeatures(data, feature: Feature,savefig: bool = False):
    sample_rate = 1000  # Hz
    yf = rfft(data)
    xf = rfftfreq(len(data), 1 / sample_rate)
    amplitude = yf  # Amplitude, no need for abs because of rfft function
    psd = amplitude ** 2  # Power spectrum density
    cumulative_sum = np.cumsum(psd)  # The sum of a given sequence that is increasing

    # Plot the frequency spectrum
    plt.plot(xf, np.abs(yf))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0.01, 150)
    plt.title('Frequency Spectrum of EMG Signal')
    plt.grid(True)

    if feature==Feature.FMD:
        fFM = xf[np.where(cumulative_sum > np.max(cumulative_sum)/2)[0][0]]  # Median is frequency that splits PSD into two identical parts
        # MDF should use cumsum on the psd and we find were the cumsum is at max(cumsum/2)
        plt.axvline(fFM, color='r', linestyle='--', label='Frequency Median')
    elif feature==Feature.FMN:
        fFM= np.real(np.sum(xf * psd) / np.sum(psd))  # Mean calculating as usual
        plt.axvline(fFM, color='g', linestyle='--', label='Frequency Mean')
    else:
        raise ValueError("Incorrect feature")

    plt.legend()
    #plt.show()
    if savefig is True:
        plt.savefig('Frequency Plots')
    return fFM