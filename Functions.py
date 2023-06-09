import matplotlib.pyplot as plt
import numpy as np

from Utils import time_to_frequency_domain


def fRootMeanSquare(data):
    fRMS = np.sqrt(np.mean(np.power(data, 2)))
    return fRMS


def fMeanAbsoluteValue(data):
    fMAV = np.mean(np.abs(data))
    return fMAV


def fIntegrated(data):
    fIEMG = np.sum(np.abs(data))
    return fIEMG


def fVariance(data):
    fVAR = np.var(data)
    return fVAR


def fSimpleSquareIntegral(data):
    fSSI = np.sum(np.power(data, 2))
    return fSSI


def fWillisonAmplitude(data):
    fWAMP = np.sum(np.abs(np.diff(data)) > 0.01)
    return fWAMP


def fWaveformLength(data):
    fWL = np.sum(np.abs(np.diff(data)))
    return fWL


def fMeanFrequency(data, plot: bool = False):
    amplitude, frequency = time_to_frequency_domain(data)
    nonzero_indices = np.where(frequency > 0.01)
    amplitude = amplitude[nonzero_indices]
    frequency = frequency[nonzero_indices]
    psd = np.abs(amplitude) ** 2  # Power spectrum density

    frequency_mean = np.sum(frequency * psd) / np.sum(psd)  # Mean calculating as usual

    if plot is True:
        # Plot the frequency spectrum
        plt.plot(frequency, np.abs(amplitude))
        plt.axvline(frequency_mean, color='r', linestyle='--', label='Frequency Mean')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0.01, 150)
        plt.title('Frequency Spectrum of EMG Signal')
        plt.legend()
        plt.grid(True)
        plt.show()

    return np.real(frequency_mean)


def fMedianFrequency(data, plot: bool = False):
    amplitude, frequency = time_to_frequency_domain(data)
    nonzero_indices = np.where(frequency > 0.01)
    amplitude = amplitude[nonzero_indices]
    frequency = frequency[nonzero_indices]
    psd = np.abs(amplitude) ** 2  # Power spectrum density

    cumulative_sum = np.cumsum(psd)  # The sum of a given sequence that is increasing

    frequency_median = frequency[np.where(cumulative_sum > np.max(cumulative_sum) / 2)[0][
        0]]  # Median is frequency that splits PSD into two identical parts
    # MDF should use cumsum on the psd and we find were the cumsum is at max(cumsum/2)

    if plot is True:
        # Plot the frequency spectrum
        plt.plot(frequency, np.abs(amplitude))
        plt.axvline(frequency_median, color='r', linestyle='--', label='Frequency Median')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0.01, 150)
        plt.title('Frequency Spectrum of EMG Signal')
        plt.legend()
        plt.grid(True)
        plt.show()

    return np.real(frequency_median)
