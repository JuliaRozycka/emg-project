from pandas import DataFrame
import pandas as pd
import numpy as np

from scipy.fft import fft, ifft


def read_data(filename: str, cutoff_frequency) -> DataFrame:

    # Set column names
    column_names = ["Biceps", "Triceps", "Zginacz", "Prostownik"]

    # Read the csv with columns
    df = pd.read_csv(filename, names=column_names, header=None)

    # Get number of rows and add times to EMG data
    rows = len(df)
    time_col = np.arange(0.001, rows * 0.001 + 0.001, 0.001)
    df['Czas'] = time_col

    fs = 1000

    # Calculate sum of four columns
    df['Sum'] = df['Biceps']*0.35+df['Triceps']*0.1+df['Prostownik']*0.2+df['Zginacz']*0.35

    signal = df['Sum'].values

    frequency_spectrum = fft(signal)

    freq_axis = np.fft.fftfreq(len(signal), 1/fs)

    frequency_spectrum[np.abs(freq_axis) > cutoff_frequency] = 0

    filtered_signal = ifft(frequency_spectrum)

    filtered_df = pd.DataFrame({'Czas' : df['Czas'].values, 'Sum': np.abs(np.real(filtered_signal))})


    return filtered_df, df


import numpy as np

def detect_hand_movements(df, window_size, threshold_factor):
    maad = np.zeros_like(df['Sum'])
    threshold = np.zeros_like(df['Sum'])

    for i in range(window_size, len(df)):
        window = df['Sum'].values[i - window_size: i]
        mean = np.mean(window)
        deviations = np.abs(window - mean)
        mad = np.mean(deviations)
        maad[i] = mad
        threshold[i] = mean + threshold_factor * mad

    above_threshold = np.where(df['Sum'].values > threshold)[0]
    hand_movement_starts = np.where(np.diff(above_threshold) > 1)[0] + 1
    hand_movement_ends = np.where(np.diff(above_threshold) > 1)[0]

    hand_movements = [(df['Czas'].values[start], df['Czas'].values[end]) for start, end in zip(hand_movement_starts, hand_movement_ends)]

    return hand_movements


def threshold_segmentation(df, threshold):
    segments = []
    start = None

    for i in range(len(df)):
        if df['Sum'].values[i] > threshold:
            if start is None:
                start = df['Czas'].values[i]
        else:
            if start is not None:
                end = df['Czas'].values[i-1]
                segments.append((start, end))
                start = None

    if start is not None:
        end = df['Czas'].values[-1]
        segments.append((start, end))

    return segments


def threshold_segmentation_with_window(df, threshold, window_size):
    segments = []
    start = None
    latest_end_index = 0
    ignore_after=5000 #devP ~ oczywiście można to dodać jako argument funkcji i ustawić odgórnie na to 5000, nie chciałam aż tyle zmieniać

    for i in range(len(df)):
        if i > latest_end_index + ignore_after: #devP
            if i >= window_size:
                window = df['Sum'].values[i - window_size: i]
                mean = np.mean(window)
                if mean > threshold:
                    if start is None:
                        start = df['Czas'].values[i]
                else:
                    if start is not None:
                        end = df['Czas'].values[i - 1]
                        segments.append((start, end))
                        start = None
                        latest_end_index = i

    if start is not None:
        end = df['Czas'].values[-1]
        segments.append((start, end))

    return segments