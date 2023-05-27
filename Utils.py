from pandas import DataFrame
import pandas as pd
import numpy as np
import os
import json
from Visualizator import save_plot

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
    df['Sum'] = df['Biceps'] * 0.35 + df['Triceps'] * 0.1 + df['Prostownik'] * 0.2 + df['Zginacz'] * 0.35

    signal = df['Sum'].values
    frequency_spectrum = fft(signal)
    freq_axis = np.fft.fftfreq(len(signal), 1 / fs)
    frequency_spectrum[np.abs(freq_axis) > cutoff_frequency] = 0
    filtered_signal = ifft(frequency_spectrum)
    filtered_df = pd.DataFrame(
        {'Czas': df['Czas'].values, 'AbsSum': np.abs(np.real(filtered_signal)), 'Sum': np.real(filtered_signal)})

    return filtered_df


def threshold_segmentation_with_window(df, threshold, window_size):
    segments = []
    start = None
    latest_end_index = 0
    ignore_after = 5000

    for i in range(len(df)):
        if i > latest_end_index + ignore_after:  # devP
            if i >= window_size:
                window = df['AbsSum'].values[i - window_size: i]
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


def save_segments_to_files(osoba: int, pomiar: int, data: DataFrame, movements: [], metadata: dict,
                           savefig: bool = False):
    i = 1

    directory = f"data/o{osoba}/p{pomiar}"

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = f"{directory}/o{osoba}_p{pomiar}.json"

    with open(file_name, "w") as json_file:
        json.dump(metadata, json_file)


    if savefig is True:
        save_plot(data, movements, f"{directory}/o{osoba}_p{pomiar}.svg")

    filtered_df = data.copy()

    for x1, x2 in movements:
        # segments
        cut_df = data[(data['Czas'] >= x1) & (data['Czas'] <= x2)]
        cut_df.to_csv(f"{directory}/o{osoba}_p{pomiar}_{i}.csv", index=False)

        # rest of the signal
        filtered_df = filtered_df.drop(filtered_df[(filtered_df['Czas'] >= x1) & (filtered_df['Czas'] <= x2)].index)

        i += 1

    filtered_df.to_csv(f"{directory}/o{osoba}_p{pomiar}_0.csv", index=False)
