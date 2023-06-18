from pandas import DataFrame
import pandas as pd
import numpy as np
import os
import json
from Visualizator import save_plot
import re
from scipy.fft import fft, ifft, rfft, rfftfreq


def read_data(filename: str, cutoff_frequency) -> DataFrame:
    """
    Function that reads data from csv file, does weighted summation into one channel,
    cleans the signal by additional filtering and returns filtered signal

    :param filename: filepath
    :param cutoff_frequency: frequency to cut of the signal
    :return: filtered signal in form of a dataframe
    """
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


def threshold_segmentation_with_window(df, threshold, window_size, ignore_after=5000):
    """
    Function used to extract different moves from signal.
    It takes two main paramteres to do this: window size and
    threshold. It calculates mean of the signal in this window,
    if it is bigger than threshold it starts capturing the movement.
    As soon as it gets below threshold the capturing ends and then function
    ignores few seconds of signal.

    :param df: signal in form of a dataframe
    :param threshold: threshold where the signal is to be detected
    :param window_size: window size
    :return: list of pairs - start time and end time of the move
    """

    segments = []
    start = None
    latest_end_index = 0

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
    """
    Function used to save segments into files, and to save figure
    that contains visual representation of which signals are taken into
    account

    :param osoba: number of person
    :param pomiar: number of measurment
    :param data: signal
    :param movements: list of pairs: start time and end time of the move
    :param metadata: file metadata - window size etc.
    :param saveifig: bool variable to choose if plot are to be saved to svg file
    """
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

def check_if_csv(filename: str) -> bool:
    """
    Functions used to check if file is a csv file

    :param filename: file path
    :returns
    """

    return bool(re.search(r"(\d+)\.csv$", filename))


def count_files_in_folders(directory):
    # Iterate over folder names from 0 to 18
    for folder_name in range(19):
        # Generate the folder path
        folder_path = os.path.join(directory, str(folder_name))

        # Check if the folder exists
        if not os.path.exists(folder_path):
            continue

        # Count the number of files in the folder
        file_count = len(os.listdir(folder_path))

        # Print the result
        print(f"Folder {folder_name}: {file_count} files")

def search_for_min_max(osoba: int, pomiar: int):
    """

    """
    directory = f"data/o{osoba}/p{pomiar}/"

    minimum_list = []
    maximum_list = []

    for files in os.listdir(directory):
        path = os.path.join(directory, files)
        if check_if_csv(path):
            df = pd.read_csv(path)
            minimum = df['Sum'].min()
            maximum = df['Sum'].max()
            minimum_list.append(minimum)
            maximum_list.append(maximum)

    minimum_list = sorted(minimum_list)[:3]
    maximum_list = sorted(maximum_list)[-3:]

    return np.mean(minimum_list), np.mean(maximum_list)


def min_max_normalisation(df, min, max):
    signal = df['Sum']

    normalized_min = -1
    normalized_max = 1
    normalized_signal = (signal - min) / (max - min) * (
            normalized_max - normalized_min) + normalized_min

    df['Sum'] = normalized_signal

    df['AbsSum'] = df['Sum'].abs()

    return df


def normalize_data(filename):
    x = re.search(r"o(\d+)_p(\d+)_(\d+)\.csv$", filename)

    osoba = int(x.group(1))
    pomiar = int(x.group(2))
    ruch = int(x.group(3))

    min, max = search_for_min_max(osoba, pomiar)

    df = pd.read_csv(filename)

    df.loc[df['Sum'] > max, 'Sum'] = max
    df.loc[df['Sum'] < min, 'Sum'] = min

    normalized = min_max_normalisation(df, min, max)

    directory = f'normalized_data/o{osoba}/p{pomiar}/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    normalized.to_csv(f'{directory}o{osoba}_p{pomiar}_{ruch}.csv', index=False)

    return  normalized


def time_to_frequency_domain(signal):
    sample_rate = 1000  # Hz
    amplitude = rfft(signal)
    frequency = rfftfreq(len(signal), 1 / sample_rate)


    return amplitude, frequency

