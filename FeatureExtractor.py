import numpy as np
import pandas as pd
from Feature import Feature
import os
import uuid

import re


def extract_feature(filename, window_size, overlap_ratio, feature: Feature):
    overlap = int(window_size * overlap_ratio)
    values = []

    # Load the signal from the file
    data = pd.read_csv(filename)
    signal = data['Sum'].values

    start = 0
    end = window_size

    while end <= len(signal):
        window = signal[start:end]

        if feature == Feature.RMS:
            x = np.sqrt(np.mean(np.abs(window) ** 2))
        elif feature == Feature.MAV:
            x = np.mean(np.abs(window))
        elif feature == Feature.IEMG:
            x = np.sum(np.abs(window))
        elif feature == Feature.VAR:
            x = np.var(window)
        elif feature == Feature.SSI:
            x = np.sum(np.abs(window) ** 2)
        else:
            raise ValueError("Incorrect feature")

        values.append(x)

        start += window_size - overlap
        end += window_size - overlap

    return values


def extract_features(filename, window_size, overlap_ratio, save_to_classes: bool = False):
    rms_feature = extract_feature(filename, window_size, overlap_ratio, Feature.RMS)
    mav_feature = extract_feature(filename, window_size, overlap_ratio, Feature.MAV)
    ssi_feature = extract_feature(filename, window_size, overlap_ratio, Feature.SSI)
    iemg_feature = extract_feature(filename, window_size, overlap_ratio, Feature.IEMG)
    var_feature = extract_feature(filename, window_size, overlap_ratio, Feature.VAR)

    feature_df = pd.DataFrame(
        {'RMS': rms_feature, 'MAV': mav_feature, 'SSI': ssi_feature, 'IEMG': iemg_feature, 'VAR': var_feature})

    # Extract the number x from the filename using regular expression
    x = re.search(r"o(\d+)_p(\d+)_(\d+)\.csv$", filename)
    move_class = int(x.group(3))
    uid = uuid.uuid4()

    directory = f"features/{move_class}"

    # check if path exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save to file
    if save_to_classes is True:
        feature_df.to_csv(f"{directory}/{uid}.csv", index=False)

    return feature_df
