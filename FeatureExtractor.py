import numpy as np
import pandas as pd
from Feature import Feature


def extract_feature(filename, feature: Feature):
    value = []
    # Load the signal from the file
    data = pd.read_csv(filename)
    signal = data['Sum'].values

    if feature == Feature.RMS:
        x = np.sqrt(np.mean(np.abs(signal) ** 2))
    elif feature == Feature.MAV:
        x = np.mean(np.abs(signal))
    elif feature == Feature.IEMG:
        x = np.sum(np.abs(signal))
    elif feature == Feature.VAR:
        x = np.var(signal)
    elif feature == Feature.SSI:
        x = np.sum(np.abs(signal) ** 2)
    else:
        raise ValueError("Incorrect feature")

    value.append(x)

    return value


def extract_features(filename, save_to_classes: bool = False):
    rms_feature = extract_feature(filename, Feature.RMS)
    mav_feature = extract_feature(filename, Feature.MAV)
    ssi_feature = extract_feature(filename, Feature.SSI)
    iemg_feature = extract_feature(filename, Feature.IEMG)
    var_feature = extract_feature(filename, Feature.VAR)

    feature_df = pd.DataFrame({'RMS': rms_feature, 'MAV': mav_feature, 'SSI': ssi_feature, 'IEMG': iemg_feature, 'VAR': var_feature})

    if save_to_classes is True:
        pass

    return feature_df

