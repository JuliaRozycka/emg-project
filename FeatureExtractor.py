import os
import re
import uuid

import pandas as pd

from Feature import Feature
from Functions import fVariance, fIntegrated, fRootMeanSquare, fMeanAbsoluteValue, \
    fWaveformLength, fWillisonAmplitude, fMeanFrequency, fMedianFrequency


def extract_feature(filename: str, feature: Feature):
    """
    #     Function extracting specific feature from file
    #
    #     :param filename: file path
    #     :param feature: enum type of feature to extract
    #     :return: extracted feature
    #     """

    # Load the signal from the file
    data = pd.read_csv(filename)
    signal = data['Sum'].values

    if feature == Feature.RMS:
        x = fRootMeanSquare(signal)
    elif feature == Feature.MAV:
        x = fMeanAbsoluteValue(signal)
    elif feature == Feature.IEMG:
        x = fIntegrated(signal)
    elif feature == Feature.VAR:
        x = fVariance(signal)
    elif feature == Feature.WL:
        x = fWaveformLength(signal)
    elif feature == Feature.WAMP:
        x = fWillisonAmplitude(signal)
    elif feature == Feature.FMN:
        x = fMeanFrequency(signal, False)
    elif feature == Feature.FMD:
        x = fMedianFrequency(signal, False)
    else:
        raise ValueError("Incorrect feature")
    return x


def extract_features(filename, save_to_classes: bool = False):
    """
    Function to extract all features from file and saving the data into 18 classes

    :param filename: filepath
    :param save_to_classes: bool variable to choose if features are to be saved to files
    :return: features dataframe
    """
    rms_feature = extract_feature(filename, Feature.RMS)
    mav_feature = extract_feature(filename, Feature.MAV)
    iemg_feature = extract_feature(filename, Feature.IEMG)
    var_feature = extract_feature(filename, Feature.VAR)
    wl_feature = extract_feature(filename, Feature.WL)
    wamp_feature = extract_feature(filename, Feature.WAMP)
    fmn_feature = extract_feature(filename, Feature.FMN)
    fmd_feature = extract_feature(filename, Feature.FMD)

    feature_df = pd.DataFrame(
        {'RMS': rms_feature, 'MAV': mav_feature, 'IEMG': iemg_feature, 'VAR': var_feature,
         'WL': wl_feature, 'WAMP': wamp_feature, 'FMN': fmn_feature, 'FMD': fmd_feature}, index=[0])

    # Extract the number x from the filename using regular expression
    x = re.search(r"o(\d+)_p(\d+)_(\d+)\.csv$", filename)
    move_class = int(x.group(3))
    person = int(x.group(1))
    measurement = int(x.group(2))
    uid = uuid.uuid4()

    directory = f"features/{move_class}"

    # check if path exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save to file
    if save_to_classes is True:
        feature_df.to_csv(f"{directory}/{uid}-o{person}p{measurement}.csv", index=False)

    return feature_df


def extract_features_to_csv(directory):
    # directory  is features/
    # iterate over folders 1-18 in features
    df = pd.DataFrame(columns=['RMS', 'MAV', 'IEMG', 'VAR', 'WL', 'WAMP', 'FMN', 'FMD', 'Class'])

    for _class in range(1, 19):
        class_path = os.path.join(directory, str(_class))
        for file in os.listdir(class_path):
            f = os.path.join(class_path, file)
            if os.path.isfile(f):
                extracted_features = pd.read_csv(f)

                row = [
                    extracted_features['RMS'].values[0],
                    extracted_features['MAV'].values[0],
                    extracted_features['IEMG'].values[0],
                    extracted_features['VAR'].values[0],
                    extracted_features['WL'].values[0],
                    extracted_features['WAMP'].values[0],
                    extracted_features['FMN'].values[0],
                    extracted_features['FMD'].values[0],
                    int(_class)
                ]

                df.loc[len(df)] = row

            # break
            # Break after reading first file - to jest do usunięcia jak ustalimy co bierzemy za faetures i czy lista czy co, bo wsm tak sobie
            # myślę że niepotzrebnie chyba bierzemy tyle podobnych features z time domain, można wywalić jakieś 3 i dodać 2 z frequency domain
    df.to_csv('features_for_training.csv', index=False)
    return df
