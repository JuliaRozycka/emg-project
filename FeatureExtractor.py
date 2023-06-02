import os
import re
import uuid
import numpy as np
import pandas as pd
from Feature import Feature
from Functions import RootMeanSquare, MeanAbsoluteValue, Integrated, Variance, WillisonAmplitude, WaveformLength, FrequencyFeatures


def extract_feature(filename: str, feature: Feature):

    """
    #     Function extracting specific feature from file
    #
    #     :param filename: file path
    #     :param feature: enum type of feature to extract
    #     :return: extracted feature
    #     """

    #Load the signal from the file
    data = pd.read_csv(filename)
    signal = data['Sum'].values

    if feature == Feature.RMS:
        x=RootMeanSquare(signal)
    elif feature == Feature.MAV:
        x=MeanAbsoluteValue(signal)
    elif feature == Feature.IEMG:
        x=Integrated(signal)
    elif feature == Feature.VAR:
        x=Variance(signal)
    elif feature == Feature.WL:
        x=WaveformLength(signal)
    elif feature == Feature.WAMP:
        x=WillisonAmplitude(signal)
    elif feature == Feature.FMN: # Frequency Mean
        x=FrequencyFeatures(signal,feature,savefig=False)
    elif feature == Feature.FMD: # Frequency Median
        x=FrequencyFeatures(signal,feature,savefig=False)
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
         'WL': wl_feature, 'WAMP': wamp_feature, 'FMN':fmn_feature,'FMD':fmd_feature}, index=[0])

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
