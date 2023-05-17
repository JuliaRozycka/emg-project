from pandas import DataFrame
import pandas as pd
import numpy as np


def read_data(filename: str) -> DataFrame:

    # Set column names
    column_names = ["Biceps", "Triceps", "Zginacz", "Prostownik"]

    # Read the csv with columns
    df = pd.read_csv(filename, names=column_names, header=None)

    # Get number of rows and add times to EMG data
    rows = len(df)
    time_col = np.arange(0.001, rows * 0.001 + 0.001, 0.001)
    df['Czas'] = time_col

    # Calculate sum of four columns
    df['Sum'] = (df[column_names]*0.25).sum(axis=1)

    return df
