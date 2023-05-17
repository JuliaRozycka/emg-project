# This is a sample Python script.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import Visualizator as v
# ss Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage


def print_hi(name):  # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    # cred = credentials.Certificate("emg-signals-firebase-adminsdk-q9auf-9f867ef3d8.json")
    # firebase_admin.initialize_app(cred, {'storageBucket': 'emg-signals.appspot.com'})
    # bucket = storage.bucket()
    # blob = bucket.blob("o2/osoba_2_lewa_p2.csv")
    # x = blob.download_to_filename('dane_testowe.csv')

    column_names = ["Biceps", "Triceps", "Zginacz", "Prostownik"]
    df = pd.read_csv('dane_testowe.csv', names=column_names, header=None)
    rows = len(df)
    time_col = np.arange(0.001, rows * 0.001 + 0.001 , 0.001)

    df['Czas'] = time_col
    df['Sum'] = df[column_names].sum(axis=1)

    print(df)

    #
    # # Add title and axis labels
    print(-0.0006103515625 + 0.001220703125 - 0.00091552734375 -0.00091552734375)

    plt.title('Line Plot')
    plt.xlabel('Czas')
    plt.ylabel('Napięcie')

    # Show the plot
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
