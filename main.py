from Utils import read_data, detect_hand_movements, threshold_segmentation_with_window
from Visualizator import visualize_signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = read_data('dane_testowe.csv')

    column_names = ["Biceps", "Triceps", "Zginacz", "Prostownik"]

    # Read the csv with columns
    df = pd.read_csv('dane_testowe.csv', names=column_names, header=None)

    # Get number of rows and add times to EMG data
    rows = len(df)
    time_col = np.arange(0.001, rows * 0.001 + 0.001, 0.001)
    df['Czas'] = time_col

    # df.Biceps.plot(x='Czas')
    # df.Triceps.plot(x='Czas')
    # df.Zginacz.plot(x='Czas')
    # df.Prostownik.plot(x='Czas')
    # plt.legend()
    # plt.show()

    window_size = 1000  # Adjust the window size based on your needs
    threshold_factor = 100  # Adjust the threshold factor based on your needs

    hand_movements = threshold_segmentation_with_window(data, 0.01, 2000)


    print(hand_movements)

    visualize_signal(data)






