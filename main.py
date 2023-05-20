from Utils import read_data, detect_hand_movements, threshold_segmentation_with_window
from Visualizator import visualize_signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filtered_data, data = read_data('dane_testowe.csv',100)

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

    hand_movements = threshold_segmentation_with_window(filtered_data, 0.0025, 2000)


    print(len(hand_movements))
    print(hand_movements) #devP
    print(filtered_data.head()) #devP

    #visualize_signal(data)
    #visualize_signal(filtered_data)


    # devP - to poniżej tylko żeby zrobić wykresik
    start_points=[point[0] for point in hand_movements]
    end_points=[point[1] for point in hand_movements]
    y_svalue=[0.03]*len(start_points)
    y_evalue = [0.04] * len(end_points)
    xfd=filtered_data['Czas']
    yfd=filtered_data['Sum']
    plt.figure(3)
    plt.plot(xfd, yfd)
    plt.plot(start_points,y_svalue,'o', c='r',alpha=1.0)
    plt.plot(end_points,y_evalue,'o',c='b',alpha=1.0)
    plt.xticks(np.arange(0, 301, 10))
    plt.show()










