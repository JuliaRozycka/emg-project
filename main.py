from Feature import Feature
from FeatureExtractor import extract_feature, extract_features
from Utils import read_data, threshold_segmentation_with_window, save_segments_to_files, check_if_csv
from Visualizator import visualize_selected_moves, visualize_signal
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

if __name__ == '__main__':
    window_size = 1000  # Adjust the window size based on your needs
    threshold = 0.002 # Adjust the threshold factor based on your needs
    cutoff = 100  # Cutoff value in Hz

    # tutaj zmieniać tylko nazwę pliku
    # nazwa_pliku = 'osoba_2_prawa_p1.csv'
    # data = read_data(f'raw_signals/{nazwa_pliku}', cutoff)
    #
    # hand_movements = threshold_segmentation_with_window(data, threshold, window_size,ignore_after=5000)
    # print(len(hand_movements))
    # print(hand_movements)
    #
    # metadata = {
    #     "window_size": window_size,
    #     "threshold": threshold,
    #     "cutoff": cutoff,
    #     "segments": hand_movements
    # }
    # visualize_selected_moves(data, hand_movements)
    # zapisywanie od plików, TODO: tutaj należy zmieniać osobę i pomiar
    #save_segments_to_files(2, 4, data, hand_movements, metadata, savefig=True)


    # window_size = 100  # window size in samples
    # overlap_ratio = 0.9  # overlap size in samples
# Extract features
#     ---------------------------------------------------------------------------------

    rootdir = 'data/'

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            csv_name = os.path.join(subdir, file)
            if check_if_csv(csv_name) is True:
                df_features = extract_features(csv_name, save_to_classes=True)
                print(f'{csv_name} extracted')
    # ---------------------------------------------------------------------------------
    root_dir = "features/"

    # Create an empty list to store MAV data
    mav_data = []

    # Iterate over folder names from 0 to 18
    for folder_name in range(1,19):
        # Generate the folder path
        folder_path = os.path.join(root_dir, str(folder_name))

        # Check if the folder exists
        if not os.path.exists(folder_path):
            continue

        # Iterate over files in the folder
        for file_name in os.listdir(folder_path):
            # Generate the file path
            file_path = os.path.join(folder_path, file_name)

            # Process the file
            # Do something with the file path (e.g., extract features)
            df = pd.read_csv(file_path)
            mav_data.append(df['SSI'].mean())

            # Example: Print the file path
            print(file_path)

    # Create a scatter plot with all MAV data
    fig, ax = plt.subplots()
    for i, data in enumerate(mav_data):
        ax.scatter(data, data, label=f'Folder {i}')
    ax.set_xlabel('Index')
    ax.set_ylabel('MAV')
    ax.set_title('Scatter Plot of MAV')
    ax.legend()
    plt.show()

    print("Extraction done")
