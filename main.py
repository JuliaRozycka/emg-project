from Feature import Feature
from FeatureExtractor import extract_feature, extract_features
from Utils import read_data, threshold_segmentation_with_window, save_segments_to_files
from Visualizator import visualize_selected_moves, visualize_signal
import pandas as pd
import matplotlib.pyplot as plt
import re

if __name__ == '__main__':
    window_size = 300  # Adjust the window size based on your needs
    threshold = 0.0017  # Adjust the threshold factor based on your needs
    cutoff = 100  # Cutoff value in Hz



    # # tutaj zmieniać tylko nazwę pliku
    # nazwa_pliku = 'dane_testowe.csv'
    #
    # data = read_data(f'raw_signals/{nazwa_pliku}', cutoff)
    #
    # hand_movements = threshold_segmentation_with_window(data, threshold, window_size)
    #
    # metadata = {
    #     "window_size": window_size,
    #     "threshold": threshold,
    #     "cutoff": cutoff,
    #     "segments": hand_movements
    # }
    #
    # visualize_selected_moves(data, hand_movements)
    #
    # # zapisywanie od plików, TODO: tutaj należy zmieniać osobę i pomiar
    # save_segments_to_files(1, 2, data, hand_movements, metadata, savefig=True)

    window = 100  # window size in samples
    overlap_ratio = 0.9  # overlap size in samples

    for i in range(0,19):
        filename = f"data/o1/p2/o1_p2_{i}.csv"
        df_features = extract_features(filename, window, overlap_ratio, save_to_classes=True)



