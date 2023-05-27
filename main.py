from Feature import Feature
from FeatureExtractor import extract_feature
from Utils import read_data, threshold_segmentation_with_window, save_segments_to_files
from Visualizator import visualize_selected_moves, visualize_signal
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    window_size = 1000  # Adjust the window size based on your needs
    threshold = 0.0019 # Adjust the threshold factor based on your needs
    cutoff = 100  # Cutoff value in Hz



    # tutaj zmieniać tylko nazwę pliku
    nazwa_pliku = 'osoba_7_lewa_p3.csv'

    data = read_data(f'raw_signals/{nazwa_pliku}', cutoff)

    hand_movements = threshold_segmentation_with_window(data, threshold, window_size,ignore_after=6000)
    print(len(hand_movements))
    print(hand_movements)

    metadata = {
        "window_size": window_size,
        "threshold": threshold,
        "cutoff": cutoff,
        "segments": hand_movements
    }

    visualize_selected_moves(data, hand_movements)


    # zapisywanie od plików, TODO: tutaj należy zmieniać osobę i pomiar
    #save_segments_to_files(7, 3, data, hand_movements, metadata, savefig=True)


    window_size = 100  # window size in samples
    overlap_ratio = 0.9  # overlap size in samples

