from Utils import read_data, threshold_segmentation_with_window, save_segments_to_files
from Visualizator import visualize_selected_moves
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    window_size = 800  # Adjust the window size based on your needs
    threshold = 0.003  # Adjust the threshold factor based on your needs
    cutoff = 100  # Cutoff value in Hz

    metadata = {
        "window_size": window_size,
        "threshold": threshold,
        "cutoff": cutoff
    }

    # tutaj zmieniać tylko nazwę pliku
    nazwa_pliku = 'osoba_6_lewa_p1.csv'

    data = read_data(f'raw_signals/{nazwa_pliku}', cutoff)

    hand_movements = threshold_segmentation_with_window(data, threshold, window_size,ignore_after=8000)
    print(len(hand_movements))
    print(hand_movements)
    visualize_selected_moves(data, hand_movements)


    # zapisywanie od plików, TODO: tutaj należy zmieniać osobę i pomiar
    save_segments_to_files(6, 1, data, hand_movements, metadata, savefig=True)
