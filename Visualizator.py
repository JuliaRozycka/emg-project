import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_signal(file):
    file = pd.read_csv(file)
    print(file.head())
