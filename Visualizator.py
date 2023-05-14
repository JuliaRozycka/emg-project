import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def visualize_signal(file):
    file = pd.read_csv(file)
    print(file.head())

