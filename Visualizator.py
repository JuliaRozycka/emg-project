import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


def visualize_signal(data: DataFrame):

    data.plot(x='Czas',y='Sum')
    plt.title('Line Plot')
    plt.xlabel('Czas')
    plt.ylabel('Napięcie')

    # Show the plot
    plt.show()
