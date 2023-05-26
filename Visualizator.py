import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def visualize_signal(data: DataFrame):

    data.plot(x='Czas',y='Sum')
    plt.title('Line Plot')
    plt.xlabel('Czas')
    plt.xticks(np.arange(0,301,10)) #devP
    plt.ylabel('Napięcie')

    # Show the plot
    plt.show()

def visualize_selected_moves(data: DataFrame, movements: []):

    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 12

    xfd = data['Czas']
    yfd = data['Sum']
    plt.figure(3)
    plt.plot(xfd, yfd, linewidth=1)

    for x1, x2 in movements:
        cut_df = data[(data['Czas'] >= x1) & (data['Czas'] <= x2)]
        plt.plot(cut_df['Czas'], cut_df['Sum'], color='r', linewidth=1)

    plt.ylabel('Voltage [mV]')
    plt.xlabel('Time [s]')

    plt.show()


def save_plot(data: DataFrame, movements: [], directory: str):

    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 12

    xfd = data['Czas']
    yfd = data['Sum']
    plt.plot(xfd, yfd, linewidth=1)

    for x1, x2 in movements:
        cut_df = data[(data['Czas'] >= x1) & (data['Czas'] <= x2)]
        plt.plot(cut_df['Czas'], cut_df['Sum'], color='r', linewidth=1)

    plt.ylabel('Voltage [mV]')
    plt.xlabel('Time [s]')

    plt.gcf().set_size_inches(10, 5)
    plt.savefig(f'{directory}')
