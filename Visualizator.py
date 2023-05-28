import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame


def visualize_signal(data: DataFrame):
    """
    Simple functions used to visualize
    signal from dataframe

    :param data: dataframe, signal
    """
    data.plot()
    plt.title('Line Plot')
    plt.xlabel('Próbki')
    plt.ylabel('Napięcie')

    # Show the plot
    plt.show()


def visualize_selected_moves(data: DataFrame, movements: [], show: bool = True):
    """
    Function used to visualize extracted moves from signal, colors them in red.

    :param data: signal
    :param movements: list of movements times pairs
    :param show: bool varible if funtions is to show the plot or not
    :return: plot
    """
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 10

    fig, axs = plt.subplots(2)

    xfd = data['Czas']
    yfd = data['Sum']
    yafd = data['AbsSum']

    axs[0].plot(xfd, yfd, linewidth=1)
    axs[1].plot(xfd, yafd, linewidth=1)

    for x1, x2 in movements:
        cut_df = data[(data['Czas'] >= x1) & (data['Czas'] <= x2)]
        axs[0].plot(cut_df['Czas'], cut_df['Sum'], color='r', linewidth=1)
        axs[1].plot(cut_df['Czas'], cut_df['AbsSum'], color='r', linewidth=1)

    for ax in axs.flat:
        ax.set(xlabel='Time [s]', ylabel='Voltage [mV]')

    for ax in axs.flat:
        ax.label_outer()

    if show is True:
        plt.show()
    return plt


def save_plot(data: DataFrame, movements: [], directory: str):
    """
    Function used to save the plot

    :param data: signal
    :param movements: list of movements times pairs
    :param directory: directory to which the file should be saved
    """
    plot = visualize_selected_moves(data, movements, show=False)
    plot.savefig(f'{directory}')
