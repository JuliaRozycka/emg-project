from sklearn.svm import SVC

from Classifiers import Classifiers
from Statistics import normal_distribution_check, boxplot, manova
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from Feature import Feature
from FeatureExtractor import extract_features, extract_feature
import os
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from FeatureExtractor import extract_features, extract_features_to_csv
from Utils import read_data, threshold_segmentation_with_window, save_segments_to_files, check_if_csv, \
    normalize_data
from Visualizator import visualize_selected_moves
from Classifier import Validation_and_Classification, Plot_tree_model, evaluation_statistics


def filtering_n_segmenting_signals():
    window_size = 800  # Adjust the window size based on your needs
    threshold = 0.003  # Adjust the threshold factor based on your needs
    cutoff = 100  # Cutoff value in Hz

    nazwa_pliku = 'osoba_6_lewa_p2.csv'

    data = read_data(f'raw_signals/{nazwa_pliku}', cutoff)

    hand_movements = threshold_segmentation_with_window(data, threshold, window_size)

    metadata = {
        "window_size": window_size,
        "threshold": threshold,
        "cutoff": cutoff,
        "segments": hand_movements
    }

    visualize_selected_moves(data, hand_movements)

    # zapisywanie od plików,
    # save_segments_to_files(1, 2, data, hand_movements, metadata, savefig=True)


def extracting_features():
    rootdir = 'normalized_data/'

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            csv_name = os.path.join(subdir, file)
            if check_if_csv(csv_name) is True:
                df_features = extract_features(csv_name, save_to_classes=True)
                print(f'{csv_name} extracted')
    # ---------------------------------------------------------------------------------
    root_dir = "features/"

    # Create an empty list to store MAV data
    # mav_data = []


def normalizing_data():
    rootdir = 'data/'

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            csv_name = os.path.join(subdir, file)
            if check_if_csv(csv_name) is True:
                df = normalize_data(csv_name)
                print(f'{csv_name} normalized')

def Train_SVM():
    directory = 'features_for_training.csv'
    clf = SVC(kernel='rbf', C=25, gamma='scale')
    svm_model = Validation_and_Classification(directory, clf, 6)

    print('Balanced accuracy scores: ', svm_model[0])
    print('F1 scores: ', svm_model[1])
    print('Precision scores: ', svm_model[2])
    print('Recall scores: ', svm_model[3])

    df = pd.DataFrame(columns=['bal_acc', 'f1_score', 'precision', 'recall'])
    df['bal_acc'] = svm_model[0]
    df['f1_score'] = svm_model[1]
    df['precision'] = svm_model[2]
    df['recall'] = svm_model[3]

    df.to_csv('metrics/svm_metrics.csv', index=False)


def Train_Decision_Tree():
    directory = 'features_for_training.csv'
    clf = DecisionTreeClassifier(max_depth=4, random_state=10)
    tree_model = Validation_and_Classification(directory, clf, 6)

    print('Balanced accuracy scores: ', tree_model[0])
    print('F1 scores: ', tree_model[1])
    print('Precision scores: ', tree_model[2])
    print('Recall scores: ', tree_model[3])
    df = pd.DataFrame(columns=['bal_acc', 'f1_score', 'precision', 'recall'])
    df['bal_acc'] = tree_model[0]
    df['f1_score'] = tree_model[1]
    df['precision'] = tree_model[2]
    df['recall'] = tree_model[3]

    df.to_csv('metrics/dt_metrics.csv', index=False)


def Train_KNN():
    directory = 'features_for_training.csv'
    clf = KNeighborsClassifier(n_neighbors=6)
    knn_model = Validation_and_Classification(directory, clf, 6)

    print('Balanced accuracy scores: ', knn_model[0])
    print('F1 scores: ', knn_model[1])
    print('Precision scores: ', knn_model[2])
    print('Recall scores: ', knn_model[3])

    df = pd.DataFrame(columns=['bal_acc', 'f1_score', 'precision', 'recall'])
    df['bal_acc'] = knn_model[0]
    df['f1_score'] = knn_model[1]
    df['precision'] = knn_model[2]
    df['recall'] = knn_model[3]

    df.to_csv('metrics/knn_metrics.csv', index=False)


def check_distribution(directory: str):
    data = pd.read_csv(directory)
    for columns in list(data.columns.values):
        print(columns, normal_distribution_check(data[columns]))


if __name__ == '__main__':
    pass

    # ---------------------------------------------------------------------------------
    # mpl.rcParams['font.family'] = 'serif'
    # mpl.rcParams['font.serif'] = ['Times New Roman']
    # mpl.rcParams['font.size'] = 10
    # df_plot = pd.read_csv('data/o1/p1/o1_p1_1.csv')
    # df_plot_normalized = pd.read_csv('normalized_data/o1/p1/o1_p1_1.csv')
    # figs, axs = plt.subplots(2, 1)
    # plt.subplots_adjust(hspace=0.3)
    # sum1 = df_plot['Sum'].values
    # time1 = df_plot['Czas'].values
    #
    # sum2 = df_plot_normalized['Sum'].values
    # time2 = df_plot_normalized['Czas'].values
    #
    # axs[0].plot(time1, sum1)
    # axs[0].set_title('Segment przed normalizacją')
    # axs[0].grid()
    # axs[0].set_xlabel('Czas [s]')
    # axs[0].set_ylabel('Amplituda [mV]')
    # axs[1].plot(time2, sum2)
    # axs[1].set_title('Segment po normalizacji')
    # axs[1].set_xlabel('Czas [s]')
    # axs[1].set_ylabel('Amplituda [mV]')
    # axs[1].grid()
    #
    # plt.show()


    # df_plot_normalized = pd.read_csv('normalized_data/o1/p1/o1_p1_3.csv')
    # data=df_plot_normalized['Sum'].values


    # Balanced accuracy , kfoldwalidacja
    # Select best feature (domyślnie anova, test stats -> chi square) ewentualnie PCA, ale to będzie prostsze
    # Im wyższe p value tym większe związanie, skorelowanie

    # ---------------------------------------------------------------------------------
    # print('DECISION TREE CLASSIFICATION METRICS: ')
    # Train_Decision_Tree()
    #
    # print('K-NEAREST NEIGHBOUR METRICS: ')
    # Train_KNN()
    #
    # print('SVM METRICS: ')
    # Train_SVM()

    # ---------------------------------------------------------------------------------

    #  Set column names
    # column_names = ["Biceps", "Triceps", "Zginacz", "Prostownik"]
    # filename='raw_signals/osoba_6_lewa_p2.csv'
    # # Read the csv with columns
    # df = pd.read_csv(filename, names=column_names, header=None)
    #
    # # Get number of rows and add times to EMG data
    # rows = len(df)
    # time_col = np.arange(0.001, rows * 0.001 + 0.001, 0.001)
    # df['Czas'] = time_col
    # df['Sum'] = df['Biceps'] * 0.35 + df['Triceps'] * 0.1 + df['Prostownik'] * 0.2 + df['Zginacz'] * 0.35
    #
    # biceps=df['Biceps'].values
    # triceps = df['Triceps'].values
    # zginacz = df['Zginacz'].values
    # prostownik = df['Prostownik'].values
    # czas=df['Czas'].values
    # one_signal = df['Sum'].values
    #
    # plt.figure(1)
    # plt.subplot(4,1,1)
    # plt.plot(czas,biceps,'b')
    # plt.grid()
    # plt.xlim(0,301)
    # plt.xticks(np.arange(0,301,25))
    # plt.title('Mięsień dwugłowy ramienia - Biceps')
    # plt.xlabel('Czas [s]')
    # plt.ylabel('Amplituda [mV]')
    # plt.subplot(4, 1, 2)
    # plt.plot(czas, triceps,'r')
    # plt.grid()
    # plt.xlim(0, 301)
    # plt.xticks(np.arange(0,301,25))
    # plt.title('Mięsień trójgłowy ramienia - Triceps')
    # plt.xlabel('Czas [s]')
    # plt.ylabel('Amplituda [mV]')
    # plt.subplot(4, 1, 3)
    # plt.plot(czas, zginacz,'g')
    # plt.grid()
    # plt.xlim(0, 301)
    # plt.xticks(np.arange(0,301,25))
    # plt.title('Zginacz łokciowy nadgarstka')
    # plt.xlabel('Czas [s]')
    # plt.ylabel('Amplituda [mV]')
    # plt.subplot(4, 1, 4)
    # plt.plot(czas, prostownik,'m')
    # plt.grid()
    # plt.xlim(0, 301)
    # plt.xticks(np.arange(0,301,25))
    # plt.title('Prostownik palców')
    # plt.xlabel('Czas [s]')
    # plt.ylabel('Amplituda [mV]')
    # plt.subplots_adjust(hspace=0.6)
    #
    # plt.figure(2)
    # plt.plot(czas,one_signal,'b')
    # plt.grid()
    # plt.xlim(0, 301)
    # plt.xticks(np.arange(0,301,25))
    # plt.title('Zsumowany sygnał')
    # plt.xlabel('Czas [s]')
    # plt.ylabel('Amplituda [mV]')
    #
    # plt.show()

    # Generacja wykresów do sprawka
    # Set column names
    # column_names = ["Biceps", "Triceps", "Zginacz", "Prostownik"]
    # filename='raw_signals/osoba_5_lewa_p1.csv'
    # # Read the csv with columns
    # df_plot = pd.read_csv(filename, names=column_names, header=None)
    # # Get number of rows and add times to EMG data
    # rows = len(df_plot)
    # time_col = np.arange(0.001, rows * 0.001 + 0.001, 0.001)
    # df_plot['Czas'] = time_col
    #
    # biceps=df_plot['Biceps'].values
    # triceps = df_plot['Triceps'].values
    # prostownik = df_plot['Prostownik'].values
    # zginacz = df_plot['Zginacz'].values
    # time_plot=df_plot['Czas'].values
    #
    # plt.figure(1)
    # plt.subplot(4,1,1)
    # plt.plot(time_plot, biceps)
    # plt.title('Biceps')
    # plt.grid()
    # plt.xlim(0,300)
    # plt.xticks(list(range(0,301,25)))
    # plt.subplot(4, 1, 2)
    # plt.plot(time_plot, triceps)
    # plt.title('Triceps')
    # plt.subplot(4, 1, 3)
    # plt.plot(time_plot, zginacz)
    # plt.title('Zginacz')
    # plt.subplot(4, 1, 4)
    # plt.plot(time_plot, prostownik)
    # plt.title('Prostownik')
    # plt.grid()
    #
    # df_plot['Sum'] = df_plot['Biceps'] * 0.35 + df_plot['Triceps'] * 0.1 + df_plot['Prostownik'] * 0.2 + df_plot['Zginacz'] * 0.35
    # signal_one = df_plot['Sum'].values
    #
    # plt.figure(2)
    # plt.plot(time_plot,signal_one)
    # plt.show()

    #-------------------------------------------------------------------------------
    # # ANALIZA OTRZYAMANYCH CECH
    # df = pd.read_csv('features_for_training.csv')
    #
    # print(df.describe())
    # print(df.isnull().sum())
    #
    # corr = df[df.columns].corr()
    # sns.heatmap(corr, cmap="YlGnBu", annot=True)
    # plt.title('Heatmap for Correlation of Parameters')
    # plt.show()
    #
    # columns_to_include = list(df.columns)[:-1]
    # fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(7, 10))
    #
    # for i, column in enumerate(columns_to_include):
    #     ax = axes[i // 2, i % 2]  # Get the current subplot axes
    #     df[column].plot.hist(ax=ax, color='Pink')
    #     ax.set_title(column)
    #
    # plt.tight_layout()  # Adjust spacing between subplots
    # plt.show()
    #
    # sns.pairplot(df[['MAV','RMS','Class']], hue='Class')
    # plt.show()
    #
    # features = list(df.columns.values)[:-1]
    #
    # # Visualize feature distributions
    # for feature in features:
    #     plt.figure(figsize=(8, 6))
    #     sns.histplot(data=df, x=feature, hue='Class', kde=True, palette= 'hls')
    #     sns.color_palette("Spectral", as_cmap=True)
    #     plt.title(f"Distribution of {feature}")
    #     plt.show()


    print(boxplot(Classifiers.kNN, 'recall'))
    print(manova('bal_acc','precision'))

