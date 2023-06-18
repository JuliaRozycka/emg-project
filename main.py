from Feature import Feature
from FeatureExtractor import extract_features, extract_feature
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from FeatureExtractor import extract_features
from Utils import read_data, threshold_segmentation_with_window, save_segments_to_files, check_if_csv, \
    normalize_data
from Visualizator import visualize_selected_moves
from SVM_classifier import extract_features_to_csv, train_SVM
from DT_classifier import train_DecisionTreeClassifier, train_DecisonTreeClassifier_OneHotEncodingAddition, \
    trainOVR_DecisionTree, evaluation_statistics, trainOVR_kNN, kfold

def filtering_n_segmenting_signals():
    window_size = 300  # Adjust the window size based on your needs
    threshold = 0.0017  # Adjust the threshold factor based on your needs
    cutoff = 100  # Cutoff value in Hz

    nazwa_pliku = 'dane_testowe.csv'

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
    save_segments_to_files(1, 2, data, hand_movements, metadata, savefig=True)


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
    #mav_data = []


def normalizing_data():
    rootdir = 'data/'

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            csv_name = os.path.join(subdir, file)
            if check_if_csv(csv_name) is True:
                df = normalize_data(csv_name)
                print(f'{csv_name} normalized')


def DTCcheck():
    directory='features_for_training.csv'
    treemodel=trainOVR_DecisionTree(directory)
    y_test=treemodel[1]
    prediction=treemodel[2]
    evaluation_of_tree=evaluation_statistics(y_test,prediction)

    print('Separate statistics: ', '\n', evaluation_of_tree[0])
    print('Full package statistics: ','\n', evaluation_of_tree[2])
    print('Full package statistics (but the df): ', '\n', evaluation_of_tree[1])

def kNNcheck():
    directory='features_for_training.csv'
    knnmodel = trainOVR_kNN(directory)
    y_test = knnmodel[1]
    prediction = knnmodel[2]
    evaluation_of_knn = evaluation_statistics(y_test, prediction)

    print('Separate statistics: ', '\n', evaluation_of_knn[0])
    print('Full package statistics: ', '\n', evaluation_of_knn[2])
    print('Full package statistics (but the df): ', '\n', evaluation_of_knn[1])


if __name__ == '__main__':
    # extracting_features()
    # extract_features_to_csv('features/')

    # ---------------------------------------------------------------------------------
    # df_plot = pd.read_csv('data/o1/p1/o1_p1_1.csv')
    # df_plot_normalized = pd.read_csv('normalized_data/o1/p1/o1_p1_1.csv')
    #
    # figs, axs = plt.subplots(2, 1)
    #
    # sum1 = df_plot['Sum'].values
    # time1 = df_plot['Czas'].values
    #
    # sum2 = df_plot_normalized['Sum'].values
    # time2 = df_plot_normalized['Czas'].values
    #
    # axs[0].plot(time1, sum1)
    # axs[0].set_title('Raw')
    # axs[1].plot(time2, sum2)
    # axs[1].set_title('Normalized')
    #
    # plt.show()

    # ---------------------------------------------------------------------------------
    # extraction_process=extracting_features()
    # print('Extraction is done')

    # ---------------------------------------------------------------------------------
    # root_dir = "features/"
    # df_test_list=extract_features_to_csv(root_dir)
    # df_test_list.to_csv('DataLearningSet',index=False)
    # print(df_test_list)

    # df_plot_normalized = pd.read_csv('normalized_data/o1/p1/o1_p1_3.csv')
    # data=df_plot_normalized['Sum'].values
    # root_dir = "features/"
    # extract_features_to_csv(root_dir)
    # df = pd.read_csv('features_for_training.csv')
    # df = df.groupby(['Class']).FMN.mean()
    #
    # extract_feature('data/o2/p1/o2_p1_2.csv', Feature.FMD)
    #
    #
    # dictionary = df.to_dict()
    # keys  = list(dictionary.keys())
    # values = list(dictionary.values())
    # plt.xlabel("Class")
    # plt.ylabel(df.name)
    #
    # plt.scatter(keys, values)
    # plt.xlim(0,19)
    # plt.xticks([i for i in range(1,19)])
    # plt.show()






    # ---------------------------------------------------------------------------------
    #directory = 'features_for_training.csv'
    #print(DTCcheck())



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

    directory = 'features_for_training.csv'
    train_SVM(directory)

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









