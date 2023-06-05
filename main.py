from Feature import Feature
from FeatureExtractor import extract_features, extract_feature
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from FeatureExtractor import extract_features
from Utils import read_data, threshold_segmentation_with_window, save_segments_to_files, check_if_csv, \
    normalize_data
from Visualizator import visualize_selected_moves
from SVM_classifier import extract_features_to_csv
from DT_classifier import train_DecisionTreeClassifier, train_DecisonTreeClassifier_OneHotEncodingAddition, \
    trainOVR_DecisionTree, evaluation_statistics, trainOVR_kNN

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
    extract_features_to_csv('features/')

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
    directory = 'features_for_training.csv'
    #print(DTCcheck())



