import graphviz
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    roc_auc_score, classification_report, roc_curve, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree, export_graphviz


def Plot_tree_model(tree_model):
    plot_tree(tree_model)
    dot_data = export_graphviz(tree_model,
                               # out_file='emg_features_tree_graph.dot',
                               out_file=None,
                               feature_names=['RMS', 'MAV', 'IEMG', 'VAR', 'WL', 'WAMP', 'FMN', 'FMD'],
                               class_names=[str(i) for i in range(1, 19)],
                               rounded=True,
                               filled=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)


def evaluation_statistics(y_test, prediction):
    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction, average='weighted')
    precision = precision_score(y_test, prediction, average='weighted')
    recall = recall_score(y_test, prediction, average='weighted')
    # roc_auc = roc_auc_score(y_test, prediction, average='weighted', multi_class='ovr')
    # roc = roc_curve(y_test, prediction)
    full_stats = classification_report(y_test, prediction)

    eval_stats = pd.DataFrame({
        'Accuracy': accuracy, 'F1 score': f1, 'Precision': precision, 'Recall': recall}, index=[0])
    eval_raport = pd.DataFrame({'Classification report': full_stats}, index=[0])

    return eval_stats, eval_raport, full_stats


def Validation_and_Classification(directory: str, classifier, best_feature_amount: int):
    df = pd.read_csv(directory)
    X = df[['RMS', 'MAV', 'IEMG', 'VAR', 'WL', 'WAMP', 'FMN', 'FMD']].values
    y = df['Class']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    select = SelectKBest(chi2, k=best_feature_amount)
    X_new = select.fit_transform(X_scaled, y)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=None)
    ovr_clf = OneVsRestClassifier(classifier)

    bal_acc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    full_metrics = []

    for fold_index, (train_index, test_index) in enumerate(rskf.split(X_new, y)):
        x_train, x_test = X_new[train_index], X_new[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ovr_clf.fit(x_train, y_train)

        # Predict
        prediction = ovr_clf.predict(x_test)

        # Evaluation metrics
        bal_acc_scores.append(balanced_accuracy_score(y_test, prediction))
        f1_scores.append(f1_score(y_test, prediction, average='weighted', zero_division=0))
        precision_scores.append(precision_score(y_test, prediction, average='weighted', zero_division=0))
        recall_scores.append(recall_score(y_test, prediction, average='weighted'))
        full_metrics.append(classification_report(y_test, prediction, zero_division=0))
        # Confusion matrix
        classes = range(1, 19)
        fig, axes = plt.subplots(3, 6, figsize=(12, 6))
        r = 0
        c = 0
        for i in classes:
            class_test = (y_test == i)
            class_prediction = (prediction == i)
            cm = confusion_matrix(class_test, class_prediction)
            # print(f'Confusion matrix for {i} label/class: {cm}')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['rest', i])
            axi = axes[r, c]
            axi.set_title(i)
            disp.plot(ax=axi)
            disp.im_.colorbar.remove()
            c += 1
            if c >= 6:
                r += 1
                c = 0
        fig.colorbar(disp.im_, ax=axes)

    # eval_metrics= pd.DataFrame({
    #     'Balanced accuracy': bal_acc_scores, 'F1 score': f1_scores, 'Precision': precision_scores, 'Recall': recall_scores}, index=[0])
    # eval_raport= pd.DataFrame({'Classification report': full_metrics}, index=[0])
    #plt.show()

    return bal_acc_scores, f1_scores, precision_scores, recall_scores, full_metrics


def returnKbest(directory, best_feature_amount):
    df = pd.read_csv(directory)
    X = df[['RMS', 'MAV', 'IEMG', 'VAR', 'WL', 'WAMP', 'FMN', 'FMD']]
    y = df['Class']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    select = SelectKBest(chi2, k=best_feature_amount)
    X_new = select.fit_transform(X_scaled, y)

    mask = select.get_support()
    new_features = X.columns[mask]

    return new_features




