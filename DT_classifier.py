import graphviz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,\
    roc_auc_score, classification_report, roc_curve, confusion_matrix

def train_DecisionTreeClassifier(directory: str):
    df = pd.read_csv(directory)
    x=df[['RMS','MAV', 'IEMG', 'VAR', 'WL', 'WAMP','FMN','FMD']].values
    y=df['Class']

    # Training a model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    tree_clf=DecisionTreeClassifier(max_depth=6,random_state=42)
    tree_clf.fit(x_train,y_train)

    prediction=tree_clf.predict_proba(x_test)

    # Plotting the tree
    plot_tree(tree_clf)
    dot_data=export_graphviz(tree_clf,
                    #out_file='emg_features_tree_graph.dot',
                    out_file=None,
                    feature_names=['RMS','MAV', 'IEMG', 'VAR', 'WL', 'WAMP','FMN','FMD'],
                    class_names=[str(i) for i in range(1,19)],
                    rounded=True,
                    filled=True,
                    special_characters=True)
    graph=graphviz.Source(dot_data)

    # Evaluation
    accuracy = accuracy_score(y_test,prediction)
    f1 = f1_score(y_test, prediction, average='weighted')
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    full_stats=classification_report(y_test,prediction)
    roc_auc = roc_auc_score(y_test, prediction, average='weighted', multi_class='ovr') # Jeżeli nie robimy OneHotEncoder
    # to oznacza że też mamy OVR (One vs All/Rest)?
    roc=roc_curve(y_test,prediction)
    eval_stats=pd.DataFrame({
        'Accuracy': accuracy, 'F1 score': f1, 'Precision': precision, 'Recall': recall, 'AUC ROC Curve': roc_auc,
        'ROC Curve': roc})
    report_stats=pd.DataFrame({'Classification report': full_stats})

    return tree_clf, eval_stats, report_stats

def train_DecisonTreeClassifier_OneHotEncodingAddition(directory: str):
    df = pd.read_csv(directory)
    x = df[['RMS', 'MAV', 'IEMG', 'VAR', 'WL', 'WAMP', 'FMN', 'FMD']].values
    y = df['Class']

    # One Hot Encoding
    enc=OneHotEncoder()
    y_enc=enc.fit_transform(y.values.reshape(-1,1)).toarray()

    # Training a model
    x_train, x_test, y_train, y_test = train_test_split(x, y_enc, test_size=0.2, random_state=42)
    tree_clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    tree_clf.fit(x_train, y_train)

    # Predicting results
    prediction = tree_clf.predict_proba(x_test)
    y_lab_enc=enc.inverse_transform(y_enc)

    # Evaluation
    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction, average='weighted')
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    roc_auc=roc_auc_score(y_test,prediction,average='weighted',multi_class='ovr')
    roc = roc_curve(y_test, prediction)
    full_stats = classification_report(y_test, prediction)

    eval_stats = pd.DataFrame({
        'Accuracy': accuracy, 'F1 score': f1, 'Precision': precision, 'Recall': recall, 'AUC ROC Curve': roc_auc,
        'ROC Curve': roc})
    report_stats = pd.DataFrame({'Classification report': full_stats})

    return tree_clf, eval_stats, report_stats

def trainOneVsRestClassifier_DecisionTree(directory: str):
    df=pd.read_csv(directory)
    x = df[['RMS', 'MAV', 'IEMG', 'VAR', 'WL', 'WAMP', 'FMN', 'FMD']].values
    y = df['Class']

    # Training a model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    tree_clf=DecisionTreeClassifier(max_depth=6,random_state=42)
    ovr_clf=OneVsRestClassifier(tree_clf)
    ovr_clf.fit(x_train,y_train)

    # Predicting results
    prediction=ovr_clf.predict(x_test)

    # Confusion matrix # I have no idea what I'm doing xd
    classes=range(1,19)
    for i in classes:
        class_test= (y_test==i)
        class_prediction=(prediction==i)
        cm=confusion_matrix(class_test,class_prediction)
        print(f'Confusion matrix for {i} label/class: {cm}')

    return ovr_clf


def evaluation_statistics(y_test, prediction):
    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction, average='weighted')
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    roc_auc = roc_auc_score(y_test, prediction, average='weighted', multi_class='ovr')
    roc = roc_curve(y_test, prediction)
    full_stats = classification_report(y_test, prediction)

    eval_stats=pd.DataFrame({
        'Accuracy': accuracy, 'F1 score': f1, 'Precision': precision, 'Recall': recall, 'AUC ROC Curve': roc_auc,
        'ROC Curve': roc})
    eval_raport= pd.DataFrame({'Classification report': full_stats})

    return eval_stats, eval_raport



