import graphviz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.metrics import classification_report

def train_DecisionTreeClassifier(directory: str):
    df = pd.read_csv(directory)
    x=df[['RMS','MAV', 'IEMG', 'VAR', 'WL', 'WAMP','FMN','FMD']].values
    y=df['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    tree_clf=DecisionTreeClassifier(max_depth=6,random_state=42)
    tree_clf.fit(x_train,y_train)

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

    #return



