import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from statsmodels.multivariate.manova import MANOVA

from Classifiers import Classifiers
from Metrics import Metrics


def dekor(f):
    def wrapper(*args, **kwargs):
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman']
        mpl.rcParams['font.size'] = 10
        return f(*args, **kwargs)

    return wrapper

def t_test_corrected(a, b, J=5, k=5):
    """
    Corrected t-test for repeated cross-validation.
    input, two 2d arrays. Repetitions x folds
    As default for 5x5CV
    """
    if J * k != a.shape[0]:
        raise Exception('%i scores received, but J=%i, k=%i (J*k=%i)' % (
            a.shape[0], J, k, J * k
        ))

    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (J * k) + 1 / (k - 1)) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), (k * J) - 1) * 2
    return t_stat, pval


def normal_distribution_check(data: pd.DataFrame):
    '''
     Dla liczebności populacji powyżej 20 stosujemy test D’Agostino–Pearsoa (normaltest)
     Dla liczebności populacji poniżej 20 stosujemy test SHapiro-Wilka (shapiro)
    :param data:
    :return:
    '''
    if len(data) >= 20:
        stats, p = normaltest(data)
    else:
        stats, p = shapiro(data)

    if p > 0.05:
        isnormal = True
    else:
        isnormal = False
    return stats, p, isnormal

@dekor
def ttest_independent(a: Classifiers, b: Classifiers, metric: Metrics):
    data_a = pd.read_csv(f'metrics/{a.value}_metrics.csv')
    data_a = data_a[metric.name]
    print(a.name, data_a.mean())
    data_b = pd.read_csv(f'metrics/{b.value}_metrics.csv')
    data_b = data_b[metric.name]
    print(b.name, data_b.mean())
    sns.kdeplot(data_a, fill=True, label=a.name)
    sns.kdeplot(data_b, fill=True, label=b.name)
    plt.title("Independent Sample T-Test")
    plt.xlabel(metric.value)
    plt.legend()
    plt.show()
    return stats.ttest_ind(data_a, data_b)




@dekor
def manova(metric_a: Metrics, metric_b: Metrics):
    data = []

    directories = ['metrics/knn_metrics.csv', 'metrics/svm_metrics.csv', 'metrics/dt_metrics.csv']

    for classifier in directories:
        # Read the metrics file
        df = pd.read_csv(classifier)

        # Extract the classifier name from the file path
        classifier_name = classifier.split('/')[-1].split('_')[0]

        for _, row in df.iterrows():
            # Get the balanced accuracy and f1-score for each row
            metric1 = row[metric_a.name]
            metric2 = row[metric_b.name]

            # Create a new dictionary with the classifier name, balanced accuracy, and f1-score
            data_row = {'Classifier': classifier_name, metric_a.name: metric1, metric_b.name: metric2}

            # Append the row to the data list
            data.append(data_row)

    # Convert the data list to a DataFrame
    data = pd.DataFrame(data)

    fit = MANOVA.from_formula(f'{metric_a.name} + {metric_b.name} ~ Classifier', data=data)

    X = data[[metric_a.name, metric_b.name]]
    y = data["Classifier"]
    post_hoc = lda().fit(X=X, y=y)

    # plot
    X_new = pd.DataFrame(lda().fit(X=X, y=y).transform(X), columns=[f"LDA ({metric_a.value})", f"LDA ({metric_b.value})"])
    X_new["Classifier"] = data["Classifier"]
    sns.scatterplot(data=X_new, x=f"LDA ({metric_a.value})", y=f"LDA ({metric_b.value})", hue=data.Classifier.tolist())
    plt.show()

    return fit.mv_test()


def metrics_to_one_csv():
    data = []

    directories = ['metrics/knn_metrics.csv', 'metrics/svm_metrics.csv', 'metrics/dt_metrics.csv']

    for classifier in directories:
        # Read the metrics file
        df = pd.read_csv(classifier)

        # Extract the classifier name from the file path
        classifier_name = classifier.split('/')[-1].split('_')[0]

        for _, row in df.iterrows():
            metric1 = row['bal_acc']
            metric2 = row['f1_score']
            metric3 = row['precision']
            metric4 = row['recall']

            # Create a new dictionary with the classifier name, balanced accuracy, and f1-score
            data_row = {'Classifier': classifier_name, 'bal_acc': metric1, 'f1_score': metric2, 'precision': metric3,
                        'recall': metric4}

            # Append the row to the data list
            data.append(data_row)

    # Convert the data list to a DataFrame
    data = pd.DataFrame(data)
    return data


def manova_all_metrics(a: Classifiers, b: Classifiers):
    data = metrics_to_one_csv()
    data = data[(data['Classifier'] == a.value) | (data['Classifier'] == b.value)]
    maov = MANOVA.from_formula('bal_acc + f1_score + \
                                precision + recall  ~ Classifier', data=data)

    test = maov.mv_test()

    x = pd.DataFrame((test.results['Classifier']['stat']))
    return x

@dekor
def boxplot(classifier: Classifiers, metric: Metrics):
    data = metrics_to_one_csv()
    data = data[metric.name][(data['Classifier'] == classifier.value)]
    column_data = list(data)
    matrix_data = np.array(column_data).reshape(5, 5)
    plt.boxplot(matrix_data.T, showmeans=True,
                meanline=True)  # transpose the array to have iterations on x-axis and folds on y-axis
    plt.xlabel('Iteration')
    plt.ylabel(metric.value.capitalize())
    plt.title(f'{metric.value.capitalize()} distribution for {classifier.name} classifier')
    plt.xticks(np.arange(1, matrix_data.shape[0] + 1), labels=np.arange(1, matrix_data.shape[0] + 1))
    plt.show()






