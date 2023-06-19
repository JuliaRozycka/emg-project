import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import normaltest, shapiro

def t_test_corrected(a, b, J=5, k=5):
    """
    Corrected t-test for repeated cross-validation.
    input, two 2d arrays. Repetitions x folds
    As default for 5x5CV
    """
    if J*k != a.shape[0]:
        raise Exception('%i scores received, but J=%i, k=%i (J*k=%i)' % (
            a.shape[0], J, k, J*k
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
    if len(data)>=20:
        stats, p =normaltest(data)
    else:
        stats, p = shapiro(data)

    if p > 0.05:
        isnormal=True
    else:
        isnormal=False
    return stats, p, isnormal