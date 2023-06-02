from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import classification_report

def train_kNN(directory: str):
    df = pd.read_csv(directory)
    x = df[['RMS', 'MAV', 'IEMG', 'VAR', 'WL', 'WAMP', 'FMN', 'FMD']].values
    y = df['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return