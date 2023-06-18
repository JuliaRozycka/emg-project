import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier


def extract_features_to_csv(directory):
    """
    Okej, więc problemy:

    1) Czy dane mają być jako wektory cech czy jedna wartość dla całego sygnału z danej klasy?

    2) Czy normalizować w jakiś sposób wartości tych cech?
    https://www.mdpi.com/1424-8220/22/13/5005, https://www.intechopen.com/chapters/40113
    podzielić kwantylowo ni wszystko co z 4 ćwiartki usunąć
    odcinamy do którejś wartości i min-max
    jeśli będzie za niska wartość to przesterujemy sygnał

    nie problem multipklasowy tylko klasa versus szum, albo klasa versus reszta.
    Klasa versus reszta będzie lepsza
    OVA - podejści, budujemy coś jako zepsół który robi każda klasa versus reszta.
    OVO - każdy z każdym i musimy sobie to połączyć
    zbinaryzowac zbiór i użyć klasyfikatorów podstawowych
    3) Czy wymagany one hot encoding klas?

    4) Problem jest tez taki, że lity mają różne długości dla różnych sygnałów,
    bo czasami da się wyekstrachować 7 sekund a czasem jedną, jak radzić sobie z listami

    *************************************************************************************************************
    UWAGAAA TO EKSTRACHUJE DO PLIKU CECHY W POSTACI WEKTORA CECH NALEZY TO ZMIENIĆ
    Zostały zmienione funkcje ekstrakcji cech, tak aby ekstrachowały jedną cechę z całego sygnału, jednak:
    TODO: Należy przeprowadzić ekstrakcję patrz klasa main.py funckja @extracting_features()
    TODO: Zmienić aby odczytywało z plików w directory features i zapisywało pojedyncze wartości MAV, RMS...
    *************************************************************************************************************

    """

    # directory  is features/
    # iterate over folders 1-18 in features
    df = pd.DataFrame(columns=['RMS','MAV','IEMG','VAR','WL','WAMP','FMN','FMD','Class'])

    for _class in range(1,19):
        class_path = os.path.join(directory, str(_class))
        for file in os.listdir(class_path):
            f = os.path.join(class_path, file)
            if os.path.isfile(f):
                extracted_features = pd.read_csv(f)

                row = [
                    extracted_features['RMS'].values[0],
                    extracted_features['MAV'].values[0],
                    extracted_features['IEMG'].values[0],
                    extracted_features['VAR'].values[0],
                    extracted_features['WL'].values[0],
                    extracted_features['WAMP'].values[0],
                    extracted_features['FMN'].values[0],
                    extracted_features['FMD'].values[0],
                    int(_class)
                ]

                df.loc[len(df)] = row

            #break
            # Break after reading first file - to jest do usunięcia jak ustalimy co bierzemy za faetures i czy lista czy co, bo wsm tak sobie
            # myślę że niepotzrebnie chyba bierzemy tyle podobnych features z time domain, można wywalić jakieś 3 i dodać 2 z frequency domain
    df.to_csv('features_for_training.csv', index=False)
    return df


def train_SVM(directory: str):

    data = pd.read_csv(directory)
    features = list(data.columns.values)[:-1]
    X = data[features]
    y = data['Class']
    print(X.shape)
    print(y.shape)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    select = SelectKBest(chi2, k=6)
    X_new = select.fit_transform(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_new,y,test_size=0.2,random_state=42)

    svm = SVC()
    model = OneVsRestClassifier(svm)

    model.fit(X_train,y_train)

    prediction = model.predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, prediction)
    print("Balanced Accuracy:", balanced_accuracy)









