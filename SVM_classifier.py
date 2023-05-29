import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

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


    """

    # directory  is features/
    # iterate over folders 1-18 in features
    df = pd.DataFrame(columns=['RMS','MAV','SSI','IEMG','VAR','WL','WAMP','Class'])

    for _class in range(1,19):
        class_path = os.path.join(directory, str(_class))
        for file in os.listdir(class_path):
            f = os.path.join(class_path, file)
            if os.path.isfile(f):
                extracted_features = pd.read_csv(f)
                rms_list = extracted_features['RMS'].tolist()
                mav_list = extracted_features['MAV'].tolist()
                ssi_list = extracted_features['SSI'].tolist()
                iemg_list = extracted_features['IEMG'].tolist()
                var_list = extracted_features['VAR'].tolist()
                wl_list = extracted_features['WL'].tolist()
                wamp_list = extracted_features['WAMP'].tolist()
                row = [rms_list, mav_list, ssi_list, iemg_list, var_list, wl_list, wamp_list,_class]
                df.loc[len(df)] = row

            break
            # Break after reading first file - to jest do usunięcia jak ustalimy co bierzemy za faetures i czy lista czy co, bo wsm tak sobie
            # myślę że niepotzrebnie chyba bierzemy tyle podobnych features z time domain, można wywalić jakieś 3 i dodać 2 z frequency domain
    df.to_csv('features_for_training.csv', index=False)
    return df


def train_SVM(directory: str):

    feature_names = ['MAV', 'SSI', 'RMS', 'IEMG', 'VAR', 'WL', 'WAMP']
    df = pd.read_csv(directory)

    # Trained classifiers
    classifiers = {}

    # Iterate over the feature names
    for feature in feature_names:
        y = df['Class']
        X = df[feature]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create an SVM classifier
        svm = SVC()

        # Train the SVM classifier
        svm.fit(X_train, y_train)

        # Store the trained classifier in the dictionary
        classifiers[feature] = svm

        # Make predictions on the test set
        y_pred = svm.predict(X_test)

        # Print classification report
        print(classification_report(y_test, y_pred))

    return classifiers
