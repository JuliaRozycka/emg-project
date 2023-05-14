# This is a sample Python script.
import Visualizator as v
# ss Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage


def print_hi(name):  # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    v.visualize_signal("osoba_2_prawa_p1.csv")
    cred = credentials.Certificate("emg-signals-firebase-adminsdk-q9auf-cc504f173c.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'emg-signals.appspot.com'})
    bucket = storage.bucket()
    blob = bucket.blob("o2/osoba_2_lewa_p1.csv")
    blob.download_to_filename('dane1.csv')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
