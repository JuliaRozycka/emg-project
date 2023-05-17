from Utils import read_data
from Visualizator import visualize_signal

if __name__ == '__main__':
    data = read_data('dane_testowe.csv')
    visualize_signal(data)





