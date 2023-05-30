import math

import numpy as np

# Jeżeli nie bierzemy pod uwagę okien czasowych
''' Okej to są najprostsze funckje, myślę że dla naszych potrzeb można to również rozszrzyć biorąc dwa argumenty - mp./
    ten z czasami start-end i dla wszystkich ruchów od razu to wyznaczać. Druga sprawa czy dodajemy tutaj/
    z uwzględnieniem okien czasowych - ponieważ jeśli tak, to przez wiele cech nie ma sensu powtarzać kilkakrotnie/
    kodu podziału data na okna tylko jakoś osobno. No to tak - tyle do ustalenia. 
'''


def fRootMeanSquare(data):
    # dRMS1=math.sqrt(np.mean(x**2 for x in data))
    dRMS = math.sqrt(np.mean(np.power(data, 2)))
    return dRMS


def fMeanAbsoluteValue(data):
    dMAV = np.mean(np.abs(data))
    return dMAV


def fIntegrated(data):
    dIEMG = np.sum(np.abs(data))
    return dIEMG


def fVariance(data):
    N = len(data)
    dVAR = (1 / (N - 1)) * np.sum(np.power(data, 2))
    return dVAR


def fWillisonAmplitude(data, opts=None):
    threshold = 0.01  # threshold
    if opts is not None and 'threshold' in opts:
        threshold = opts['threshold']
    N = len(data)
    dWA = 0
    for k in range(N - 1):
        if abs(data[k] - data[k + 1]) > threshold:
            dWA += 1
    return dWA


def fSimpleSquareIntegral(data):
    dSSI = np.sum(np.power(data, 2))
    return dSSI


def fWaveformLength(data):
    N = len(data)
    dWL = 0
    for k in range(1, N):
        dWL += abs(data[k] - data[k - 1])
    return dWL
