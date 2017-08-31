#Module containing chart creating prototype.
#Was not used

import numpy as np
import matplotlib.pyplot as plt

#numAr kann auch mehr dimensional sein.
#Plot interpretiert automatisch zwei Spalten als x,y-Werte für eine Linie.
#So können dann auch alle Gewichte angezeigt werden

def createChart(numAr, title, ylabel):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.figure(1)
    plt.plot(numAr)
    plt.show()

if __name__ == '__main__':
    yvalues = np.random.rand(100)
    createChart(yvalues, 'Testtitel', 'Gewicht')
