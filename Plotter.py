import pandas as pd
from matplotlib import pyplot as plt

class Plotter:
    def __init__(self, time):
        self.data = pd.read_csv('BTCUSDT-' + time + '-data.csv')

    def plot_candles():
        plt.plot(data['timestamp'], data['high'])
        plt.plot(data['timestamp'], data['low'])
        plt.show()
