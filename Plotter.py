import plotly.graph_objects as go
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Plotter:
    def __init__(self, time):
        self.data = pd.read_csv('data/BTCUSDT-' + time + '-data.csv', index_col=0, parse_dates=True)
        self.data.shape

    def plot_mpl_candlesticks(self):
        ''' Use matplotlib finance OHLC candlestick graph. '''
        data = self.data
        mpf.plot(data, type='candle', mav=(50,100,200), volume=True)

    def plot_go_candlesticks(self):
        ''' Render interactive candlestick graph using plotly. '''
        data = self.data
        
        fig = go.Figure(data=[go.Candlestick(x=data['timestamp'],
                open=data['open'], high=data['high'], low=data['low'], close=data['close'])])

        fig.update_layout(
            title='USDT/BTC Price History',
            yaxis_title='BTC Price ($USD)',
            xaxis_title='Date',
        )

        fig.show()
    
    def plot_simple_graphs(self):
        # down-sample labels on the x-axis
        xlabels = []
        for i in range(len(data['timestamp'])):
            if (i % 100 == 0): 
                xlabels.append(data['timestamp'][i])
            else:
                xlabels.append('')

        plt.plot(xlabels, data['high'])
        plt.plot(xlabels, data['low'])
        plt.xlabel('Timestamp')
        plt.ylabel('BTC Price ($USD)')
        plt.title('USDT/BTC Price History')
        plt.show()


def plot():
    plotter = Plotter('1d')
    plotter.plot_mpl_candlesticks()