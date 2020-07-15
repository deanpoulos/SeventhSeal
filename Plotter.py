import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('BTCUSDT-1d-data.csv')
plt.plot(data['timestamp'], data['high'])
plt.plot(data['timestamp'], data['low'])
plt.show()

