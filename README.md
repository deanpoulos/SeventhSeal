# Seventh Seal
## Dedication
I dedicate this project to my friend Lyndon who has helped inspire entreprenuership.
## Introduction
At this point I wish to explore various trading strategies which utilise some
form of machine learning to successfully predict price action of cryptocurrency
assets.
## Guide
 - Insert your Binance API/Secret key to keys.py (this will be ignored by git add)
 - Download new .csv data using:
 `python3 -i API-Interface.py` 
 > `get_all_binance("BTCUSDT", "1m", save=True")`
 - Plot 1d BTCUSDT price data by running `python3 Plotter.py`
## Todo
 - [x]  Interface with Binance API
 - [x]  Plot historical price data of Bitcoin
 - [ ]  1d dataset into training and test data
 - [ ]  DNN skeleton with data loaders
 - [ ]  1 layer neural network