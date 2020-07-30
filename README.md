# Seventh Seal
## Introduction
At this point I wish to explore various trading strategies which utilise some
form of machine learning to successfully predict price action of cryptocurrency
assets.
## Guide
 - Insert your Binance API/Secret key to a file `mykeys.py` using the `mykeys_template.py` as a guide.
 - Interface with all data using `python3 -i DataInferface.py`
 - Within the interfae module, plot data using `plot()` and update .csv using `load_data(1d)`

## Todo
 - [x]  Interface with Binance API
 - [x]  Plot historical price data of Bitcoin
 - [ ]  Split 1d dataset into training and test data
 - [ ]  DNN skeleton with data loaders
 - [ ]  1 layer neural network

## Ideas
 - Make a success predictor for leveraged cryptocurrency contracts which can identify patterns which precede volatility and price growth. Suppose there is a market reaction to a certain piece of news which causes price growth and low volatility. A successful neural network should be able to identify this news and take advantage of a small-margin leveraged contract.
 - Make a price action predictor for long-term intraday trading which can identify patterns which precede price growth or decline.