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