from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    # linear function followed by log_softmax
    def __init__(self, il=1, hl=100, ol=1):
        super(LSTM, self).__init__()
        # store # of hidden layers
        self.hl = hl
        # start with LSTM layer from input space to hidden space
        self.lstm = nn.LSTM(il, hl)
        # add linear function to transform hidden to output
        self.linear = nn.Linear(hl, ol)
        self.hidden_cell = (torch.zeros(1,1,self.hl),
                            torch.zeros(1,1,self.hl))


    def forward(self, x):
        # propagate data through the network
        lstm_out, self.hidden_cell = self.lstm(x.view(len(x), 1, -1), self.hidden_cell)

        # make final prediction
        pred = self.linear(lstm_out.view(len(x), -1))

        return pred[-1]
        
def main():
    # use GPU is available
    device = torch.device('cuda')

    # read .csv data frame using pandas
    df = pd.read_csv('data/BTCUSDT-1d-data.csv', index_col=0, parse_dates=True)

    # transform `open` price column into a pytorch tensor.
    data = torch.tensor(df['open'].values)

    # store an ndarray version as well
    all_data = df['open'].to_numpy()

    # split data into a training set and test set
    test_data_size = 100
    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]

    # normalize training data between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1,1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1,1))
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

    # determine short-term memory window
    train_window = 91
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    # instantiate model, MSE loss function and adaptive momentum optimizer
    model = LSTM()
    #model.to(device)
    lf = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, device, train_inout_seq, lf, optimizer)
    pred = predict(model, device, train_data_normalized, train_window, scaler)

    x = np.arange(len(all_data) - len(pred), len(all_data), 1)
    print(x)

    plt.title('Price vs Prediction')
    plt.ylabel('Open Price')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(all_data)
    plt.plot(x,pred)
    plt.show()

def train(model, device, data, lf, optimizer):
    # number of times iterated over training set
    epochs = 3
    for i in range(epochs):
        for seq, labels in data:
            #seq, labels = seq.to(device), labels.to(device)
            seq.requires_grad=True
            labels.requires_grad=True
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hl),
                            torch.zeros(1, 1, model.hl))

            y_pred = model(seq)

            single_loss = lf(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i%1 == 0:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

def predict(model, device, tdn, window, scaler):
    fut_pred = 365
    test_inputs = tdn[-window:].tolist()

    model.eval()

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hl),
                            torch.zeros(1, 1, model.hl))
            test_inputs.append(model(seq).item())

    actual_predictions = scaler.inverse_transform(np.array(test_inputs[window:] ).reshape(-1, 1))
    return actual_predictions


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

if __name__ == '__main__':
    main()