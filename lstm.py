import cv2
import torch, random, sklearn
import numpy as np
from torch import nn, optim
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from torch.nn import ModuleList, functional as F
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm_notebook
from tqdm.auto import tqdm
import json, pickle
from collections import deque
from ultralytics import YOLO
from torch.autograd import Variable 
from matplotlib import pyplot as plt
import torch.utils.data as Data
import pandas as pd

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, layer_widths, seq_length, dist):
        super(LSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.layers = ModuleList()
        self.layers.append(nn.Linear(hidden_size, layer_widths[0]))
        for i in range(1, len(layer_widths)):
            self.layers.append(nn.Linear(layer_widths[i - 1], layer_widths[i]))
        self.layers.append(nn.Linear(layer_widths[-1], num_classes))
        for l in self.layers:
            nn.init.kaiming_normal_(l.weight)

        self.softmax = nn.Softmax()
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)
        self.loss = nn.CrossEntropyLoss(dist)
        self.act = nn.Tanh()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        # print(hn.shape)
        # out = out.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out=out[:, -1, :]
        # print(out[:5])
        # out = self.relu(out)
        for layer in self.layers[:-1]:
            out = layer(out)
            out = self.act(out)
        out = self.layers[-1](out) #Final Output
        # print(out[:5])

        out = self.softmax(out)
        # print(out[:5])

        return out
    def train(self, n_epochs, X, y, X_val, y_val, intervals=30):
        losses = []
        min_val_loss = float('inf')
        for epoch in tqdm_notebook(range(n_epochs)):
            # for X_batch, y_batch in loader:
            y_pred = self(X)
            if epoch %50==0:
                print(y_pred.sum(dim=0))
            # print(y_pred.shape)
            loss = self.loss(y_pred, y)
            losses.append(loss.detach())
            if epoch % intervals == 0:
                val_predictions = self(X_val)
                
                val_loss = self.loss(val_predictions, y_val)
                min_val_loss = min(min_val_loss, val_loss)
                if val_loss > min_val_loss * 1.05:
                    print(f"Stopped at Epoch {epoch}\nLoss: {val_loss}")
                    break
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        plt.plot(list(range(epoch+1)), losses)
        plt.show()