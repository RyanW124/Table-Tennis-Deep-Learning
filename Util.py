import cv2
import torch, random, sklearn
import numpy as np
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from torch.nn import ModuleList, functional as F
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
import json, pickle

K = 3
S = 2
P = 0
class NN(nn.Module):
    def __init__(self, channels, layers):
        super(NN, self).__init__()
        self.conv, self.dense = nn.ModuleList(), nn.ModuleList()
        self.ylim = 1000
        for i in range(1, len(channels)):
            self.conv.append(nn.Conv2d(in_channels=channels[i-1], out_channels=channels[i],
                                kernel_size=K, padding=P))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=S)
        for i in range(1, len(layers)):
            self.dense.append(nn.Linear(in_features=layers[i-1], out_features=layers[i]))
        for layer in self.conv+self.dense:
            nn.init.kaiming_normal_(layer.weight)
        self.loss_f = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())
        self.to("cuda:0")
        

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        for layer in self.conv:
            x = layer(x)
            x = self.maxpool(x)
        x = torch.flatten(x, 1)
        for layer in self.dense[:-1]:
            x = layer(x)
            x = self.relu(x)
        x = self.dense[-1](x)
        return x
    def get_loss(self, x, y):
        predictions = self(x)
        loss = self.loss_f(predictions, y)
        return loss

    def my_train(self, train_x, train_y, val_x, val_y, epochs, intervals=20, batch_size=100):
        try:
            min_epochs = epochs/10
            epochs = tqdm(range(epochs))
            min_val_loss = float('inf')
            losses = []
            batches_X = np.array_split(train_x, np.ceil(len(train_x)/batch_size))
            batches_y = np.array_split(train_y, np.ceil(len(train_y)/batch_size))
            for epoch in epochs:
                total_loss = 0
                for X, y in zip(batches_X, batches_y):
                    self.zero_grad()
                    loss = self.get_loss(X, y)
                    total_loss += float(loss)
                    loss.backward()
                    self.optimizer.step()
                losses.append(total_loss)
                if epoch >= min_epochs and epoch % intervals == 0:
                    val_loss = self.get_loss(val_x, val_y)
                    min_val_loss = min(min_val_loss, val_loss)
                    if val_loss > min_val_loss * 1.05:
                        print(f"Stopped at Epoch {epoch}\nLoss: {val_loss}")
                        break
            plt.ylim((0, self.ylim))
            plt.plot(range(len(losses)), losses)
            plt.show()
        except KeyboardInterrupt:
            torch.cuda.empty_cache()
class NN2(NN):
    def __init__(self, channels, layers):
        super().__init__(channels, layers)
        self.softmax = nn.Softmax(1)
        self.loss_f = nn.CrossEntropyLoss()
        self.ylim = .5
    # def forward(self, x):
    #     x = super().forward(x)
    #     x = self.softmax(x)
    #     return x
    def predict(self, x):
        squeeze = False
        if x.dim() == 3:
            x.unsqueeze(0)
            squeeze = True
        x = self(x)
        x = self.softmax(x)
        if squeeze:
            x = x.squeeze(0)
        return x
    def ratio(self, x, y):
        predictions = self.predict(x)
        correct = 0
        total = 0
        for p, l in zip(predictions, y):
            print(p, l)
            if (l[0] >= .5) == (p[0] >= .5):
                correct += 1
            total += 1  
        return correct/total
    
device = torch.device("cuda:0")

def preprocess(X, y):
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y)
    train_X = torch.from_numpy(train_X).to(device)
    val_X = torch.from_numpy(val_X).to(device)
    test_X = torch.from_numpy(test_X).to(device)

    train_y = torch.from_numpy(train_y).to(device)
    val_y = torch.from_numpy(val_y).to(device)
    test_y = torch.from_numpy(test_y).to(device)
    return train_X, val_X, test_X, train_y, val_y, test_y

def load(file_name, init_val):
    data = None
    try:
        with open(file_name, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = init_val
    return data
def imshow(img):
    cv2.imshow("i", img)
    cv2.waitKey(0)
    cv2.destroyWindow("i")

def lowhigh(w, margin):
    return int(w/2-margin*w), int(w/2+margin*w)