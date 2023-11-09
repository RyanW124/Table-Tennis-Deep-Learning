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

def preprocess(X, y, shuffle=True):
    train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=shuffle)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, shuffle=shuffle)
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
class NN3(nn.Module):
    def __init__(self, channels, layers, n_features, n_hidden, seq_len, n_layers):
        super().__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
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
        self.loss_f = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())
        self.to("cuda:0")
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True
        )
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, 1, self.n_hidden, device="cuda:0"),
            torch.zeros(self.n_layers, 1, self.n_hidden, device="cuda:0")
        )
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        for layer in self.conv:
            x = layer(x)
            x = self.maxpool(x)
        lstm_out, self.hidden = self.lstm(
            x.flatten(1).unsqueeze(0),
            self.hidden
        )
        x = lstm_out.view(self.seq_len-1, len(x), self.n_hidden)[-1]
        for layer in self.dense[:-1]:
            x = layer(x)
            x = self.relu(x)
        x = self.dense[-1](x)
        return x
    
    def train_model(self, train_data, train_labels, val_data=None, val_labels=None, num_epochs=100):
        train_hist = []
        val_hist = []
        for t in range(num_epochs):

            epoch_loss = 0

            for idx, seq in enumerate(train_data):

                self.reset_hidden_state()

                # seq = torch.unsqueeze(seq, 0)
                y_pred = self(seq)
                loss = self.loss_f(y_pred[0].float(), train_labels[idx]) # calculated loss after 1 step

                # update weights
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                epoch_loss += loss.item()
SIZE = 200
def make_data(end, window, rate):
    cap = cv2.VideoCapture("game_1.mp4")
    size = window//rate
    X = np.zeros((end//window, size, SIZE, SIZE), dtype=np.float32)
    y = np.zeros((end//window))
    for i in range(0, end-end%window, rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = cap.read()
        w, h, _ = frame.shape
        frame = frame[:, :w//2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        frame = cv2.resize(frame, (SIZE, SIZE))
        X[i//window, i%window//rate] = frame
    with open("timestamp.dat", "r") as f:
        _dict = json.load(f)
        keys = list(_dict.keys())
    count = np.zeros((3))
    current = 0
    for i in range(end):
        if i >= float(keys[current]):
            current += 1
        count[_dict[keys[current]]] += 1
        if i % window == window - 1:
            y[i//window] = np.argmax(count)
            print(count)
            count = np.zeros((3))
    return X, y


        
