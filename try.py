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
from tqdm import tqdm

cam = cv2.VideoCapture("game_1.mp4")
cam.set(cv2.CAP_PROP_POS_FRAMES, 14)
cv2.imshow("hi", cam.read()[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
cam.release()

class NN(nn.Module):
    def __init__(self, channels, layers):
        super(NN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            self.layers.append(nn.Conv2d(in_channels=channels[i-1], out_channels=channels[i],
                                kernel_size=(5, 5)))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(in_features=layers[i-1], out_features=layers[i]))
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight)
        self.loss_f = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())


    def forward(self, x):
        for layer in self.layers[:-1]:

            x = layer(x)
            x = nn.ReLU()(x)


        x = self.layers[-1](x)
        return x


    def my_train(self, train, validation, intervals=10, epochs=10000):
        epochs = tqdm(range(epochs))
        min_val_loss = float('inf')
        for epoch in epochs:
            self.zero_grad()

            predictions = self(train)
            loss = self.loss_f(predictions, train)
            if epoch % intervals == 0:
                val_predictions = self(validation)

                val_loss = self.loss_f(val_predictions, validation)
                min_val_loss = min(min_val_loss, val_loss)
                if val_loss > min_val_loss * 1.05:
                    print(f"Stopped at Epoch {epoch}\nLoss: {val_loss}")
                    break
            loss.backward()
            self.optimizer.step()
