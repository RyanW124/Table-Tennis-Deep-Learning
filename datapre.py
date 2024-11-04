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

look_back=60
skip = 5
W, H = 640, 480

if __name__ == "__main__":
    model = YOLO("yolov8n-pose.pt")
    X = []
    Y = []
    save = False

    if save:
        with open("data.dat", 'rb') as f:
            X, Y = pickle.load(f)
    for game in [1, 2, 3, 4, 5]:
        with open(f'frames{game}.dat', 'rb') as f:
            frames = pickle.load(f)
        with open(f'shots{game}.dat', 'rb') as f:
            label = pickle.load(f)
        cam = cv2.VideoCapture(f"game_{game}.mp4")
        
        for frame, l in tqdm_notebook(list(zip(frames, label))):
            if frame < look_back:
                continue
            start = frame-look_back
            current =[]
            for i in range(look_back//skip+1):
                cam.set(cv2.CAP_PROP_POS_FRAMES, start + i*skip)
                _, im = cam.read()
                im = cv2.resize(im, (W, H))
                results = model.predict(im, half=True,verbose=False)
                # results[0].show()
                boxes = results[0].boxes.xywh
                if boxes.shape[0] < 2:
                    break
                i = max(range(boxes.shape[0]), key=lambda x: boxes[x][0])
                keys = results[0].keypoints.xy[i].cpu().numpy()
                _, _, w, h =results[0].boxes.xywh[i].cpu().numpy()
                rarea = np.sqrt(w*h)
                nose = keys[0]
                for i, t in enumerate(keys):
                    x, y= t
                    if x == 0 and y == 0:
                        continue
                    keys[i] = (keys[i]-nose)/rarea
                current.append(keys[5:])
            else:
                Y.append(l)
                X.append(current)
    with open("data.dat", 'wb') as f:
        pickle.dump((X, Y), f)
def process(X, y):
    X = np.array(X)
    X = X.reshape(-1, look_back//skip+1, 24)
    # df = pd.DataFrame(X)
    X = torch.tensor(X.astype(np.float32))
    y = nn.functional.one_hot(torch.tensor(y))
    y = y.type(torch.float32)


    return X, y
