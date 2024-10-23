import pickle
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
import json, pickle
from collections import deque
from ultralytics import YOLO

#0 FH top
#1 BH top
#2 FH back
#3 BH back
#4 FH serve
#5 BH serve
#/ NA
game=1
with open(f'frames{game}.dat', 'rb') as f:
    data = pickle.load(f)
cam = cv2.VideoCapture(f"game_{game}.mp4")
look_back=60
shots=[]
print(len(data))
save = False
if save:
    with open(f"shots{game}.dat", 'rb') as f:
        shots = pickle.load(f)
index = len(shots)
while index < len(data):
    frame = data[index]
    if frame < look_back: 
        shots.append(-1)
        index+=1
        continue
    print(frame)
    cam.set(cv2.CAP_PROP_POS_FRAMES, frame-look_back)
    for i in range(look_back):
        _, im = cam.read()
        cv2.imshow('frame',im)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    else:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            continue
        if key == ord('q'):
            break
        shots.append(key-ord('0'))
        index+=1
        continue
    break
cv2.destroyAllWindows()
with open(f"shots{game}.dat", 'wb') as f:
    pickle.dump(shots, f)
print(shots)
cam.release()
