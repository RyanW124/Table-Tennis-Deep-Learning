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
from lstm import LSTM


posenet = YOLO("yolov8n-pose.pt")

def classify(frames, lstm):
    X =[]
    for im in frames:
        im = cv2.resize(im, (640, 480))
        results = posenet.predict(im, half=True,verbose=False)
        # results[0].show()
        boxes = results[0].boxes.xywh
        if boxes.shape[0] < 2:
            X.append(np.zeros((12, 2)))
            continue
        i = max(range(boxes.shape[0]), key=lambda x: boxes[x][0])
        keys = results[0].keypoints.xy[i].cpu().numpy()
        nose = keys[0]
        for i, t in enumerate(keys):
            x, y= t
            if x == 0 and y == 0:
                continue
            keys[i] = (keys[i]-nose)/200
        X.append(keys[5:])
    X = np.array([X])
    X = X.reshape(-1, look_back//skip+1, 24)
    # df = pd.DataFrame(X)
    X = torch.tensor(X.astype(np.float32))
    return lstm(X)[0].argmax()

# Load a model
g = 4
cam = cv2.VideoCapture(f"test_{g}.mp4")
model = YOLO("yolov8n.pt")
lstm = torch.load("lstm.pt")

cont = True
timer = 60
text = ""
look_back=60
skip = 5

first = True
W,H,OW, OH = [None]*4
ball_color=np.array([177, 159, 113])
low = 110
cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
timestamps, frames = [], []
last_time=-9999
# cam.set(cv2.CAP_PROP_FPS,10)
count = 0
prevs = deque([False]*8)
state = 0
cn = float('inf')
prev_frames = deque()
prev = None
blank = None
def err(im1, color, e):
    im2 = np.zeros(im1.shape, np.uint8)
    im2[:] = color
    thresh = cv2.absdiff(im1, im2)
    _, ret = cv2.threshold(np.clip(thresh.sum(2), 0, 255).astype('uint8'), e, 255, cv2.THRESH_BINARY_INV)
    return ret

while cont:
    cont, oimg = cam.read()
    if not cont:
        break
    img = oimg.copy()
    if first:
        OH, OW, _ = img.shape
        W = OW
        H = W*OH//OW
        vid = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), cam.get(cv2.CAP_PROP_FPS), (OW, OH))
        img = cv2.resize(img, (W, H))
        
        results = model(img)
        a = None
        m = 0
        for i, (_, _, w, h) in enumerate(results[0].boxes.xywh):
            if w/h > m:
                m = w/h
                a = i
        a = results[0].boxes.xyxy[a].cpu().numpy().astype(int)
        x1, y1, x2, y2 = a
        mask = cv2.inRange(cv2.resize(img, (W, H))[y1:y2, x1:x2], (80, 0, 0), (255, 100, 100))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)
        img = img[y1:y2, x1:x2]
        # thresh2 = cv2.inRange(img, ball_color-50, ball_color+70)
        thresh2 = err(img, ball_color, 100)
        
        ball_color = cv2.mean(img, thresh2)
        blank = cv2.bitwise_and(img, img, mask=mask)
        ball_color = np.array([*ball_color][:-1]).astype(int)
        first = False
        continue
    
    img = cv2.resize(img, (W, H))
    img = img[y1:y2, x1:x2]

    # bg sub
    thresh = cv2.absdiff(blank, cv2.bitwise_and(img,img,mask = mask))
    kernel = np.ones((2,2),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
    ret, thresh1 = cv2.threshold(np.clip(thresh.sum(2), 0, 255).astype('uint8'), 100, 255, cv2.THRESH_BINARY) 
    thresh2 = err(img, ball_color, 150)
    kernel = np.ones((2,2),np.uint8)
    thresh2 = cv2.dilate(thresh2, kernel, iterations=5) 
    
    thresh = cv2.bitwise_and(thresh1,thresh1,mask = thresh2)
    params = cv2.SimpleBlobDetector_Params()
 
    
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 1200
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.maxCircularity = 1
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(cv2.bitwise_not(thresh))
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("h", im_with_keypoints)
    # cv2.imshow("0", thresh)
    # cv2.imshow("1", thresh1)
    # cv2.imshow("2", thresh2)
    # # break
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     print(cam.get(cv2.CAP_PROP_POS_FRAMES))
    #     break
    # continue
    pos, _ = cv2.KeyPoint_convert(keypoints)[0] if len(keypoints) else (None, None)
    # print(pos)
    # print(pos)00000000
    if pos is None or cn > 3:
        count += int(prevs[-1]) - int(prevs.popleft())
        prevs.append(prevs[-1])
    else:
        left = pos < prev
        count += int(left) - int(prevs.popleft())
        prevs.append(left)
    if count > 4:
        if state == 0 and cam.get(cv2.CAP_PROP_POS_MSEC)/1000-last_time>0.8:
            state = 1
            shot_type=classify(prev_frames, lstm)
            text = ["FH top", "BH top", "FH back", "BH back", "FH serve", "BH serve"][shot_type]
            timer = 60
            last_time = cam.get(cv2.CAP_PROP_POS_MSEC)/1000
            print(text)
            
    else:
        state = 0
    # print(count, pos)
    # print(prevs, state, count > 4)
    cn += 1
    if pos: 
        cn = 0
        prev = pos
    if cam.get(cv2.CAP_PROP_POS_FRAMES) % skip:
        prev_frames.append(img)
        if len(prev_frames) > look_back//skip+1:
            prev_frames.popleft()
    if timer:
        timer -= 1
        if not timer: text = ""
    font = cv2.FONT_HERSHEY_SIMPLEX

            # org
    org = (50, 120)

    # fontScale
    fontScale = 5

    # Blue color in BGR
    color = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    oimg = cv2.putText(oimg, text, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    vid.write(oimg)
    # cv2.imshow('frame',oimg)
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     break
    # if cam.get(cv2.CAP_PROP_POS_MSEC)/1000>30:
    #     break
cv2.destroyAllWindows()
vid.release()
cam.release()