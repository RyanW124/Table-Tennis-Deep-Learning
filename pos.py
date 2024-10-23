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

# Load a model
g = 3
cam = cv2.VideoCapture(f"game_{g}.mp4")
model = YOLO("yolov8n.pt")
cont = True
first = True
W,H,OW, OH = [None]*4
ball_color=np.array([177, 159, 113])
low = 110
frame = 1500
cam.set(cv2.CAP_PROP_POS_FRAMES, frame)
# cam.set(cv2.CAP_PROP_FPS,10)
count = 0
prevs = deque([False]*8)
state = 0 
timestamps, frames = [],[]
cn = float('inf')
prev = None
blank = None
def err(im1, color, e):
    im2 = np.zeros(im1.shape, np.uint8)
    im2[:] = color
    thresh = cv2.absdiff(im1, im2)
    _, ret = cv2.threshold(np.clip(thresh.sum(2), 0, 255).astype('uint8'), e, 255, cv2.THRESH_BINARY_INV)
    return ret


while cont:
    cont, img = cam.read()
    frame += 1

    if first:
        OH, OW, _ = img.shape
        W = OW
        H = W*OH//OW
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
        # ball_color = img[np.transpose(thresh2.nonzero())].mean(axis=0).mean(axis=0)
        ball_color = np.array([*ball_color][:-1]).astype(int)
        # cv2.imshow("w", blank)
        # cv2.imshow("j", cv2.bitwise_and(img, img, mask=thresh2))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        first = False
        cam.set(cv2.CAP_PROP_POS_FRAMES, 5200)#11565
        # cam.set(cv2.CAP_PROP_POS_MSEC, 105*1000)
        # print(cam.get(cv2.CAP_PROP_POS_MSEC))
        # bar = tqdm(total=cam.get(cv2.CAP_PROP_FRAME_COUNT), position=0, leave=True)

        continue
    if not cont:
        break
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
    params.minCircularity = 0.7
    params.maxCircularity = 1

    
    # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.01
    
    # Filter by Inertia
    # params.filterByInertia = True
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(cv2.bitwise_not(thresh))
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("h", im_with_keypoints)
    cv2.imshow("0", thresh)
    cv2.imshow("1", thresh1)
    cv2.imshow("2", thresh2)
    # break
    if cv2.waitKey(0) & 0xFF == ord('q'):
        print(cam.get(cv2.CAP_PROP_POS_FRAMES))
        break
    continue
    pos, _ = cv2.KeyPoint_convert(keypoints)[0] if len(keypoints) else (None, None)
    # print(pos)
    # print(pos)00000000
    if pos is None or cn > 2:
        count += int(prevs[-1]) - int(prevs.popleft())
        prevs.append(prevs[-1])
    else:
        left = pos < prev
        count += int(left) - int(prevs.popleft())
        prevs.append(left)
    if count > 4:
        if state == 0 and (not timestamps or cam.get(cv2.CAP_PROP_POS_MSEC)/1000-timestamps[-1]>0.8):
            state = 1
            timestamps.append(cam.get(cv2.CAP_PROP_POS_MSEC)/1000)
            frames.append(cam.get(cv2.CAP_PROP_POS_FRAMES))
    else:
        state = 0
    # print(prevs, state, count > 4)
    cn += 1
    if pos: 
        cn = 0
        prev = pos
    # if cam.get(cv2.CAP_PROP_POS_MSEC)/1000>200:
    #     break
cv2.destroyAllWindows()
for i in timestamps:
    print(f"{i:.2f}")
with open(f'frames{g}.dat', 'wb') as f:
    pickle.dump(frames, f)