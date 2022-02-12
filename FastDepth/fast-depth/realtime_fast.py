import os
from telnetlib import OUTMRK
import cv2
import time
import numpy as np

import torch
from torchvision import transforms

import imagenet.mobilenet as im
from collections import OrderedDict
import utils

model_name = "mobilenet-nnconv5dw-skipadd-pruned.pth.tar"
model_path = os.path.join("../results",model_name)

device = torch.device("cuda")
# device = torch.device("cpu")

checkpoint = torch.load(model_path, map_location=device)
if type(checkpoint) is dict:
    # args.start_epoch = checkpoint['epoch']
    best_result = checkpoint['best_result']
    model = checkpoint['model']
    print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
else:
    model = checkpoint
    # args.start_epoch = 0

model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('https://192.168.2.109:8080/video')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

prev_ftime = 0
new_ftime = 0

while True:
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_resized = cv2.resize(frame_rgb,(feed_width,feed_height))
    frame_pytorch = transforms.ToTensor()(frame_rgb).unsqueeze(0)
    frame_cuda = frame_pytorch.to(device)

    with torch.no_grad():
        prediction = model(frame_cuda)
    
    output = prediction.squeeze().cpu().numpy()
    output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    new_ftime = time.time()
    fps = 1/(new_ftime - prev_ftime)
    prev_ftime = new_ftime
    fps = str(int(fps))

    cv2.putText(frame, fps, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(output, fps, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('webcam',frame)
    # cv2.imshow('webcam',cv2.resize(frame,(860,640)))
    cv2.imshow('depth',output)
    # cv2.imshow('depth',cv2.resize(disp_resized_np,(860,640)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
