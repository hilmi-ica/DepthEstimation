from __future__ import absolute_import, division, print_function
# %matplotlib inline

import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import cv2
import time

import torch
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist
from layers import disp_to_depth

model_name = "mono_640x192"
# model_name = "mono+stereo_640x192"
# model_name = "mono_no_pt_640x192"
# model_name = "mono+stereo_no_pt_640x1092"
# model_name = "mono_1024x320"
# model_name = "mono+stereo_1024x320"

download_model_if_doesnt_exist(model_name)
encoder_path = os.path.join("models", model_name, "encoder.pth")
depth_decoder_path = os.path.join("models", model_name, "depth.pth")

device = torch.device("cuda")
# device = torch.device("cpu")

# LOADING PRETRAINED MODEL
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location=device)
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)

loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(device)

encoder.eval()
depth_decoder.eval()
# print(loaded_dict_enc)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('https://192.168.2.109:8080/video')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']

prev_ftime = 0
new_ftime = 0

while True:
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,(feed_width,feed_height))
    frame_pytorch = transforms.ToTensor()(frame_resized).unsqueeze(0)
    frame_cuda = frame_pytorch.to(device)

    with torch.no_grad():
        features = encoder(frame_cuda)
        outputs = depth_decoder(features)
    
    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(disp,
        (height, width), mode="bilinear", align_corners=False)
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)

    new_ftime = time.time()
    fps = 1/(new_ftime - prev_ftime)
    prev_ftime = new_ftime
    fps = str(int(fps))

    cv2.putText(frame, fps, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(disp_resized_np, fps, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('webcam',frame)
    # cv2.imshow('webcam',cv2.resize(frame,(860,640)))
    cv2.imshow('depth',disp_resized_np)
    # cv2.imshow('depth',cv2.resize(disp_resized_np,(860,640)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()