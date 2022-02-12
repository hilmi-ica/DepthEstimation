# import os
# import glob
import torch
# import utils
import cv2
# import argparse
import numpy as np
import time

from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

model_path = "weights/dpt_large-midas-2f21e586.pt"
model_type = "dpt_large"

# model_path = "weights/dpt_hybrid-midas-501f0c75.pt"
# model_type = "dpt_hybrid"

# model_path = "weights/midas_v21_small-70d6b9c8.pt"
# model_type = "midas_v21_small"

# model_path = "weights/midas_v21-f6b98070.pt"
# model_type = "midas_v21"

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load network
if model_type == "dpt_large": # DPT-Large
    model = DPTDepthModel(
        path=model_path,
        backbone="vitl16_384",
        non_negative=True,
    )
    net_w, net_h = 384, 384
    resize_mode = "minimal"
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
elif model_type == "dpt_hybrid": #DPT-Hybrid
    model = DPTDepthModel(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
    )
    net_w, net_h = 384, 384
    resize_mode="minimal"
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
elif model_type == "midas_v21":
    model = MidasNet(model_path, non_negative=True)
    net_w, net_h = 384, 384
    resize_mode="upper_bound"
    normalization = NormalizeImage(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
elif model_type == "midas_v21_small":
    model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
    net_w, net_h = 256, 256
    resize_mode="upper_bound"
    normalization = NormalizeImage(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
else:
    print(f"model_type '{model_type}' not implemented, use: --model_type large")
    assert False

transform = Compose(
    [
        Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method=resize_mode,
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        normalization,
        PrepareForNet(),
    ]
)

model.eval()

optimization = True
# optimization = False
if optimization==True:
        # rand_example = torch.rand(1, 3, net_h, net_w)
        # model(rand_example)
        # traced_script_module = torch.jit.trace(model, rand_example)
        # model = traced_script_module
    
        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)  
            model = model.half()

model.to(device)

cap = cv2.VideoCapture(0)
width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# depth_min = prediction.min()
# depth_max = prediction.max()
bits = 2
max_val = (2**(8*bits))-1

if width_orig > height_orig:
    scale = width_orig / 384
else:
    scale = height_orig / 384

height = (np.ceil(height_orig / scale / 32) * 32).astype(int)
width = (np.ceil(width_orig / scale / 32) * 32).astype(int)

prev_ftime = 0
new_ftime = 0

while True:
    ret, frame_orig = cap.read()

    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB) / 255.0

    # input
    frame_input = transform({"image": frame})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(frame_input).to(device).unsqueeze(0)
        if optimization==True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)  
            sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        depth = max_val * (prediction - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros(prediction.shape, dtype=prediction.type)

    # out = out.astype("uint8")
    depth = depth.astype("uint16")
    depth_normalized = cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    new_ftime = time.time()
    fps = 1/(new_ftime - prev_ftime)
    prev_ftime = new_ftime
    fps = str(int(fps))

    cv2.putText(frame_orig, fps, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
    # cv2.putText(depth, fps, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("webcam",frame_orig)
    cv2.imshow("depth",depth)
    # cv2.imshow("predict",depth)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# print(depth)
# print(depth.shape)
# print(depth[int(height_orig/2),int(width_orig/2)])

# print(depth_normalized)
# print(depth_normalized.shape)
# print(depth_normalized[int(height_orig/2),int(width_orig/2)])

# print(prediction)
# print(prediction.shape)
# print(type(prediction))

cap.release()