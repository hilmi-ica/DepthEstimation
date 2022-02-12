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

# mobilenet = im.MobileNet()
# model = im.MobileNet(relu6=True)
checkpoint = torch.load(model_path, map_location=device)
if type(checkpoint) is dict:
    # args.start_epoch = checkpoint['epoch']
    best_result = checkpoint['best_result']
    model = checkpoint['model']
    print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
else:
    model = checkpoint
    # args.start_epoch = 0

# model = torch.load(model_path,map_location=device)
# model = model['model']
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

# print(checkpoint)

# feed_height = model['height']
# feed_width = model['width']

img_name = "20220120-110735.png"
img_path = os.path.join("../data",img_name)

img = cv2.imread(img_path)

original_height, original_width, channels = img.shape

img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img_rgb = img_rgb.cuda()
# img_resized = cv2.resize(img_rgb,(32,32))
img_pytorch = transforms.ToTensor()(img_rgb).unsqueeze(0)
img_cuda = img_pytorch.to(device)
# img_cuda = torch.from_numpy(img_pytorch).float().cuda()

print(img_cuda.shape)

with torch.no_grad():
    prediction = model(img_cuda)

print(prediction.shape)
# prediction_resized = torch.nn.functional.interpolate(prediction,
#     (original_height, original_width), mode="bilinear", align_corners=False)

# print(prediction)

output = prediction.squeeze().cpu().numpy()
# output = utils.colored_depthmap(output,0.1,100)
# output = output[:,:,0]

output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

print(output)
print(output.shape)
print(type(output))

cv2.imshow("img",img)
cv2.imshow("depth",output)

cv2.waitKey(0)
cv2.destroyAllWindows
