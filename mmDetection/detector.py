from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np

config_file = 'mmDetection/gfl_x101_611.py'
checkpoint = 'mmDetection/weights.pth'
device = 'cuda:0'

model = init_detector(config_file, checkpoint=checkpoint, device=device)

# inference the demo image
img = cv2.imread("image/full.jpg")
result = inference_detector(model, img)

bbox_result = result[0]

for bbox in bbox_result:
    acc = bbox[4]
    bbox = bbox.astype(int)
    left_top = (bbox[0], bbox[1])
    right_bottom = (bbox[2], bbox[3])

    if acc>=0.5:
        img = cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 3)
        print(f'left_top: {left_top} right_bottom: {right_bottom}, acc: {acc}')