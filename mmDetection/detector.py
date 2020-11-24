from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np

config_file = 'mmDetection/gfl_x101_611.py'
checkpoint = 'mmDetection/weights.pth'
device = 'cuda:0'
det_thresh = 0.5

model = init_detector(config_file, checkpoint=checkpoint, device=device)

# inference the demo image
img = cv2.imread("image/truck.jpg")
result = inference_detector(model, img)

if isinstance(result, tuple):
    bbox_result, segm_result = result
else:
    bbox_result, segm_result = result, None
bboxes = np.vstack(bbox_result)
labels = [
    np.full(bbox.shape[0], i, dtype=np.int32)
    for i, bbox in enumerate(bbox_result)
]
labels = np.concatenate(labels)[:len(bboxes)]

for bbox, label in zip(bboxes, labels):
    acc = bbox[4]
    
    if acc>=det_thresh:
        print(label)
        bbox = bbox[:4].astype(int)
        x1, y1, x2, y2 = bbox    
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

cv2.imwrite("out.jpg", img)