import argparse
import cv2

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import natsort


from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.presets import SimpleTransform
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.vis import vis_frame_fast


from mmdet.apis import init_detector, inference_detector

config_file = 'mmDetection/gfl_x101_611.py'
checkpoint = 'mmDetection/weights.pth'


parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, default='configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, default='pretrained_models/fast_res50_256x192.pth',
                    help='checkpoint file name')
parser.add_argument('--image', type=str, required=True,
                    help='input image')
parser.add_argument('--showbox', default=True, action='store_true',
                    help='visualize human bbox')
"""----------------------------- Clothe Color options -----------------------------"""
parser.add_argument('--clothe_color', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--output', type=str, required=True,
                    help='output image')
parser.add_argument('--det_thresh', type=float, default=0.5,
                    help='threshold value for detection')

args = parser.parse_args()
cfg = update_config(args.cfg)

# Device configuration
args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.tracking = False
args.pose_track = False


# Preprocess transformation
pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
transformation = SimpleTransform(
            pose_dataset, scale_factor=0,
            input_size=cfg.DATA_PRESET.IMAGE_SIZE,
            output_size=cfg.DATA_PRESET.HEATMAP_SIZE,
            rot=0, sigma=cfg.DATA_PRESET.SIGMA,
            train=False, add_dpg=False, gpu_device=args.device)

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
norm_type = cfg.LOSS.get('NORM_TYPE', None)

heatmap_to_coord = get_func_heatmap_to_coord(cfg)


# Load Detector Model
model = init_detector(config_file, checkpoint=checkpoint, device=args.device)


# Load pose model
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

print(f'Loading pose model from {args.checkpoint}...')
pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

pose_model.to(args.device)
pose_model.eval()

# Load Image
img = cv2.imread(args.image)

# Detector
det = inference_detector(model, img)[0]

bboxes = []
cropped_boxes = []
inps = []

# Preprocess
for bbox in det:
    acc = bbox[4]
    
    if acc>=args.det_thresh:

        bbox = bbox[:4].astype(int)
        x1, y1, x2, y2 = bbox    
        inp, cropped_box = transformation.test_transform(img[y1:y2, x1:x2], torch.Tensor([0, 0, x2-x1, y2-y1]))

        inps.append(inp.unsqueeze(0))
        bboxes.append(bbox)
        cropped_boxes.append(cropped_box)

# Run pose model
inps = torch.cat(inps).to(args.device)
hm_datas = pose_model(inps).cpu()
del inps

# Convert heatmap to coord and score
pose_coords = []
pose_scores = []

for (hm_data, cropped_box, bbox) in zip(hm_datas, cropped_boxes, bboxes):
    pose_coord, pose_score = heatmap_to_coord(hm_data[EVAL_JOINTS], cropped_box, hm_shape=cfg.DATA_PRESET.HEATMAP_SIZE, norm_type=norm_type)

    pose_coords.append(torch.from_numpy(pose_coord + bbox[:2]))
    pose_scores.append(torch.from_numpy(pose_score))


result = []

# Draw bboxs and pose coordinates
for bbox, pose_coord, pose_score in zip(bboxes, pose_coords, pose_scores):

    # # Bbox
    # left_top = (bbox[0], bbox[1])
    # right_bottom = (bbox[2], bbox[3])
    # img = cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 3)

    # Pose coords
    result.append({"keypoints": pose_coord, "kp_score": pose_score, "box": bbox})

    # for point in pose_coord:
    #     img = cv2.circle(img, tuple(point.astype(int)), 3, (255, 0, 0), -1)

img = vis_frame_fast(img, {"result": result}, args)


cv2.imwrite(args.output, img)





