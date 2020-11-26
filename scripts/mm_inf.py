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

# Constants
BBOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 0, 0)
FONT_SCALE = 0.5



parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, default='configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, default='pretrained_models/fast_res50_256x192.pth',
                    help='checkpoint file name')
parser.add_argument('--image', type=str, required=False,
                    help='input image')
parser.add_argument('--video', type=str, required=False,
                    help='input video')
parser.add_argument("--stream", type=str, required=False,
                    help="video stream (CCTV)")
parser.add_argument('--showbox', default=True, action='store_true',
                    help='visualize human bbox')
"""----------------------------- Clothe Color options -----------------------------"""
parser.add_argument('--clothe_color', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--output', type=str, required=False,
                    help='output file')
parser.add_argument('--det_thresh', type=float, default=0.5,
                    help='threshold value for detection')

args = parser.parse_args()

# Check for image or video input

if int(args.image is not None) + int(args.video is not None) + int(args.stream is not None)!=1:
    print("Please select either[JUST ONE] --image or --video or --stream")
    exit(0)

if args.image is not None:
    src_type="image"
elif args.video is not None:
    src_type="video"
else:
    src_type="stream"


if src_type=="image" or src_type=="video" and args.output is None:
    print("--output is required")
    exit(0)


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


def recognize_video_ext(ext=''):
    if ext == 'mp4':
        return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
    elif ext == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else:
        print("Unknow video format {}, will use .mp4 instead of it".format(ext))
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


#region Process one sample
def process(img):

    # Detector
    det_result = inference_detector(model, img)

    if isinstance(det_result, tuple):
        bbox_result, segm_result = det_result
    else:
        bbox_result, segm_result = det_result, None

    det = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)[:len(det)]

    # For human objects
    bboxes = []
    cropped_boxes = []
    inps = []

    # Other objects
    other_objects = []

    # Preprocess
    for bbox, label in zip(det, labels):
        acc = bbox[4]
        
        if acc>=args.det_thresh:

            bbox = bbox[:4].astype(int)

            # Person type & prepare for pose estimation
            if model.CLASSES[label]=='person':
                x1, y1, x2, y2 = bbox    
                inp, cropped_box = transformation.test_transform(img[y1:y2, x1:x2], torch.Tensor([0, 0, x2-x1, y2-y1]))

                inps.append(inp.unsqueeze(0))
                bboxes.append(bbox)
                cropped_boxes.append(cropped_box)
            
            # Other objects, just take label and bbox
            else:
                other_objects.append( (label, bbox) )

    poses = []

    if len(inps)>0:
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


        # Draw bboxs and pose coordinates
        for bbox, pose_coord, pose_score in zip(bboxes, pose_coords, pose_scores):

            # # Bbox
            # left_top = (bbox[0], bbox[1])
            # right_bottom = (bbox[2], bbox[3])
            # img = cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 3)

            # Pose coords
            poses.append({"keypoints": pose_coord, "kp_score": pose_score, "box": bbox})
        
    return poses, other_objects
    
#endregion

#region Draw results
def draw_results(img, poses, other_objects):
    # Draw human results
    img = vis_frame_fast(img, {"result": poses}, args)


    if args.showbox:
        # Draw other objects with name:
        for label, bbox in other_objects:
            
            label_text = model.CLASSES[label]
            bbox = bbox.astype(int)

            # Bbox
            left_top = (bbox[0], bbox[1])
            right_bottom = (bbox[2], bbox[3])
            cv2.rectangle(img, left_top, right_bottom, BBOX_COLOR, 3)
            
            # Label name
            cv2.putText(img, label_text, (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, FONT_SCALE, TEXT_COLOR)

    return img
#endregion


if src_type=="image":    
    #region Process image
    # Load Image
    img = cv2.imread(args.image)

    # Process single image
    poses, other_objects = process(img)

    # Draw Results
    img = draw_results(img, poses, other_objects)


    cv2.imwrite(args.output, img)
    #endregion

elif src_type=="video":
    #region Process video
    cap = cv2.VideoCapture(args.video)

    # Test for validness
    if not cap.isOpened():
        print("Problem with %s"%args.video)
        exit(0)
    
    # Video info
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Identify fourcc
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Resolution: (%d, %d), fps: %d, fourcc: %d, frame_count: %d"%(w, h, fps, fourcc, frame_count))
    
    # Writer with same fps, fourcc and resolution
    writer = cv2.VideoWriter(*[args.output, fourcc, fps, (w, h)])
    if not writer.isOpened():
        print("Try to use other video encoders...")
        ext = args.output.split('.')[-1]
        fourcc, _ext = recognize_video_ext(ext)
        args.output = args.output[:-4] + _ext
        writer = cv2.VideoWriter(*[args.output, fourcc, fps, (w, h)])
    assert writer.isOpened(), 'Cannot open video for writing'
    
    # Frame by frame
    for i in tqdm(range(frame_count), "Processing %s"%args.video):
        
        ret, img = cap.read()
        if not ret:
            break

        poses, other_objects = process(img)
        img = draw_results(img, poses, other_objects)

        writer.write(img)

        
    
    # Close reader
    cap.release()

    # Close writer
    writer.release()

    #endregion    

elif src_type=="stream":

    #region Process video
    cap = cv2.VideoCapture(args.stream)

    # Test for validness
    if not cap.isOpened():
        print("Problem with %s"%args.stream)
        exit(0)
    
    while True:
        ret, img = cap.read()
        
        if ret is None:
            break
            
        poses, other_objects = process(img)
        img = draw_results(img, poses, other_objects)

        cv2.imshow("Stream", img)
        cv2.waitKey(1)
