from argparse import Namespace
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

# Constants
BBOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 0, 0)
FONT_SCALE = 0.5

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


#region Dominant color
from sklearn.cluster import KMeans
from scipy.ndimage.morphology import binary_fill_holes as imfill


def area_mask(img, polygon, neg_polygon=[]):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Positive polygon
    if type(polygon) is list:
        for pol in polygon:
            mask = cv2.fillConvexPoly(mask, pol, 1)
    else:
        mask = cv2.fillConvexPoly(mask, polygon, 1)

    if type(neg_polygon) is list:
        for pol in neg_polygon:
            mask = cv2.fillConvexPoly(mask, pol, 0)

    else:
        mask = cv2.fillConvexPoly(mask, neg_polygon, 0)

    
    mask = mask.astype(bool)
    
    out = img[imfill(mask)]
    return out

def rotate(p, theta):                  
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])                                           
    return np.matmul(p, R)


def find_forearm(elbow, wrist, ratio=0.3):
    lst = [ elbow + rotate( (wrist-elbow)*ratio, np.radians(90)),
            wrist + rotate( (elbow-wrist)*ratio, np.radians(-90)),
            wrist + rotate( (elbow-wrist)*ratio, np.radians(90)),
            elbow + rotate( (wrist-elbow)*ratio, np.radians(-90))
            ]
    return np.asarray([ (int(p[0]), int(p[1])) for p in lst ])

def find_clothe_color(img, kp_preds):
    # Shoulders
    l_sh_x, l_sh_y = int(kp_preds[5, 0]), int(kp_preds[5, 1])
    r_sh_x, r_sh_y = int(kp_preds[6, 0]), int(kp_preds[6, 1])
    # Hips
    l_h_x, l_h_y = int(kp_preds[11, 0]), int(kp_preds[11, 1])
    r_h_x, r_h_y = int(kp_preds[12, 0]), int(kp_preds[12, 1])

    # Body region
    region = np.asarray([(l_sh_x, l_sh_y), (r_sh_x, r_sh_y), (l_h_x, l_h_y), (r_h_x, r_h_y)])

    ## Wrist region
    # Left
    l_elbow_x, l_elbow_y = kp_preds[7, 0], kp_preds[7, 1]
    l_wrist_x, l_wrist_y = kp_preds[9, 0], kp_preds[9, 1]

    l_forearm = find_forearm(np.array([l_elbow_x, l_elbow_y]), np.array([l_wrist_x, l_wrist_y]))

    # Right
    r_elbow_x, r_elbow_y = kp_preds[8, 0], kp_preds[8, 1]
    r_wrist_x, r_wrist_y = kp_preds[10, 0], kp_preds[10, 1]

    r_forearm = find_forearm(np.array([r_elbow_x, r_elbow_y]), np.array([r_wrist_x, r_wrist_y]))


    mask = area_mask(img, region, [l_forearm, r_forearm])
    clt = KMeans(n_clusters=1)  # cluster number
    clt.fit(mask)
    color = clt.cluster_centers_[0].astype(int)
    color = (color[0].item(), color[1].item(), color[2].item())

    return color
#endregion


class Model:

    # Constructor
    def __init__(self, args=None):

        if args is None:

            args = Namespace(
                # Pose config
                pose_cfg='configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                # Pose checkpoint
                pose_checkpoint='pretrained_models/fast_res50_256x192.pth',
                # GPUS
                gpus='0',
                # Detection thresh
                det_thresh=0.5,
                # Detection config
                det_cfg='mmDetection/gfl_x101_611.py',
                # Detection checkpoint
                det_checkpoint='mmDetection/weights.pth',
                # Show clothe color
                clothe_color=True,
                # show bboxes
                showbox=True

            )
    
        
        self.pose_cfg = update_config(args.pose_cfg)
        

        # Device configuration
        args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
        args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
        args.tracking = False
        args.pose_track = False

        # Copy args
        self.args = args

        # Preprocess transformation
        pose_dataset = builder.retrieve_dataset(self.pose_cfg.DATASET.TRAIN)
        self.transformation = SimpleTransform(
            pose_dataset, scale_factor=0,
            input_size=self.pose_cfg.DATA_PRESET.IMAGE_SIZE,
            output_size=self.pose_cfg.DATA_PRESET.HEATMAP_SIZE,
            rot=0, sigma=self.pose_cfg.DATA_PRESET.SIGMA,
            train=False, add_dpg=False, gpu_device=args.device)

        self.norm_type = self.pose_cfg.LOSS.get('NORM_TYPE', None)

        # Post process        
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.pose_cfg)


        # Load Detector Model
        self.det_model = init_detector(args.det_cfg, checkpoint=args.det_checkpoint, device=args.device)

        # Load pose model
        self.pose_model = builder.build_sppe(self.pose_cfg.MODEL, preset_cfg=self.pose_cfg.DATA_PRESET)

        print(f'Loading pose model from {args.pose_checkpoint}...')
        self.pose_model.load_state_dict(torch.load(args.pose_checkpoint, map_location=args.device))

        self.pose_model.to(args.device)
        self.pose_model.eval()


    #region Process one sample
    def process(self,img):

        # Detector
        det_result = inference_detector(self.det_model, img)

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
            
            if acc>=self.args.det_thresh:

                bbox = bbox[:4].astype(int)

                # Person type & prepare for pose estimation
                if self.det_model.CLASSES[label]=='person':
                    x1, y1, x2, y2 = bbox    
                    inp, cropped_box = self.transformation.test_transform(img[y1:y2, x1:x2], torch.Tensor([0, 0, x2-x1, y2-y1]))

                    inps.append(inp.unsqueeze(0))
                    bboxes.append(bbox)
                    cropped_boxes.append(cropped_box)
                
                # Other objects, just take label and bbox
                else:
                    other_objects.append( (label, bbox) )

        poses = []

        if len(inps)>0:
            # Run pose model
            inps = torch.cat(inps).to(self.args.device)
            hm_datas = self.pose_model(inps).cpu()
            del inps

            # Convert heatmap to coord and score
            pose_coords = []
            pose_scores = []

            for (hm_data, cropped_box, bbox) in zip(hm_datas, cropped_boxes, bboxes):
                pose_coord, pose_score = self.heatmap_to_coord(hm_data[EVAL_JOINTS], cropped_box, hm_shape=self.pose_cfg.DATA_PRESET.HEATMAP_SIZE, norm_type=self.norm_type)

                pose_coords.append(torch.from_numpy(pose_coord + bbox[:2]))
                pose_scores.append(torch.from_numpy(pose_score))


            # Draw bboxs and pose coordinates
            for bbox, pose_coord, pose_score in zip(bboxes, pose_coords, pose_scores):

                # # Bbox
                # left_top = (bbox[0], bbox[1])
                # right_bottom = (bbox[2], bbox[3])
                # img = cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 3)

                # Pose coords
                poses.append({"keypoints": pose_coord, "kp_score": pose_score, "box": bbox, "clothe_color": find_clothe_color(img, pose_coord)})
                
            
        return poses, other_objects
        
    #endregion

    #region Draw results
    def draw_results(self, img, poses, other_objects):
        # Draw human results
        img = vis_frame_fast(img, {"result": poses}, self.args)


        # Draw other objects with name:
        for label, bbox in other_objects:
            
            label_text = self.det_model.CLASSES[label]
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