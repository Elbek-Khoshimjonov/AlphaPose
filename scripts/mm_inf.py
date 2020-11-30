import argparse
import cv2
from alphapose.api import Model


parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--pose_cfg', type=str, default='configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                    help='experiment configure file name')
parser.add_argument('--pose_checkpoint', type=str, default='pretrained_models/fast_res50_256x192.pth',
                    help='checkpoint file name')
parser.add_argument('--det_cfg', type=str, default='mmDetection/gfl_x101_611.py',
                    help='experiment configure file name')
parser.add_argument('--det_checkpoint', type=str, default='mmDetection/weights.pth',
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
parser.add_argument('--clothe_color', default=True, action='store_true',
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


if (src_type=="image" or src_type=="video") and args.output is None:
    print("--output is required")
    exit(0)

# Load model
model = Model(args)



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



if src_type=="image":    
    #region Process image
    # Load Image
    img = cv2.imread(args.image)

    # Process single image
    poses, other_objects = model.process(img)

    # Draw Results
    img = model.draw_results(img, poses, other_objects)


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

        poses, other_objects = model.process(img)
        img = model.draw_results(img, poses, other_objects)

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
            
        poses, other_objects = model.process(img)
        img = model.draw_results(img, poses, other_objects)

        cv2.imshow("Stream", img)
        cv2.waitKey(1)
