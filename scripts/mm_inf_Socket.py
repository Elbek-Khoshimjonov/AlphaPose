import argparse
import cv2
from alphapose.api import Model
import os
import socket
import struct
import numpy as np
import datetime
import time
from pynput import keyboard

def TCPIP_Client(labels=0, bboxes=[0,0,0,0,0], color=16, PACKET_TYPE=0):
    # header
    magicCode = 1234
    pType = 12
    pVer = 1
    # data = client_socket.recv(1024) # 서버가 보낸 메시지를 받기위해서 대기함.
    # IF-SC400 (Detection)
    IF_SC400_PACKET_TYPE = 4
    IF_SC400_IP = bytes(socket.gethostbyname(socket.gethostname() + '.local'), encoding='UTF-8')
    # IF_SC400_IP = bytes("166.104.14.77", encoding='UTF-8')
    OBJECT_TYPE = labels # 0=사람, 1=차량, 2=트럭
    IF_SC400_TIME = bytes(str(datetime.datetime.now()), encoding='UTF-8') # 시간 문자열은 예제와 같은 (yyyy-MM-dd HH:mm:ss.ffffff) 포맷으로 해주시면 됩니다.
    # 추후에는 이미지의 instance가 여러개니까 bbox와 정확도를 list로 보낼 수 있어야 함.
    # OBJ_X1 : float = bboxes[0]
    # OBJ_Y1 : float = bboxes[1]
    # OBJ_WIDTH : float = bboxes[2]
    # OBJ_HEIGHT : float = bboxes[3]
    # PERSENT : float = bboxes[-1]
    OBJ_BOXES = bboxes[:-1]
    # PERSENT : float = bboxes[-1]
    ACTION = 2 # 0=정상상태, 1=쓰러짐, 2=월담, 3=싸움, 4=밀수
    COLOR = color
    # print(f'OBJ_BOXES is {OBJ_BOXES}, And COLOR is {COLOR}')
    # IF_SC400_Struct = (IF_SC400_PACKET_TYPE, IF_SC400_IP, OBJECT_TYPE, IF_SC400_TIME, OBJ_BOXES, ACTION, COLOR)
    # IF_SC400_Format = '<B 15s B 30s f B B'
    IF_SC400_Struct = (IF_SC400_PACKET_TYPE, IF_SC400_IP)
    IF_SC400_Format = '<B 15s'
    IF_SC400_Packer = struct.Struct(IF_SC400_Format)
    IF_SC400_Packet = IF_SC400_Packer.pack(*IF_SC400_Struct)


    if PACKET_TYPE == 4:
        client_socket.send(IF_SC400_Packet)
        # print('Received from the server :',repr(data.decode()))

def handleRelease(key):
    try:
        k = key.char
    except:
        k = key.name

    if k in 's':
        client_socket.close() # socket disconnect

    # if key == key.esc:
    #     return False # stop keyboard listener

def color_conclusion(BGR):
    # change from BGR to RGB
    rgb = list(BGR)
    rgb = np.array(rgb[::-1])
    
    # color Standard
    black = [0,0,0] #
    white = [255,255,255]#
    gray = [128,128,128]#
    purple = [128,0,128]#
    red = [255,0,0]#
    yr = [255,128,0] #yellow red = orange
    yellow = [255,255,0] #
    yg = [128,255,0] #yello green
    green = [0,255,0] #
    bg = [13,152,186] # blue green
    blue = [0,0,255] #
    pb = [0,0,128] # purple blue = navy
    pink = [255,0,255] #
    brown = [160,42,42] #
    rp = [102,0,153] #red purple

    # color_name = ['red', 'yellow red', 'yellow', 'green yellow', 'green', 'blue green', 'blue', 'purple blue', 'purple', 'red purple', 'pink', 'brown', 'white', 'gray', 'black']
    color_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    color = np.array([red, yr, yellow, yg, green, bg, blue, pb, purple, rp, pink, brown, white, gray, black])
    distant = np.sum((color-rgb)**2,axis=1)**(1/2)
    return color_name[np.argmin(distant)]

def main():
    parser = argparse.ArgumentParser(description='AlphaPose Demo')
    parser.add_argument('--pose_cfg', type=str, default='configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                        help='experiment configure file name')
    parser.add_argument('--pose_checkpoint', type=str, default='pretrained_models/fast_res50_256x192.pth',
                        help='checkpoint file name')
    parser.add_argument('--det_cfg', type=str, default='mmDetection/r50_b8_p5_gn_v2.py',
                        help='experiment configure file name')
    parser.add_argument('--det_checkpoint', type=str, default='mmDetection/epoch_8.pth',
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

    # socket connection
    HOST = '166.104.14.109' # Server IP
    # HOST = 'devcoretec.iptime.org' # 철기연 Server IP
    # HOST = '222.99.97.171' # 철기연 Server IP
    PORT = 4211

    print("start commu")
    global client_socket
    client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
    client_socket.connect((HOST, PORT))
    print("succ")

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
    # src_type="stream"


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
        # cap = cv2.VideoCapture("rtsp://166.104.14.77:8550/live1")

        # Test for validness
        if not cap.isOpened():
            print("Problem with %s"%args.stream)
            exit(0)
        
        lis = keyboard.Listener(on_release=handleRelease) 
        lis.start() # start keyboard listener
        print("start")
        while True:
            ret, img = cap.read()
            
            if ret is None:
                break
                
            poses, labels, dets, other_objects = model.process(img)
            if bool(poses):
                print("color send")
                BGR = poses[0]["clothe_color"]
                color = color_conclusion(BGR)
                print(f'color is {color}')
                TCPIP_Client(labels = labels, bboxes= dets, color= color, PACKET_TYPE=4)
                time.sleep(0.1)

            TCPIP_Client(labels = labels, bboxes= dets, PACKET_TYPE=4)
            time.sleep(0.1)

if __name__=='__main__':
    main()