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
from threading import Thread
import random

class Stack():
    def __init__(self):
        self.data = []
    
    def push(self, element):
        self.data.append(element)
    
    def pop(self):
        if not len(self.data) == 0:
            img = self.data[-1]
            if len(self.data) >= 500:
                self.data.clear()
            return img

class VideoScreenshot(object):
    def __init__(self, src, stack):
        self.capture = cv2.VideoCapture(src)
        self.screenshot_interval = 1
        self.s = stack
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
                self.s.push(self.frame)

    def show_frame(self):
        if self.status:
            cv2.imshow('frame', self.frame)
            k = cv2.waitKey(1)
        if k == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
        # def show_frame_thread():
        #     if self.status:
        #         cv2.imshow('frame', self.frame)
        #         k = cv2.waitKey(1)
        #         if k == ord('q'):
        #             self.capture.release()
        #             cv2.destroyAllWindows()
        #             exit(1)
        # Thread(target=show_frame_thread, args=()).start

def TCPIP_Client(labels=0, bboxes=[0,0,0,0,0], color=50):
    data_IP = socket.gethostbyname(socket.gethostname() + '.local')
    data_time = str(datetime.datetime.now())  # 시간 문자열은 예제와 같은 (yyyy-MM-dd HH:mm:ss.ffffff) 포맷으로 해주시면 됩니다.
    data_objNum = len(labels) # 리스트 최대 길이는 40입니다. random.randrange(1, 41)

    print(data_IP + ' ' + data_time + ' ' + str(data_objNum))

    IF_SC400_header = struct.pack('b', 4) # IF-SC400
    while len(data_IP) < 15 :
        data_IP += ' '
    IF_SC400_IP = bytes(data_IP, encoding='UTF-16')
    while len(data_time) < 28 :
        data_time += ' '
    IF_SC400_time = bytes(data_time, encoding='UTF-16')
    IF_SC400_objNum = struct.pack('b', data_objNum) # 0 - 40

    sendBuffer = IF_SC400_header + IF_SC400_IP + IF_SC400_time + IF_SC400_objNum

    for i in range(data_objNum) :
        import pdb; pdb.set_trace()
        data_objType = labels[i] # 0=사람, 1=차량, 2=트럭
        ObjX1 = int(bboxes[i][0])
        ObjY1 = int(bboxes[i][1])
        ObjX2 = int(bboxes[i][2])
        ObjY2 = int(bboxes[i][3])
        data_percent = round(bboxes[i][4],2)
        data_action = random.randrange(0, 5) # 0=정상상태, 1=쓰러짐, 2=월담, 3=싸움, 4=밀수
        data_color = color # 0=빨강, 1=주황, 2=노랑, 3=연두, 4=초록, 5=청록, 6=파랑, 7=남색, 8=보라, 9=자주, 10=분홍, 11=갈색, 12=하양, 13=회색, 14=검정

        print(str(data_objType) + ' ' + str(ObjX1) + ' ' + str(ObjY1) + ' ' + str(ObjX2) + ' ' + str(ObjY2) + ' ' + str(data_percent) + ' ' + str(data_action) + ' ' + str(data_color))

        IF_SC400_objType = struct.pack('b', data_objType)
        IF_SC400_objX1 = struct.pack('i', ObjX1)
        IF_SC400_objY1 = struct.pack('i', ObjY1)
        IF_SC400_objX2 = struct.pack('i', ObjX2)
        IF_SC400_objY2 = struct.pack('i', ObjY2)
        IF_SC400_percent = struct.pack('f', data_percent)
        IF_SC400_action = struct.pack('b', data_action)
        IF_SC400_color = struct.pack('b', data_color)

        sendBuffer = sendBuffer + IF_SC400_objType + IF_SC400_objX1 + IF_SC400_objY1 + IF_SC400_objX2 + IF_SC400_objY2 + IF_SC400_percent + IF_SC400_action + IF_SC400_color
    
    client_socket.send(sendBuffer)

# def handleRelease(key):
#     try:
#         k = key.char
#     except:
#         k = key.name

#     if k in 's':
#         client_socket.close() # socket disconnect

def color_conclusion(BGR):
    # change from BGR to RGB
    rgb = list(BGR)
    rgb = np.array(rgb[::-1])
    
    # color Standard
    black = [0,0,0]
    white = [255,255,255]
    gray = [128,128,128]
    Light_purple = [197,166,242]
    purple = [102,0,153]
    Dark_purple = [51,0,51]
    Light_red = [255,90,90]
    red = [255,0,0]
    Dark_red = [139,0,0]
    Light_yr = [255,183,123]
    yr = [255,128,0] # yellow red = orange
    Dark_yr = [211,91,27]
    Light_yellow = [252,248,156]
    yellow = [255,255,0]
    Dark_yellow = [211,201,3]
    yg = [128,255,0] # yello green
    Light_green = [142,255,114]
    green = [0,255,0]
    Dark_green = [0,50,0]
    bg = [13,152,186] # blue green
    Light_blue = [165,206,252]
    blue = [0,0,255]
    Dark_blue = [0,0,170]
    pb = [0,0,128] # purple blue = navy
    Dark_pb = [0,0,85]
    Light_pink = [255,208,232]
    pink = [255,0,255] 
    Dark_pink = [236,0,118]
    Light_brown = [190,124,71]
    brown = [160,42,42] 
    Dark_brown = [71,29,3]
    rp =  [128,0,128]# red purple

    # color_name = ['Light_red', 'red', 'Dark_red', 'Light_yr', 'yellow red', 'Dark_yr', 'Light_yellow', 'yellow', 'Dark_yellow', 'green yellow', 'Light_green', 'green', 'Dark_green','blue green', 'Light_blue', 'blue', 'Dark_blue', 'purple blue', 'Dark_pb', 'Light_purple', 'purple', 'Dark_purple', 'red purple', 'Light_pink', 'pink', 'Dark_pink', 'Light_brown', 'brown', 'Dark_brown','white', 'gray', 'black']
    color_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    color = np.array([Light_red, red, Dark_red, Light_yr, yr, Dark_yr, Light_yellow, yellow, Dark_yellow, yg, Light_green, green, Dark_green, bg, Light_blue, blue, Dark_blue, pb, Dark_pb, Light_purple, purple, Dark_purple, rp, Light_pink, pink, Dark_pink, Light_brown, brown, Dark_brown, white, gray, black])
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
    # HOST = '192.168.0.21' # 철기연 Server IP
    PORT = 4211

    print("start commu")
    global client_socket
    client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
    client_socket.connect((HOST, PORT))
    print("succ")

    # Check for image or video input

    # if int(args.image is not None) + int(args.video is not None) + int(args.stream is not None)!=1:
    #     print("Please select --stream")
    #     exit(0)

    s1 = Stack()
    #region Process video
    cap = VideoScreenshot(args.stream, s1)

    # Load model
    model = Model(args)

   
    
    # lis = keyboard.Listener(on_release=handleRelease) 
    # lis.start() # start keyboard listener
    print("start")
    while True:
        try:
            st = time.time()
            # cap.show_frame()
            img = s1.pop()
            # import pdb; pdb.set_trace()
            #if bool(img) == True:
            poses, labels, dets, other_objects = model.process(img)

            # show stream inference
            img = model.draw_results(img, poses, other_objects)
            cv2.imshow("Stream", img)
            cv2.waitKey(1)
            dt = time.time() - st
            print(dt)
            time.sleep(1)
            if bool(poses):                                               
                print("color send")
                BGR = poses[0]["clothe_color"]
                color = color_conclusion(BGR)
                print(f'color is {str(color)}')
                TCPIP_Client(labels=labels, bboxes=dets, color=color)
                
            else:
                TCPIP_Client(labels=labels, bboxes=dets)
        except AttributeError:
            pass



if __name__=='__main__':
    main()