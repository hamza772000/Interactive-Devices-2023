#!/usr/local/opt/python@3.8/bin/python3.8
import freenect
import serial.tools.list_ports
import math
import numpy as np
import time
import cv2
import ctypes
import threading
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# model = load_model('/users/hamza/Downloads/ID/medium_densenet.h5')
model = load_model('/users/hamza/Downloads/ID/medium_densenet_3.h5')

dict = {0:'hug', 1:'None', 2:'Poke',3:'Punch',4:'Slap'}
# model = load_model('/users/hamza/Downloads/medium_densenet_1.h5')
def process_frame(x):
    arr = []
    x = x.copy()
    print(len(x))
    for i in range (len(x)):
        # print("Hello",i)
        frame = x[i]
        frame = cv2.resize(frame, (224,224))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame =  np.expand_dims(frame, axis=0)
        frame = preprocess_input(frame)
        predictions =  model.predict(frame)
        prediction_class = np.argmax(predictions,axis =  1)
        arr.append((max(predictions[0]), prediction_class[0]))
    # print(arr)
    if len(x) > 1:
        arr = sorted(arr, key = lambda x:x[0])
        print(dict[arr[1][1]])
    else:
        print(dict[arr[0][1]])
    # print(dict[arr[1][1]])
    return 0

#     frame = np.repeat(frame, 3, axis=-1)
#     frame = frame.reshape(1,224,224,3)
#     print(frame.shape)
#     # frame = tf.keras.applications.resnet.preprocess_input(frame)


#     model.predict(frame)
# vid = cv2.VideoCapture(1)
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
x = 10
lightsOn = False
libfreenect_sync = ctypes.CDLL('/usr/local/lib/libfreenect_sync.dylib')
libfreenect_sync.freenect_sync_set_tilt_degs.argtypes = [ctypes.c_int, ctypes.c_int]
libfreenect_sync.freenect_sync_set_tilt_degs.restype = ctypes.c_int
libfreenect_sync.freenect_sync_set_led.argtypes = [ctypes.c_int, ctypes.c_int]
libfreenect_sync.freenect_sync_set_led.restype = ctypes.c_int
all_frames = []

def get_video():
    array,_ = freenect.sync_get_video(0,freenect.VIDEO_IR_10BIT)
    return array

def get_video_rgb():
    array,_ = freenect.sync_get_video(1,freenect.VIDEO_RGB)
    # array,_ = freenect.sync_get_video(1,freenect.VIDEO_IR_10BIT)

    return array

def pretty_depth(depth):
    np.clip(depth, 0, 2**10-1, depth)
    depth >>=2
    depth=depth.astype(np.uint8)
    return depth
init_frame = 0
frames = []

def get_original_frame():
    for i in range(10):
        frames.append(pretty_depth(get_video()))
    # time.sleep(1)
    init_frame = np.mean(frames, axis=0).astype(dtype=np.uint8)
    init_frame = cv2.GaussianBlur(init_frame, (5, 5), 0)
    frames.clear()

    return init_frame

ports = serial.tools.list_ports.comports()
arduino_port = None
for port in ports:
    if 'usbmodem' in port.device:
        arduino_port = port
if arduino_port is not None:
    arduino_port =  arduino_port.device
init_frame = get_original_frame()
prev_diff = 100
model_frames = []
hit_time= time.time()
# optical_flow = cv2.DualTVL1OpticalFlow_create()
motion_threshold = 20
prev_frame = None
stopped = False
count = 0
hit_cout = 0
frame_count = 200
ishit = False
start = time.time()
counts = 0
curr_diff = 0
framee = 0
previous_diff = 0
diff_count = 0
tmp = 0
cumulative_diff = 0
got_diff = False
maintain = False
cumulative_thresh  = 100
p_count = 0
capture_count = 0
prev = 9
# params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
prev = 0
hit_time = time.time()
avg_magnitude = 0
prev_frame = np.zeros_like(init_frame)
if __name__ == "__main__":
    while 1:

        
        
        # print(cumulative_diff)
        # rgb = get_video_rgb()
        # rgb_colour = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
        # rgb_colour = rgb.copy()
        # ret, rgb = vid.read()
   
        # rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        # contours, hierarchy = cv2.findContours(rgb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(rgb, contours, -1, (0, 255, 0), 3)
        # all_frames.append(rgb)
        # framee += 1
        # cv2.imwrite('/users/hamza/Downloads/slap_frames_medium/' + str(framee) + '.png',rgb)

        # ret,rgb = vid.read()
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('RGB',rgb)
        # cv2.imshow("init", init_frame)
        #get a frame from RGB camera
        frame = get_video()
        #display IR image
        frame = pretty_depth(frame)
        
        cv2.imshow('Frame', frame)
        # cv2.imshow('IR', frame)
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        # if prev_frame is not None:
            # flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, **params)
            # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # avg_magnitude = np.mean(magnitude)
            # prev_frame = frame.copy()

            # print(avg_magnitude)


        lower_skin = 0
        upper_skin = 35
        diff = cv2.absdiff(blur, init_frame)
        # cv2.imshow('blur',diff)
        
        cumulative_area = 0

        hand_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # rgb = cv2.subtract(rgb,30)
        # cv2.imshow('diff',frame)
        # cv2.imshow('rgb',rgb)
        mean_diff = np.mean(diff)
        # print(cumulative_diff,mean_diff,mean_diff - previous_diff)

        print(mean_diff)
        
        # cv2.imshow('frame', frame)


        # print(mean_diff - previous_diff)
        # print(mean_diff)
        # if (mean_diff - previous_diff) > 0.3:
        #     print("INCREASING")
        #     tmp += 1
        # print(cumulative_diff)
  
        # print(mean_diff)
        # else:
            # print("DECREASING")
        # cv2.imshow('rgb',frame)

        # print(arduino_port.device)
        if mean_diff> 3.5:
            print(mean_diff-previous_diff)
            # if mean_diff - previous_diff > 2:

                # print(mean_diff,mean_diff - previous_diff)
            # print(mean_diff - previous_diff)
            # print((mean_diff - previous_diff))
            ser = serial.Serial(arduino_port, 9600)
            
            sent = True
            count = 0
            lightsOn = True
            # print(mean_diff - previous_diff)
            # print("MOTION")
            # ishit = True
            # if(mean_diff < 20 and (mean_diff - previous_diff) < 3):
            

            if(mean_diff - previous_diff) >= 0.5 and (time.time() - hit_time >= 0.3 or not stopped):
                print(mean_diff - previous_diff)

                

                
                # print("INCREASING")
                if cumulative_diff > cumulative_thresh:
                    ser.write(str.encode('t'))
                else:
                    ser.write(str.encode('h'))
                hit_path = '/users/hamza/Downloads/tets/' + str(hit_cout)
                if not os.path.isdir(hit_path):
                    os.mkdir(hit_path)
                
                if capture_count <= 3:
                    model_frames.append(frame)

                    # if len(model_frames) ==  1:
                    #     model_frames[0].append(rgb)
                    # else:
                    #     model_frames.append([rgb])
                    cv2.imwrite(hit_path + '/' + str(len(os.listdir(hit_path))) + '.png',frame)
                    capture_count+=1
                if got_diff:
                    got_diff = False
                prev = mean_diff - previous_diff
                hit_time =  time.time()
                stopped = False

            else:
                stopped = True
                # print("DECREASING")
                capture_count = 0
                ser.write(str.encode('h'))
                # print("MAINTIAN")
                if not got_diff:
                    prev = prev * 0.7
                    t = threading.Thread(target=process_frame, args=(model_frames.copy(),))
                    t.start()
                    # process_frame(frame=frame)
                    # process_frame(model_frames[0][0])
                    model_frames.clear()

                    hit_cout += 1
                    cumulative_diff += mean_diff
                    got_diff = True
           
            
            

                # break



        else:
            
            # if count == 5:
            #     prev = 0
            ser = serial.Serial(arduino_port, 9600)
            ser.flush()
            # if maintain:
                # ser.write(str.encode('p'))
                # p_count += 1
                # maintain = True
            if lightsOn and count >= 30:
                print("NEW FRAMEEEE")
                init_frame = get_original_frame()
                lightsOn = False
            
                


            # print("NOTHING")
        
        # prev_frame = frame
        
        if abs(prev_diff - mean_diff) <= 0.15 and mean_diff > 1 and mean_diff < 3:
            count += 1
            # print(count)
        else:
            count = 0

        prev_diff = mean_diff
        if diff_count % 1 == 0:
            previous_diff  = mean_diff
        diff_count += 1

        


        # quit program when 'esc' key is pressed
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        elif k == 97:
            x += 10
            libfreenect_sync.freenect_sync_set_tilt_degs(x, 0)

        elif chr(k) == 'x':
            x -= 10
            libfreenect_sync.freenect_sync_set_tilt_degs(x, 0)

            
        

    cv2.destroyAllWindows()