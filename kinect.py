#!/usr/local/opt/python@3.8/bin/python3.8
import freenect
import math
import numpy as np
import time
import cv2
import ctypes



x = 10
lightsOn = False
libfreenect_sync = ctypes.CDLL('/usr/local/lib/libfreenect_sync.dylib')
libfreenect_sync.freenect_sync_set_tilt_degs.argtypes = [ctypes.c_int, ctypes.c_int]
libfreenect_sync.freenect_sync_set_tilt_degs.restype = ctypes.c_int
libfreenect_sync.freenect_sync_set_led.argtypes = [ctypes.c_int, ctypes.c_int]
libfreenect_sync.freenect_sync_set_led.restype = ctypes.c_int


def get_video():
    array,_ = freenect.sync_get_video(0,freenect.DEPTH_10BIT_PACKED)
    return array

def get_video_rgb():
    array,_ = freenect.sync_get_video(0,freenect.VIDEO_RGB)
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

init_frame = get_original_frame()
prev_diff = 100




count = 0
frame_count = 0



if __name__ == "__main__":
    while 1:
        cv2.imshow("init", init_frame)
        #get a frame from RGB camera
        frame = get_video()
        #display IR image
        frame = pretty_depth(frame)
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
       

        lower_skin = 190
        upper_skin = 255
        mask = cv2.inRange(frame, lower_skin, upper_skin)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=9)
        mask = cv2.dilate(mask, kernel, iterations=5)
        diff = cv2.absdiff(blur, init_frame)
        
            
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cumulative_area = 0
        top_contours = contours_sorted[:round(len(contours_sorted) * 0.3)]

        hand_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(hand_mask, [max_contour], 0, (255, 255, 255), -1)
            for contour in top_contours:

                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Draw the polygon around the contour
                cv2.polylines(frame, [approx], True, (0, 255, 0), 2)
        
            # cv2.drawContours(frame, [max_contour], 0, (255, 255, 255), -1)
        
        cv2.imshow('mask', hand_mask)

        # cv2.imshow('IR Video',frame)

        print(np.mean(diff))
        cv2.imshow('diff', diff)
        mean_diff = np.mean(diff)
        cv2.imshow('frame', frame)


        # for contour in contours:
            # rect = cv2.boundingRect(contour)
            # cv2.rectangle(frame, rect, (0, 255, 0), 2)


        # largest_contour = max(contours, key=cv2.contourArea)

        # if max_contour is not None:
        #     x, y, w, h = cv2.boundingRect(max_contour)
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


     

        if mean_diff> 25:
            count = 0
            lightsOn = True
            print("MOTION")
            if len(top_contours) > 0:
                frame_count += 1
                # cv2.imwrite('/users/hamza/Downloads/Frames/punch/' + str(frame_count) + '.png', frame)
        else:
            if lightsOn and count >= 30:
                print("NEW FRAMEEEE")
                init_frame = get_original_frame()
                lightsOn = False
            
                
            # lightsOn = False


            print("NOTHING")
        
        
        if abs(prev_diff - mean_diff) <= 1 and mean_diff > 3:
            count += 1
            # print(count)
        else:
            count = 0

        prev_diff = mean_diff

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