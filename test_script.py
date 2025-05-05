import numpy as np
import cv2 as cv
from time import sleep

from camera_cv import cv_main

video_id = 0
cap = cv.VideoCapture(video_id)
ee_position = None

def get_coordinates():
    index = 0
    while True:
        ret, frame = cap.read()
        ee_position = cv_main(ret, frame)
        if ee_position is not None:
            index += 1
            sleep(1)
            if index > 5:
                x, y, z = ee_position
                x = -(x-0.01)
                y = -y
                return [x,y,z]

