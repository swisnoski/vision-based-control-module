import cv2 as cv
from camera_cv import cv_main

video_id = 0
cap = cv.VideoCapture(video_id)
ee_position = None

while True:
    ret, frame = cap.read()
    ee_position = cv_main(ret, frame)
    print(ee_position)

