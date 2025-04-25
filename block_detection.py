import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
import cv2 as cv
from scipy import linalg
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *
from time import sleep

### CALIBRATION
images = ImageCollection("C:/Users/swisnoski/OneDrive - Olin College of Engineering/2025_01 Spring/FunRobo/Final Project/vision-control-venv/vision-based-control-module/calibration_imgs/*.png")
K, distortion, frames = CentralCamera.images2C(images, gridshape=(9, 6), squaresize=24e-3)

u0 = K[0, 2]
v0 = K[1, 2]
fpixel_width = K[0, 0]
fpixel_height = K[1, 1]
k1, k2, p1, p2, k3 = distortion

### UNDISTORT FUNCTION
def undistort(frame):
    frame = Image(frame, colororder='BGR')  # Make sure it's interpreted correctly
    U, V = frame.meshgrid()
    x = (U - u0) / fpixel_width
    y = (V - v0) / fpixel_height
    r = np.sqrt(x**2 + y**2)
    delta_x = x * (k1*r**2 + k2*r**4 + k3*r**6) + 2*p1*x*y + p2*(r**2 + 2*x**2)
    delta_y = y * (k1*r**2 + k2*r**4 + k3*r**6) + p1*(r**2 + 2*y**2) + p2*x*y
    xd = x + delta_x
    yd = y + delta_y
    Ud = xd * fpixel_width + u0
    Vd = yd * fpixel_height + v0
    return frame.warp(Ud, Vd)  # Returns an Image object



def draw_red_boxes(frame, target_rgb=(230, 50, 50)):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower1 = np.array([170, 130, 170])
    upper1 = np.array([179, 255, 255])
    lower2 = np.array([0, 130, 170])
    upper2 = np.array([10, 255, 255])
    mask = cv.bitwise_or(cv.inRange(hsv, lower1, upper1), cv.inRange(hsv, lower2, upper2))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel, iterations=1)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    target_bgr = np.array(target_rgb[::-1], dtype=np.float32)
    out = frame.copy()
    for cnt in contours:
        if cv.contourArea(cnt) < 500:
            continue
        c_mask = np.zeros(mask.shape, np.uint8)
        cv.drawContours(c_mask, [cnt], -1, 255, -1)
        mean_val = cv.mean(frame, mask=c_mask)[:3]
        mean_bgr = np.array(mean_val, dtype=np.float32)
        dist = np.linalg.norm(mean_bgr - target_bgr)
        if dist < 60:
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return out



def convert_for_imshow(frame):
    if isinstance(frame, Image):
        arr = np.array(frame.bgr, dtype=np.uint8)
        return arr
    
    elif isinstance(frame, np.ndarray):  # If it's already a NumPy array
        if frame.shape[-1] == 3:  # If it's a color image (3 channels)
            frame = frame.astype(np.uint8)  # Ensure it's uint8 type
        elif len(frame.shape) == 2:  # Grayscale image (single channel)
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)  # Convert grayscale to BGR for OpenCV

    else:
        raise TypeError("Unsupported frame type. Expected PIL Image or NumPy array.")
    return frame


### MAIN LOOP
video_id = 0
cap = cv.VideoCapture(video_id)

while True:
    ret, frame = cap.read()
    if ret:
        # Undistort
        undistorted = undistort(frame)
        frame = convert_for_imshow(undistorted)
        frame = draw_red_boxes(frame)
        # Draw ArUco board
        # aruco_frame = np.array(aruco_frame.bgr, dtype=np.uint8)
        # Draw chessboard on top
        # final_frame = chessboard_corner(Image(aruco_frame, colororder='BGR'))

    else:
        print("Failed to capture frame")
        break

    # Show
    frame = convert_for_imshow(frame)
    cv.imshow("RED CUBE DETECTOR", frame)
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()