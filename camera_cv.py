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


### DRAW AXES FUNCTION
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


### CHESSBOARD CORNER OVERLAY
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = np.zeros((9*6,3), np.float32)
objpoints[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3)

def chessboard_corner(frame):
    img = np.array(frame.bgr, dtype=np.uint8)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret2, corners = cv.findChessboardCorners(gray, (9,6), None)
    if ret2:
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        _, rvecs, tvecs = cv.solvePnP(objpoints, corners2, K, distortion)
        imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, K, distortion)
        img = draw(img, corners2, imgpts)
    return img


### DRAW RECTANGLE ACURO BOARD
gridshape = (5, 7)
square_size = 33e-3
spacing_size = 3.2e-3
boardheight = (gridshape[1]*square_size + (gridshape[1]-1)*spacing_size)
boardwidth = (gridshape[0]*square_size + (gridshape[0]-1)*spacing_size)
board_corners = np.array([
        [0, 0, 0],
        [0, boardheight, 0],
        [boardwidth, boardheight, 0],
        [boardwidth, 0, 0]
    ], dtype=np.float32)
zero_dist = np.zeros_like(distortion)
board_contour=None

def draw_rectangle(img, pose):
    global board_contour
    R = pose.R  # 3x3 rotation matrix
    tvec = pose.t.reshape(3,1)
    rvec, _ = cv.Rodrigues(R)
    # Project 3D points to 2D image plane
    imgpts, _ = cv.projectPoints(board_corners, rvec, tvec, K, zero_dist)
    imgpts = imgpts.astype("int32").reshape(-1, 2)
    board_contour = imgpts.reshape(-1, 1, 2)

    # Draw rectangle on the image
    img = cv.polylines(img.image, [imgpts], isClosed=True, color=(0, 255, 255), thickness=4)

    return img


### ARUCO BOARD SETUP + ESTIMATION
objpoint_aruco = np.array([
    [0, 0, 0],
    [0, boardheight, 0],
    [boardwidth, boardheight, 0],
    [boardwidth, 0, 0]
], dtype=np.float32)
board = ArUcoBoard(gridshape, square_size, spacing_size, dict="6x6_1000", firsttag=0)
C = np.column_stack((K, np.array([0, 0, 1])))
est = CentralCamera.decomposeC(C)
camera = CentralCamera(f=est.f[0], rho=est.rho[0], imagesize=[480, 640], pp=est.pp)
index = 0

def april_tag_board_corner(frame):
    img = Image(frame, colororder='BGR')  # MVTB expects BGR
    global index
    try:
        pose_found = board.estimatePose(img, camera)
        if pose_found:
            pose = pose_found[0]
            board.draw(img, camera, length=0.05, thick=4)
            img = draw_rectangle(img, pose)
            
            if index == 10:
                print("Pose of Board:\n", pose_found[0])
                index = 0
            index+=1 
    except:
        return img
    return img # Return numpy image (BGR)


### COLOR DETECTION

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
            # Check if all points of the contour are inside the board_contour
            inside = True
            for point in cnt:
                pt = tuple(int(x) for x in point[0])
                result = cv.pointPolygonTest(board_contour, pt, False)
                if result <= 0:
                    inside = False
                    break
            if inside:
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return out


### CONVERSION FUNCTION

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
    ret, frame0 = cap.read()
    if ret:
        frame1 = undistort(frame0) # Undistort
        frame2 = april_tag_board_corner(frame1) # Draw ArUco board
        frame3 = convert_for_imshow(frame2)
        frame4 = draw_red_boxes(frame3)
        # final_frame = chessboard_corner(Image(aruco_frame, colororder='BGR'))

    else:
        print("Failed to capture frame")
        break

    # Show
    frame5 = convert_for_imshow(frame4)
    cv.imshow("RED CUBE DETECTOR", frame5)
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()
