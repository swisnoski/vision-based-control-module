import numpy as np
import cv2 as cv
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *
from time import sleep


### CALIBRATION
images = ImageCollection("C:/Users/swisnoski/OneDrive - Olin College of Engineering/2025_01 Spring/FunRobo/Final Project/vision-control-venv/vision-based-control-module/calibration_imgs/*.png")
K, distortion, _ = CentralCamera.images2C(images, gridshape=(9, 6), squaresize=30e-3)

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


### DRAW AXES FUNCTION (CHESSBOARD)
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 2)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 2)
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



### DRAW RECTANGLE (ACURO BOARD)
gridshape = (5, 7)
square_size = 32e-3
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
    img = cv.polylines(img.image, [imgpts], isClosed=True, color=(0, 255, 255), thickness=2)

    return img


### ARUCO BOARD FRAME ESTIMATION
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
        pose_found = board.estimatePose(frame1, camera)
        if pose_found:
            pose = pose_found[0]
            board.draw(img, camera, length=0.05, thick=8)
            img = draw_rectangle(img, pose)
            
            if index == 10:
                # print("Pose of Board:\n", pose_found[0])
                index = 0
            index+=1 
    except:
        return img
    return img # Return numpy image (BGR)


### RED BOX COLOR DETECTION
# Lower “red end” of the spectrum
lower1 = np.array([0,   100, 100])   # H: 0–10 (reds), S/V: ≥100
upper1 = np.array([10,  255, 255])
# Upper “red end” of the spectrum
lower2 = np.array([160, 100, 100])   # H: 160–179 (reds), S/V: ≥100
upper2 = np.array([179, 255, 255])
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
red_positions = np.empty(shape=[0, 2])

def draw_red_boxes(frame, target_rgb=(230, 50, 50)):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.bitwise_or(cv.inRange(hsv, lower1, upper1), cv.inRange(hsv, lower2, upper2))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel, iterations=1)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    target_bgr = np.array(target_rgb[::-1], dtype=np.float32)
    out = frame.copy()
    red_index = 0
    red_positions = np.empty(shape=[0, 2])
    robo_position = None
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
            inside = False
            for point in cnt:
                pt = tuple(int(x) for x in point[0])
                if board_contour is not None:
                    result = cv.pointPolygonTest(board_contour, pt, True)
                    if result >= 0:
                        inside = True
                        break
            if inside:
                x, y, w, h = cv.boundingRect(cnt)
                red_positions = np.append(red_positions, [[x + w/2, y + h/2]], axis=0)
                out, robo_position = calc_object_positions(Image(out, colororder='BGR'), red_positions)
                red_index += 1
                # cv.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
    return out, robo_position


### CONVERT IMAGE FRAME TO BOARD/CAMERA/ROBOT FRAME 

def pixels_to_board(u, v, pose):
    # Invert camera intrinsics
    Kinv = np.linalg.inv(K)
    uv1 = np.array([u, v, 1.0]).reshape(3, 1)
    ray_cam = Kinv @ uv1  # Ray in camera coordinates

    # Convert board pose to rotation and translation
    R = pose.R
    t = pose.t.reshape(3, 1)

    # Transform ray into board coordinate frame
    ray_world = R.T @ (ray_cam - t)
    return ray_world


# Precompute this once at module scope:
Kinv = np.linalg.inv(K)


# Angles in radians
theta_x = np.radians(-155)
theta_z = np.radians(-90)

# Rotation about X
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(theta_x), -np.sin(theta_x)],
    [0, np.sin(theta_x),  np.cos(theta_x)]
])

# Rotation about Z (negative 90°)
Rz = np.array([
    [np.cos(theta_z), np.sin(theta_z), 0],
    [-np.sin(theta_z),  np.cos(theta_z), 0],
    [0,                0,               1]
])

# Combined: first Rx, then Rz
R_cam_to_robot = Rz @ Rx

t_cam_to_robot = np.array([-0.16, 0, 0.23]).reshape(3,1) # 3x1 numpy array

def pixel_to_board_camera_robot_xy(u, v, pose, board_z=-0.02):
    # Step 1: Back-project pixel to normalized ray in camera frame
    uv1     = np.array([u, v, 1.0]).reshape(3,1)
    ray_cam = Kinv @ uv1

    # Step 2: Pose of board relative to camera
    R, t     = pose.R, pose.t.reshape(3,1)

    # Step 3: Ray in board frame
    ray_b    = R.T @ (ray_cam - t)
    cam_b    = -R.T @ t

    if abs(ray_b[2,0]) < 1e-6:
        raise ValueError("Ray is parallel to board plane")

    scale = (board_z - cam_b[2,0]) / ray_b[2,0]
    if scale < 0:
        raise ValueError("Intersection is behind the camera")

    # Step 4: Intersection point
    point_board = cam_b + scale * ray_b
    Xb, Yb, Zb  = point_board.flatten()

    # Step 5: In camera frame
    point_cam = R @ point_board + t
    Xc, Yc, Zc = point_cam.flatten()

    # Step 6: In robot frame
    point_robot = R_cam_to_robot @ point_cam 
    point_robot = point_robot + t_cam_to_robot
    return point_robot.flatten()



### DRAW AXIS
def draw_position_axis(img, postion, imgpts):
    center = tuple(postion.ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, center, tuple(imgpts[0].ravel()), (0,0,255), 3)
    img = cv.line(img, center, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv.line(img, center, tuple(imgpts[2].ravel()), (255,0,0), 3)
    return img      



### RED BOX POSITION FINDER 

R_predefined = np.array([
    [1,  0,  0],   # X right
    [0,  1,  0],   # Y forward
    [0,  0, 1],   # Z down
], dtype=np.float32)

def calc_object_positions(frame, positions):
    img = np.array(frame.bgr, dtype=np.uint8)
    try:
        pose_found = board.estimatePose(frame, camera)
        if pose_found:
            pose = pose_found[0]
            R = pose.R  # 3x3 rotation matrix
            T = pose.t
            tvec = T.reshape(3,1)
            rvec, _ = cv.Rodrigues(R)
            for position in positions:
                u, v = position
                robo_position = pixel_to_board_camera_robot_xy(u, v, pose)
                position_board = pixels_to_board(u, v, pose)
                obj_axis = np.float32([
                    position_board.flatten(),
                    (position_board + R_predefined @ np.array([[0.1], [0], [0]])).flatten(),
                    (position_board + R_predefined @ np.array([[0], [0.1], [0]])).flatten(),
                    (position_board + R_predefined @ np.array([[0], [0], [0.1]])).flatten()
                ])
                imgpts, _ = cv.projectPoints(obj_axis[1:], rvec, tvec, K, zero_dist)
                img = draw_position_axis(img, np.array([[u, v]]), imgpts)
    except:
        return img, None
    return img, robo_position
            



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
ee_position = None

while True:
    ret, frame0 = cap.read()
    
    
    if ret:
        frame1 = undistort(frame0) # Undistort
        frame2 = april_tag_board_corner(frame1) # Draw ArUco board
        frame3 = convert_for_imshow(frame2)
        frame4, ee_position = draw_red_boxes(frame3)
        #frame5 = chessboard_corner(Image(frame4, colororder='BGR'))
    else:
        print("Failed to capture frame")
        break

    # Show
    frame6 = convert_for_imshow(frame4)
    cv.imshow("RED CUBE DETECTOR", frame6)
    if ee_position is not None:
        print(ee_position)
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()


def main(ret, frame0):
    if ret:
        frame1 = undistort(frame0) # Undistort
        frame2 = april_tag_board_corner(frame1) # Draw ArUco board
        frame3 = convert_for_imshow(frame2)
        frame4, ee_position = draw_red_boxes(frame3)
        #frame5 = chessboard_corner(Image(frame4, colororder='BGR'))
        frame6 = convert_for_imshow(frame4)
        cv.imshow("RED CUBE DETECTOR", frame6)
        if ee_position is not None:
            print(ee_position)
            return ee_position
        cv.waitKey(1)
    return None
