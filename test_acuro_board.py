import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import math
from math import pi
np.set_printoptions(
    linewidth=120, formatter={
        'float': lambda x: f"{0:8.4g}" if abs(x) < 1e-10 else f"{x:8.4g}"})
np.random.seed(0)
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *

images = ImageCollection("C:/Users/swisnoski/OneDrive - Olin College of Engineering/2025_01 Spring/FunRobo/Final Project/vision-control-venv/vision-based-control-module/calibration_imgs/*.png")
K, distortion, frames = CentralCamera.images2C(images, gridshape=(9, 6), squaresize=24e-3)
u0 = K[0,2]
v0 = K[1,2]
fpixel_width = K[0,0]
fpixel_height = K[1,1]
k1, k2, p1, p2, k3 = distortion
U, V = images[1].meshgrid()

x = (U - u0) / fpixel_width
y = (V - v0) / fpixel_height
r = np.sqrt(x**2 + y**2)
delta_x = x * (k1*r**2 + k2*r**4 + k3*r**6) + 2*p1*x*y + p2*(r**2 + 2*x**2)
delta_y = y * (k1*r**2 + k2*r**4 + k3*r**6) + p1*(r**2 + 2*y**2) + p2*x*y
xd = x + delta_x 
yd = y + delta_y
Ud = xd * fpixel_width + u0
Vd = yd * fpixel_height + v0

scene = Image.Read("C:/Users/swisnoski/OneDrive - Olin College of Engineering/2025_01 Spring/FunRobo/Final Project/vision-control-venv/vision-based-control-module/img/ex-aruco-board.png", rgb=False)
undistorted_scene = scene.warp(Ud, Vd)

gridshape = (5, 7)
square_size = 28e-3
spacing_size = 3e-3
board = ArUcoBoard(gridshape, square_size, spacing_size, dict="6x6_1000", firsttag=0)


C = np.column_stack((K, np.array([0, 0, 1]))) # camera matrix
est = CentralCamera.decomposeC(C)

camera = CentralCamera(f=est.f[0], rho=est.rho[0], imagesize=[480, 640], pp=est.pp)

board.estimatePose(undistorted_scene, camera)
board.draw(undistorted_scene, camera, length=0.05, thick=4)


img_to_show = undistorted_scene.image[..., ::-1]  # Convert BGR -> RGB
plt.imshow(img_to_show)
plt.axis('off')
plt.title("Pose Estimation on Undistorted Scene")
plt.show()