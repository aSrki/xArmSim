import mujoco
import mujoco.viewer
import os
import roboticstoolbox as rtb
from spatialmath import SE3
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import random

def move_joint(joint_name, pos, rot):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[joint_id]

    data.qpos[qpos_addr : qpos_addr + 3] = np.array(pos)
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array(rot)

    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)

def get_image():
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "my_camera")
    mujoco.mj_step(model, data)
    renderer.update_scene(data, camera=cam_id)
    return renderer.render()

def find_centers(corners, ids):
    centers = [[0, 0], [0, 0], [0, 0], [0, 0]]
    for i, corner in enumerate(corners):
        marker_corners = corner.reshape(4, 2)
        
        center_x = int(np.mean(marker_corners[:, 0]))
        center_y = int(np.mean(marker_corners[:, 1]))

        if(ids[i][0] > 30):
            continue

        centers[ids[i][0] - 21] = [center_x, center_y]

    return centers

def transform_image(corners, ids, image):
    pts1 = np.float32(find_centers(corners, ids))
    pts2 = np.float32([[0,0],[1000,0],[0,500],[1000,500]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(image, M, (1000,500))

def find_marker_center(corners):
    for i, corner in enumerate(corners):
        marker_corners = corner.reshape(4, 2)
        
        center_x = int(np.mean(marker_corners[:, 0]))
        center_y = int(np.mean(marker_corners[:, 1]))

    return center_x, center_y

robot = rtb.Robot.URDF("C:\\Users\\srkia\\Desktop\\xArmSim\\ufactory_xarm7\\xarm7.urdf")
MODEL_NAME = "ufactory_xarm7"
MODEL_FILE = "scene.xml"

model_path = os.path.join(MODEL_NAME, MODEL_FILE)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=1080, width=1920)
aruco = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "aruco42")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# move_joint("aruco_joint", [random.uniform(0.15, 0.45), random.uniform(-0.4, 0.4), 0.0], [1.0, 0.0, 0.0, 0.0])

image = get_image()

corners, ids, rejected = detector.detectMarkers(image)

image = transform_image(corners, ids, image)

corners, ids, rejected = detector.detectMarkers(image)

center_x, center_y = find_marker_center(corners)

target_pose = SE3.Trans(center_y/1000.0, (center_x - 500.0)/1000, 0.0) * SE3.RPY(0.0, 3.14, 0.0)
ik_sol = robot.ik_LM(target_pose) 

plt.grid(False)
plt.imshow(image)
plt.show(block=False)
plt.pause(0.1)

data.ctrl = ik_sol[0]
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)