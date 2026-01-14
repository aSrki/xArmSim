import mujoco
import mujoco.viewer
import roboticstoolbox as rtb
from spatialmath import SE3
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from PIL import Image

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

model = mujoco.MjModel.from_xml_path("C:\\Users\\srkia\\Desktop\\xArmSim\\ufactory_xarm7\\scene_calib.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=1080, width=1920)
aruco = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "board")

# for i in range(10):
#     move_joint("board_joint", [random.uniform(0.15, 0.45), random.uniform(-0.4, 0.4), 0.3], R.from_euler('xyz', [random.uniform(-30.0, 30.0), random.uniform(0.0, 30.0), random.uniform(0.0, 3.14)], degrees=True).as_quat())

#     image = get_image()

#     filename = f"C:\\Users\\srkia\\Desktop\\xArmSim\\xArmMujoco\\calib_imgs_3d\\image{i}.png"
#     img = Image.fromarray(image)
#     img.save(filename)

for i in range(10):
    move_joint("board_joint", [random.uniform(0.15, 0.45), random.uniform(-0.4, 0.4), 0.02], R.from_euler('xyz', [0.0, 0.0, random.uniform(0.0, 3.14)], degrees=True).as_quat())

    image = get_image()

    filename = f"C:\\Users\\srkia\\Desktop\\xArmSim\\xArmMujoco\\calib_imgs\\calib_image{i}.png"
    img = Image.fromarray(image)
    img.save(filename)