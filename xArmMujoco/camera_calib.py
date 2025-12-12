import mujoco
import numpy as np
import random
import cv2 

# def move_joint(joint_name, pos, rot):
#     joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
#     qpos_addr = model.jnt_qposadr[joint_id]

#     data.qpos[qpos_addr : qpos_addr + 3] = np.array(pos)
#     data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array(rot)

#     mujoco.mj_forward(model, data)
#     mujoco.mj_step(model, data)

# def get_image():
#     cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "my_camera")
#     mujoco.mj_step(model, data)
#     renderer.update_scene(data, camera=cam_id)
#     return renderer.render()

# model =  mujoco.MjModel.from_xml_path("C:\\Users\\srkia\\Desktop\\xArmSim\\ufactory_xarm7\\scene_calib.xml")
# data = mujoco.MjData(model)
# renderer = mujoco.Renderer(model, height=1080, width=1920)

# for i in range(20):
#     image = get_image()
#     move_joint("board_joint", [random.uniform(0.15, 0.45), random.uniform(-0.4, 0.4), 0.0], [1.0, 0.0, 0.0, random.uniform(-1.57, 1.58)])
#     cv2.imwrite(f"C:\\Users\\srkia\\Desktop\\xArmSim\\xArmMujoco\\calib_imgs\\calib_image{i}.png", image)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.02, aruco_dict)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
charuco_detector = cv2.aruco.CharucoDetector(board)
image_paths = []

for i in range(20):
    if((i == 7) or (i == 14) or (i == 15) or (i ==17)):
        pass
    image_paths.append(f"C:\\Users\\srkia\\Desktop\\xArmSim\\xArmMujoco\\calib_imgs\\calib_image{i}.png")

all_charuco_corners = []
all_charuco_ids = []

all_obj_points = []
all_img_points = []

sample_img = cv2.imread(image_paths[0])
img_size = sample_img.shape[:2][::-1] 
del sample_img

for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

    result = charuco_detector.detectBoard(gray)
    charuco_corners, charuco_ids = result[2:]    
   
    if charuco_ids is not None and len(charuco_ids) >= 4:
        ret_interp, current_img_points, current_obj_points = cv2.aruco.interpolateCornersCharuco(
        charuco_corners,
        charuco_ids,
        gray,
        board)
        print(f"Current img points {current_img_points}, {current_obj_points}")

        if ret_interp and current_img_points is not None and current_obj_points is not None:
            current_img_points = np.squeeze(current_img_points)
            current_obj_points = np.squeeze(current_obj_points)

            if current_img_points.ndim == 2 and current_img_points.shape[1] == 2:
                all_img_points.append(current_img_points.astype(np.float32))
                all_obj_points.append(current_obj_points.astype(np.float32))
            else:
                print("invalid shape", current_img_points.shape)

if len(all_obj_points) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points,      
        all_img_points,      
        img_size,            
        None,                
        None                 
    )

    print(f"Calibration successful! Reprojection Error: {ret}")
    print("\nCamera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)
    

else:
    print("No valid ChArUco corners detected across all images for calibration. Check your images.")
