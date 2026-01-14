import numpy as np
import cv2 

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((6, 8), 0.1, 0.05, dictionary) 
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
charuco_detector = cv2.aruco.CharucoDetector(board)
image_paths = []

for i in range(10):
    image_paths.append(f"xArmMujoco\\calib_imgs_3d\\image{i}.png")

all_charuco_corners = []
all_charuco_ids = []

for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if charuco_ids is not None and len(charuco_ids) > 10: 
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

ret, camera_matrix, dist_coeffs, _, _= cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None, 
    distCoeffs=None
)

print("Reprojection Error:", ret)
print("Camera Matrix:\n", camera_matrix)
print("Distance coeffs:\n", dist_coeffs)

all_charuco_corners = []
all_charuco_ids = []

for image_path in image_paths:
    img = cv2.imread(f"xArmMujoco\\calib_imgs\\calib_image2.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if charuco_ids is not None and len(charuco_ids) > 4: 
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

_, _, _, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=camera_matrix, 
    distCoeffs=dist_coeffs
)

rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
transformation_matrix = np.eye(4)
transformation_matrix[0:3, 0:3] = rotation_matrix
transformation_matrix[0:3, 3] = tvecs[0].flatten()
print("Homogeneous Transformation Matrix:\n", np.linalg.inv(transformation_matrix))