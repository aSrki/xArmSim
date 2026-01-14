import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((6, 8), 0.1, 0.05, dictionary) 
board_image = board.generateImage((6000, 8000))
cv2.imwrite("charuco_board2.png", board_image)