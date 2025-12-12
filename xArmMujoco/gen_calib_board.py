import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.02, dictionary) # squaresX, squaresY, squareLength, markerLength
board_image = board.generateImage((600, 800)) # imageSize
cv2.imwrite("charuco_board.png", board_image)