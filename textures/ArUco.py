import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for marker_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 400)
    cv2.imwrite(f"C:\\Users\\surya\\Documents\\Robotics\\irob-project\\VisServ\\textures\\aruco_{marker_id}.png", img)
