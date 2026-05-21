# %%
import cv2
import numpy as np
import glob
import os
import json
import rawpy
import matplotlib.pyplot as plt

ARUCO_DICT = cv2.aruco.DICT_6X6_50
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 10
SQUARE_LENGTH = 23.8
MARKER_LENGTH = 17.5
img_dir = 'ChArUcoCalibDNGs'
images = np.empty((4872, 5568, 3, 109), dtype=np.uint8)
count = 0
for i in range(1,109):
    with rawpy.imread(os.path.join(img_dir, f'ChArUco_Calib_{i}.dng')) as raw:
        count += 1
        rgb = raw.postprocess(no_auto_bright=False, output_bps=8, use_camera_wb=True, half_size=False, output_color=rawpy.ColorSpace.sRGB, )
        print(count)
        images[..., i] = rgb

# %%
# Define the aruco dictionary, charuco board and detector
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_HORIZONTALLY, SQUARES_VERTICALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
board.setLegacyPattern(True) 
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)
with open('camera_calibration.json', 'r') as f:
    data = json.load(f)
mtx = np.array(data['camera_matrix'])
dist = np.array(data['dist_coeff'])

# Load images from directory
# image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".dng")]
all_charuco_ids = []
all_charuco_corners = []
# marker_ids = []
# marker_corners = []
# Loop over images and extraction of corners
for i in range(1, 109):
    image = images[..., i]
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    image = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    imgSize = image.shape
    image_copy = image.copy()
    marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(image)
    marker_corners = np.array(marker_corners)
    # print(f"Image {i}: Detected {len(marker_ids)} markers. IDs: {marker_ids.flatten()}")
    
    if len(marker_ids) > 0: # Check if any markers were detected
        cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
        ret,charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
    if charucoIds is None:
        print(f"Image {i+1}: interpolateCornersCharuco returned None for charucoIds.")
    else:
        print(f"Image {i+1}: interpolateCornersCharuco found {len(charucoIds)} ChArUco corners.")
        if len(charucoCorners) > 3: # A common heuristic for enough corners for calibration
            all_charuco_corners.append(charucoCorners)
            all_charuco_ids.append(charucoIds)
            print(f"  Added ChArUco data for image {i+1}.")

        else:
            print(f"  Image {i+1}: Interpolated ChArUco corners found, but not enough (>3) for calibration ({len(charucoCorners)} found).")
    if charucoIds is not None and len(charucoCorners) > 3:
        all_charuco_corners.append(charucoCorners)
        all_charuco_ids.append(charucoIds)
# %%
# Calibrate camera with extracted information
retval,mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, imgSize,None,None)

calibration_data = {
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
    "rvecs": [rvec.tolist() for rvec in rvecs],
    "tvecs": [tvec.tolist() for tvec in tvecs],
    "reprojection_error": retval
}

with open('camera_calibration_Charuco_noGuess.json', 'w') as f:
    json.dump(calibration_data, f, indent=4)


# %%
