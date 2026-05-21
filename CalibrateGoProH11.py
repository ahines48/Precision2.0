# %%
import cv2
import numpy as np
import glob
import os
import json

def pause():
    programPause = input("Press the <ENTER> key to continue...")
    return
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
squaresize = 17 # length of side in mm
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)*squaresize
 
# Arrays to store object points and image points from all the images.
objpoints = [] 
imgpoints = []

image_folder = 'CheckerCalib_Logi'
image_pattern = os.path.join(image_folder, '*.[jJ][pP][gG]')

image_files = glob.glob(image_pattern)

images = []
for fname in image_files:
    img = cv2.imread(fname)
    if img is not None:
        images.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        cv2.imshow('img', gray)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
         # Find the chess board corners
    ret, corners = cv2.findChessboardCornersSB(gray, (7,10))
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,10), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

# %%    
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv2.CALIB_RATIONAL_MODEL) 
print("Camera matrix (units: mm):")
print(mtx)

img = cv2.imread('headPhotos/SeanLinearHDR/Sean_test_left_HDR.JPG')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('headPhotos/SeanLinearHDR/unwarpLinear_Sean_Left.JPG', dst)

# Calculate Reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

calibration_data = {
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
    "image_size": gray.shape[::-1],
    "rvecs": [rvec.tolist() for rvec in rvecs],
    "tvecs": [tvec.tolist() for tvec in tvecs],
    "reprojection_error": mean_error / len(objpoints)
}

with open('Logi_calibration.json', 'w') as f:
    json.dump(calibration_data, f, indent=4)
# undistort an image

