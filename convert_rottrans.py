import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import json

with open('optimized_pose.json', 'r') as f:
    opt_data = json.load(f)


# Make sure you are using the optimized pose from Cell 3, or the refined PnP pose
rvec = np.array(opt_data['rvec'], dtype=np.float32) # or 'rotation' if you just want the PnP output
tvec = np.array(opt_data['tvec'], dtype=np.float32) # or 'translation'

rmat_cv, _ = cv2.Rodrigues(rvec)

# OpenCV → Blender axis flip matrix
cv_to_blend = np.array([
    [1,  0,  0],
    [0,  -1,  0],
    [0,  0, -1]
], dtype=np.float64)

# Camera location in world space (OpenCV coords), then flip to Blender coords
cam_loc_blend = -rmat_cv.T @ tvec.reshape(3, 1)

R_cam2world_cv = rmat_cv.T
rmat_blend = R_cam2world_cv @ cv_to_blend
# Blender camera object rotation = this matrix itself (not transposed)
rot = R.from_matrix(rmat_blend)
euler_angles_deg = rot.as_euler('xyz', degrees=True)  # ← 'xyz' not 'yxz'

print("\n--- BLENDER UI INPUTS ---")
print(f"Location X: {cam_loc_blend[0, 0]:.4f}")
print(f"Location Y: {cam_loc_blend[1, 0]:.4f}")
print(f"Location Z: {cam_loc_blend[2, 0]:.4f}")
print("-" * 25)
print(f"Rotation X: {euler_angles_deg[0]:.4f}°")
print(f"Rotation Y: {euler_angles_deg[1]:.4f}°")
print(f"Rotation Z: {euler_angles_deg[2]:.4f}°")

# %% --- 1. Load Calibration and Image ---
with open('Logi_calibration.json', 'r') as f:
    calib_data = json.load(f)

camera_matrix = np.array(calib_data['camera_matrix'])

# We need the image to get the exact width and height in pixels
photo_path = 'Camera1_2023-01-15_10-14-42.jpg'
image = cv2.imread(photo_path)
h, w = image.shape[:2]

# --- 2. Extract OpenCV Intrinsics ---
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

# --- 3. Convert to Blender Parameters ---
# We will use a standard Full Frame sensor width of 36mm.
# (You can use any size, but 36mm is Blender's default and makes the math easy).
sensor_width_mm = 36.0 

# Calculate the equivalent focal length in mm
focal_length_mm = fx * (sensor_width_mm / w)

# Calculate Lens Shift
# OpenCV's origin (0,0) is top-left. Blender's origin is the center of the image.
# We also have to account for Blender's shift scaling, which is relative to the largest dimension.
max_dim = max(w, h)
shift_x = ((w / 2.0)-cx) / max_dim
shift_y = (cy-(h / 2.0)) / max_dim # Note the inverted Y logic

print("\n=== BLENDER CAMERA SETTINGS ===")
print("1. Output Properties Tab (Printer Icon):")
print(f"   Resolution X : {w} px")
print(f"   Resolution Y : {h} px")
print("-" * 31)
print("2. Object Data Properties Tab (Camera Icon):")
print("   Lens:")
print(f"      Type         : Perspective")
print(f"      Focal Length : {focal_length_mm:.4f} mm")
print("   Shift:")
print(f"      X            : {shift_x:.4f}")
print(f"      Y            : {shift_y:.4f}")
print("   Camera:")
print(f"      Sensor Fit   : Horizontal")
print(f"      Sensor Width : {sensor_width_mm:.1f} mm")
print("===============================\n")