# %%
import scipy

from Photo_PointSelectionGui import select_2d_points as photoGui
from Mesh_PointSelectionGui import select_3d_points as meshGui
from Point_selection import pick_points
import json
import os
import numpy as np
import cv2
import open3d as o3d
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

json_file = 'picked_points_Logi_1612_S10_C4.json'
mesh_file = '1613_LDMesh_260109.stl'
photo_path = 'Camera4_2023-01-15_10-14-44.jpg'

# Run the concurrent picker
# mesh_points, head_points = pick_points(mesh_file, photo_path, json_file)

# # Verify outputs
# print(f"Mesh points shape: {mesh_points.shape}")
# print(f"Image points shape: {head_points.shape}")

with open('picked_points_Logi_1612_S10_C4.json', 'r') as f:
    points = json.load(f)


mesh_points_list = points[:-1]
head_points_list = points[-1]

mesh_points = np.array(mesh_points_list, dtype=np.float32)
head_points = np.array(head_points_list, dtype=np.float32)  

print(mesh_points.shape)
print(head_points.shape)

with open('Logi_calibration.json', 'r') as f:
    calib_data = json.load(f)

camera_matrix = calib_data['camera_matrix']
dist_coeff = calib_data['dist_coeff']
reproj_error = calib_data['reprojection_error']

camera_matrix = np.array(camera_matrix)
dist_coeff = np.array(dist_coeff)

flags = cv2.SOLVEPNP_SQPNP

[success, rotation, translation] = cv2.solvePnP(mesh_points, head_points,camera_matrix, dist_coeff, flags=flags)

# --- 5. Load Image and Mesh ---
image = cv2.imread(photo_path)
mesh = o3d.io.read_triangle_mesh(mesh_file)
mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

# --- 6. Project Points to 2D ---
# Project ALL mesh vertices for the wireframe overlay
proj_vertices, _ = cv2.projectPoints(vertices, rotation, translation, camera_matrix, dist_coeff)
proj_vertices = proj_vertices.squeeze().astype(int)

# Project ONLY the selected 3D target points for the green dots
proj_selected_pts, _ = cv2.projectPoints(mesh_points, rotation, translation, camera_matrix, dist_coeff)
proj_selected_pts = proj_selected_pts.squeeze().astype(int)

# --- 7. Visualization: Alpha Blended Mesh & Error Lines ---
# OpenCV uses BGR, Matplotlib uses RGB. Convert the image first.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create the Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(10, 8)) # Adjust figsize as needed
ax.imshow(image_rgb)

lines = []
for tri in triangles:
    # Get the 3 vertices of the triangle, plus the first one again to close it
    pts = [proj_vertices[i] for i in tri]
    pts.append(proj_vertices[tri[0]]) 
    lines.append(pts)

# Plot all triangles at once with 30% opacity (alpha=0.3)
lc = LineCollection(lines, colors='cyan', linewidths=0.5, alpha=0.1)
ax.add_collection(lc)

# Draw points and error lines
for i, (p2d, p3d) in enumerate(zip(head_points, proj_selected_pts)):
    # Yellow Error Line connecting target (2D) to projection (3D)
    # plt.plot takes [x1, x2], [y1, y2]
    ax.plot([p2d[0], p3d[0]], [p2d[1], p3d[1]], color='yellow', linewidth=1.5)
    
    # Red Dot: Original 2D click
    ax.scatter(p2d[0], p2d[1], color='red', s=30, zorder=5) # zorder keeps dots on top of lines
    ax.text(p2d[0]+5, p2d[1]-5, str(i), color='red', fontsize=10, weight='bold')
    
    # Green Dot: Projected 3D point
    ax.scatter(p3d[0], p3d[1], color='lime', s=30, zorder=5)
    ax.text(p3d[0]+5, p3d[1]-5, str(i), color='lime', fontsize=10, weight='bold')

# Remove axes ticks for a cleaner image look
ax.axis('off')

# Save the figure
# bbox_inches='tight' removes the white padding around the saved image
plt.savefig("mesh_overlay_matplotlib.jpg", dpi=300, bbox_inches='tight')

# Display the plot
plt.title("PnP Projection & Error Lines")
plt.tight_layout()
plt.show()
# %% --- CELL 2: HEAD SEGMENTATION USING SAM ---
import torch
from segment_anything import sam_model_registry, SamPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Initialize Segment Anything Model (SAM) ---
sam_checkpoint = "sam_vit_h_4b8939.pth" # Ensure this is downloaded!
model_type = "vit_h"

try:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
except FileNotFoundError:
    print(f"ERROR: Could not find '{sam_checkpoint}'. Please download it first.")
    raise

# --- 2. Prepare Image for SAM ---
# OpenCV loads BGR, SAM expects RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# --- 3. Prompt SAM using your existing PnP points ---
# We use the 2D head_points you already clicked as "positive" foreground prompts
input_point = head_points 
input_label = np.ones(len(input_point)) # 1 indicates a foreground point

# --- 4. Run Inference ---
print("Running SAM inference...")
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False # We only want the single best consolidated mask
)

best_mask = masks[0] # Boolean array matching image dimensions

# --- 5. Visualize the Mask ---
plt.figure(figsize=(15, 7))

# Plot Original Image
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

# Plot Segmented Image
plt.subplot(1, 2, 2)
plt.imshow(image_rgb)

# Create a blue transparent overlay for the mask
mask_overlay = np.zeros((*best_mask.shape, 4))
mask_overlay[best_mask] = [0, 0, 1, 0.5] # RGBA (Blue, 50% transparent)
plt.imshow(mask_overlay)

# Overlay the prompt points
plt.scatter(input_point[:, 0], input_point[:, 1], color='red', marker='*', s=100, label="SAM Prompts")
plt.legend()
plt.title(f"SAM Head Segmentation (Score: {scores[0]:.2f})")
plt.axis('off')

plt.tight_layout()
plt.show()

# Optional: Save binary mask as an image
binary_mask_img = (best_mask * 255).astype(np.uint8)
cv2.imwrite("head_silhouette_mask.png", binary_mask_img)
print("Saved segmentation mask to head_silhouette_mask.png")


# %% --- CELL 3: POSE OPTIMIZATION USING SILHOUETTE MATCHING ---
import cv2
from scipy.optimize import minimize

print("Starting silhouette-based pose optimization...")
binary_mask_img = cv2.imread("head_silhouette_mask.png", cv2.IMREAD_GRAYSCALE)
# --- 1. Prepare the SAM Mask Distance Transform ---
# Get the edges of the SAM mask
mask_edges = cv2.Canny(binary_mask_img, 100, 200)

# Invert edges: 0 represents the edge, 255 represents non-edge
inverted_edges = 255 - mask_edges

# Compute Distance Transform (Distance to nearest edge)
# Every pixel now holds a float value representing its distance to the SAM contour
dt = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)

# --- 2. Define the Objective Function ---
def joint_loss(params, vertices, triangles, mesh_points, head_points, camera_matrix, dist_coeff, dt, img_shape, weight_points=2.0):
    # Unpack rotation and translation vectors
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:].reshape(3, 1)

    # ==========================================
    # PART 1: SILHOUETTE LOSS (Pixels)
    # ==========================================
    proj_v, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeff)
    proj_v = proj_v.squeeze().astype(int)

    mesh_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mesh_mask, proj_v[triangles], 255)
    mesh_edges = cv2.Canny(mesh_mask, 100, 200)
    
    edge_y, edge_x = np.where(mesh_edges > 0)

    if len(edge_y) == 0:
        return 1e6 # Penalize heavily if off-screen

    edge_y = np.clip(edge_y, 0, img_shape[0] - 1)
    edge_x = np.clip(edge_x, 0, img_shape[1] - 1)

    distances = dt[edge_y, edge_x]
    silhouette_score = np.mean(distances)

    # ==========================================
    # PART 2: POINT REPROJECTION LOSS (Pixels)
    # ==========================================
    # Project the 3D target points using the CURRENT optimizer parameters
    proj_pts, _ = cv2.projectPoints(mesh_points, rvec, tvec, camera_matrix, dist_coeff)
    proj_pts = proj_pts.squeeze()

    # Calculate the Euclidean distance (in pixels) between where the 3D point 
    # projected vs where you actually clicked in the 2D image
    point_errors = np.linalg.norm(proj_pts - head_points, axis=1)
    reprojection_score = np.mean(point_errors)

    # ==========================================
    # COMBINE AND RETURN
    # ==========================================
    # weight_points lets you decide which is more important. 
    # > 1.0 means prioritize the clicked points. < 1.0 prioritizes the silhouette.
    total_loss = silhouette_score + (reprojection_score * weight_points)
    
    return total_loss

# --- Run Optimization ---
initial_rvec = rotation.copy()
initial_tvec = translation.copy()
initial_params = np.hstack((initial_rvec.flatten(), initial_tvec.flatten()))

# Notice we now pass mesh_points and head_points into the args!
res = minimize(
    joint_loss, 
    initial_params,
    args=(vertices, triangles, mesh_points, head_points, camera_matrix, dist_coeff, dt, image.shape, .2),
    method='Powell',
    options={'disp': True, 'xtol': 1e-6, 'ftol': 1e-6}
)

opt_rotation = res.x[:3].reshape(3, 1)
opt_translation = res.x[3:].reshape(3, 1)

print("Optimization Complete.")

# --- 4. Visualize the Improved Alignment ---
# Project vertices with the NEW optimized pose
opt_proj_vertices, _ = cv2.projectPoints(vertices, opt_rotation, opt_translation, camera_matrix, dist_coeff)
opt_proj_vertices = opt_proj_vertices.squeeze().astype(int)

# OpenCV uses BGR, Matplotlib uses RGB. Convert the image first.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# ==========================================
# Subplot 1: Initial PnP Alignment
# ==========================================
axes[0].imshow(image_rgb)
axes[0].set_title("Initial PnP Alignment (Cyan)")
axes[0].axis('off')

# Prepare lines for the initial projection (using proj_vertices from Cell 1)
initial_lines = []
for tri in triangles:
    pts = [proj_vertices[i] for i in tri]
    pts.append(proj_vertices[tri[0]]) 
    initial_lines.append(pts)
    
# Plot initial triangles at once
lc_initial = LineCollection(initial_lines, colors='cyan', linewidths=0.5, alpha=0.1)
axes[0].add_collection(lc_initial)

# ==========================================
# Subplot 2: Optimized Silhouette Alignment
# ==========================================
axes[1].imshow(image_rgb)
axes[1].set_title("Optimized Silhouette Alignment (Magenta)")
axes[1].axis('off')

# Prepare lines for the optimized projection
opt_lines = []
for tri in triangles:
    pts = [opt_proj_vertices[i] for i in tri]
    pts.append(opt_proj_vertices[tri[0]]) 
    opt_lines.append(pts)

# Plot optimized triangles at once
lc_opt = LineCollection(opt_lines, colors='magenta', linewidths=0.5, alpha=0.1)
axes[1].add_collection(lc_opt)

plt.tight_layout()

# Save the figure using Matplotlib instead of cv2.imwrite
plt.savefig("mesh_overlay_optimized_comparison.jpg", dpi=300, bbox_inches='tight')
plt.show()

# Optional: Save the new pose to JSON for future use
optimized_pose = {
    "rvec": opt_rotation.tolist(),
    "tvec": opt_translation.tolist()
}
with open('optimized_pose.json', 'w') as f:
    json.dump(optimized_pose, f)