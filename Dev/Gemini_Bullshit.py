#%% Cell 1: Fiducial Image Mesh Picking and matching
import scipy
import json
import os
import numpy as np
import cv2
import open3d as o3d
from glob import glob
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg
from vedo import Mesh, Plotter, Sphere, Text3D
import multiprocessing as mp
from Dev.pickers import run_2d_picker, run_3d_picker

def pick_points_simultaneous(photo_path, mesh_path):
    q_2d = mp.Queue()
    q_3d = mp.Queue()

    p2 = mp.Process(target=run_2d_picker,
                    args=(photo_path, q_2d, "Pick 2D fiducials"))
    p3 = mp.Process(target=run_3d_picker,
                    args=(mesh_path,  q_3d,
                          "Pick 3D fiducials — LEFT-CLICK add | RIGHT-CLICK delete last"))
    p2.start(); p3.start()
    p2.join();  p3.join()
    pts_2d = []
    while not q_2d.empty():
        pts_2d.append(q_2d.get())

    pts_3d = []
    while not q_3d.empty():
        pts_3d.append(q_3d.get())
    return pts_2d, pts_3d


def replace_point_pair(idx, pts_2d, pts_3d, photo_path, mesh_path):
    q_2d = mp.Queue()
    q_3d = mp.Queue()

    p2 = mp.Process(target=run_2d_picker,
                    args=(photo_path, q_2d, '', pts_2d, idx, True))
    p3 = mp.Process(target=run_3d_picker,
                    args=(mesh_path,  q_3d, '', pts_3d, idx, True))
    p2.start(); p3.start()
    p2.join();  p3.join()

    if not q_2d.empty(): pts_2d[idx] = q_2d.get()
    if not q_3d.empty(): pts_3d[idx] = q_3d.get()
    return pts_2d, pts_3d


def visualize_alignment(image_path, rvec, tvec, pts_2d, pts_3d):
    image     = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    proj_verts, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeff)
    proj_verts    = proj_verts.squeeze().astype(int)

    proj_sel, _ = cv2.projectPoints(
        np.array(pts_3d, dtype=np.float32), rvec, tvec, camera_matrix, dist_coeff)
    proj_sel = proj_sel.squeeze().astype(int)
    # handle single-point edge case
    if proj_sel.ndim == 1:
        proj_sel = proj_sel[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image_rgb)

    lc = LineCollection(
        [[proj_verts[i] for i in tri] + [proj_verts[tri[0]]] for tri in triangles],
        colors='cyan', linewidths=0.5, alpha=0.15)
    ax.add_collection(lc)

    for i, (p2d, p3d) in enumerate(zip(pts_2d, proj_sel)):
        ax.plot([p2d[0], p3d[0]], [p2d[1], p3d[1]], color='yellow', linewidth=1.5)
        ax.scatter(*p2d, color='red',  s=50, zorder=5, label='Picked 2D'   if i == 0 else '')
        ax.scatter(*p3d, color='lime', s=50, zorder=5, label='Projected 3D' if i == 0 else '')
        ax.text(p2d[0] + 5, p2d[1] - 5, str(i + 1), color='red',  fontsize=9)
        ax.text(p3d[0] + 5, p3d[1] - 5, str(i + 1), color='lime', fontsize=9)

    ax.legend(loc='upper right')
    ax.axis('off')
    plt.title("PnP Alignment — Red = picked 2D  |  Lime = projected 3D\nClose window to continue")
    plt.tight_layout()
    plt.show(block=True)

# ==========================================
# 4. MAIN INTERACTIVE LOOP
# ==========================================
if __name__ == '__main__':

    mesh_file = 'meshes/1613_LDMesh_260109.stl'
    calibration_file = 'Logi_calibration.json'
    photo_dir = '1613/S9a/'
    camera_configs = [
        {'name': 'Camera_1', 'json_out': 'points_C1_1613_S10_1.json'},
        {'name': 'Camera_2', 'json_out': 'points_C2_1613_S10_1.json'},
        {'name': 'Camera_3', 'json_out': 'points_C3_1613_S10_1.json'},
        {'name': 'Camera_4', 'json_out': 'points_C4_1613_S10_1.json'},
    ]
    for config in camera_configs:
    # Get just the number from "Camera_1" -> "1"
        cam_num = config['name'].split('_')[1] 
        
        # Create a search pattern, e.g., "1613/S10/Camera1_*.jpg"
        search_pattern = os.path.join(photo_dir, f"Camera{cam_num}_*.jpg")
        
        # Find all matching files in that directory
        matching_files = glob(search_pattern)
        
        if matching_files:
            # Take the first match found, and get just the filename (not the full path)
            config['photo_path'] = os.path.basename(matching_files[0])
        else:
            # Fallback if no photo is found in the folder
            config['photo_path'] = None
            print(f"Warning: No photo found for Camera {cam_num} in {photo_dir}")
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    camera_matrix = np.array(calib_data['camera_matrix'])
    dist_coeff    = np.array(calib_data['dist_coeff'])

    o3d_mesh = o3d.io.read_triangle_mesh(mesh_file)
    o3d_mesh.compute_vertex_normals()
    vertices  = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)

    mp.set_start_method('spawn', force=True)

    for cam in camera_configs:
        print(f"\n{'='*40}")
        print(f"STARTING ALIGNMENT FOR: {cam['name']}")
        print(f"{'='*40}")

        if os.path.exists(cam['json_out']):
            print(f"Found existing data for {cam['name']}. Skipping.")
            continue

        # ── PHASE 1: simultaneous picking ─────────────────────────────
        print("\n--- STEP 1: PICK ALIGNMENT FIDUCIALS ---")
        print("Both windows open simultaneously.")
        print("Left-click = add point   |   Right-click = delete last point")
        print("Close BOTH windows when you are done picking.\n")

        pts_2d, pts_3d = pick_points_simultaneous(cam['photo_path'], mesh_file)

        if len(pts_2d) != len(pts_3d):
            print(f"ERROR: {len(pts_2d)} 2D vs {len(pts_3d)} 3D — counts must match. Skipping.")
            continue
        if len(pts_2d) < 4:
            print("ERROR: need at least 4 point pairs for solvePnP. Skipping.")
            continue

        print(f"\nCollected {len(pts_2d)} point pairs.")

        print("\nName your fiducials (Enter = use default):")
        fiducial_names = []
        for i in range(len(pts_2d)):
            name = input(f"  Fiducial {i+1} [Point_{i+1}]: ").strip()
            fiducial_names.append(name or f"Point_{i+1}")

        # ── PHASE 2: refinement loop ───────────────────────────────────
        while True:
            pts_3d_np = np.array(pts_3d, dtype=np.float32)
            pts_2d_np = np.array(pts_2d, dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                pts_3d_np, pts_2d_np, camera_matrix, dist_coeff,
                flags=cv2.SOLVEPNP_SQPNP)

            if not success:
                print("solvePnP failed — check your correspondences.")
                break

            visualize_alignment(cam['photo_path'], rvec, tvec, pts_2d, pts_3d)

            print("\nOptions:")
            print("  Enter point number(s) to re-pick  e.g.  2   or   1,3")
            print("  Press Enter to ACCEPT and save")
            choice = input("Your choice: ").strip()

            if not choice:
                print("Alignment accepted.")
                break

            try:
                indices = [int(x.strip()) - 1 for x in choice.replace(',', ' ').split()]
            except ValueError:
                print("  Couldn't parse that — try again.")
                continue

            invalid = [i for i in indices if not (0 <= i < len(pts_2d))]
            if invalid:
                print(f"  Invalid numbers: {[i+1 for i in invalid]}")
                continue

            for idx in indices:
                pts_2d, pts_3d = replace_point_pair(
                    idx, pts_2d, pts_3d, cam['photo_path'], mesh_file)
                new_name = input(f"  New name for point {idx+1} [{fiducial_names[idx]}]: ").strip()
                if new_name:
                    fiducial_names[idx] = new_name

        # ── Save ───────────────────────────────────────────────────────
        output = {
            'camera':          cam['name'],
            'rvec':            rvec.tolist(),
            'tvec':            tvec.tolist(),
            'pts_2d':          pts_2d,
            'pts_3d':          pts_3d,
            'fiducial_names':  fiducial_names,
        }
        with open(cam['json_out'], 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved → {cam['json_out']}")
#%% Cell 2: Match optode locations between cameras
import json
import os
import numpy as np
import cv2
import multiprocessing as mp
import matplotlib

# ==============================================================================
# 1. SEQUENTIAL 2D PAIR PICKER
# ==============================================================================

def _run_pure_2d_picker(photo_path, result_list, window_title, current_point_idx):
    """
    Runs in its own process. Allows picking a single 2D coordinate 
    corresponding to the current active point index.
    """
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(photo_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(f"{window_title}\nClick matching point for Point #{current_point_idx} | Close when done")

    clicked_pt = []

    def on_click(event):
        if not event.inaxes or event.button != 1:
            return
        clicked_pt.append([event.xdata, event.ydata])
        ax.plot(event.xdata, event.ydata, 'go', markersize=8)
        ax.text(event.xdata + 5, event.ydata - 5, f"#{current_point_idx}", 
                color='lime', fontsize=12, fontweight='bold')
        fig.canvas.draw()
        print(f"  [{window_title}] Picked Point #{current_point_idx}: ({event.xdata:.1f}, {event.ydata:.1f})")
        plt.close(fig) # Advance to the next point pair automatically

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show(block=True)

    if clicked_pt:
        result_list.append(clicked_pt[0])


def match_points_between_images(cam_a, cam_b):
    """
    Spawns windows side-by-side to collect fresh 2D matching points.
    Loop breaks as soon as a window is closed without making a selection.
    """
    pts_a, pts_b = [], []
    point_counter = 1
    
    print(f"\n--- STARTING NEW POINT SELECTION: {cam_a['name']} <-> {cam_b['name']} ---")
    print("Click the corresponding features in both images sequentially.")
    print("To FINISH and process, simply close either window without clicking.")

    while True:
        with mp.Manager() as manager:
            shared_pt_a = manager.list()
            shared_pt_b = manager.list()

            p_a = mp.Process(target=_run_pure_2d_picker, 
                             args=(cam_a['photo_path'], shared_pt_a, cam_a['name'], point_counter))
            p_b = mp.Process(target=_run_pure_2d_picker, 
                             args=(cam_b['photo_path'], shared_pt_b, cam_b['name'], point_counter))
            
            p_a.start()
            p_b.start()
            p_a.join()
            p_b.join()

            # Stop loop if user closes a window without picking a point
            if len(shared_pt_a) == 0 or len(shared_pt_b) == 0:
                print(f"Selection finalized for {cam_a['name']} & {cam_b['name']}.")
                break

            pts_a.append(list(shared_pt_a)[0])
            pts_b.append(list(shared_pt_b)[0])
            point_counter += 1

    return np.array(pts_a, dtype=np.float32), np.array(pts_b, dtype=np.float32)


# ==============================================================================
# 2. MAIN ORCHESTRATION & SAVING LOOP
# ==============================================================================

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # Configuration mapped directly from your setup
    camera_configs = [
        {'name': 'Camera_1', 'photo_path': 'Camera1_2023-01-15_10-14-42.jpg'},
        {'name': 'Camera_2', 'photo_path': 'Camera2_2023-01-15_10-14-46.jpg'},
        {'name': 'Camera_3', 'photo_path': 'Camera3_2023-01-15_10-14-43.jpg'},
        {'name': 'Camera_4', 'photo_path': 'Camera4_2023-01-15_10-14-44.jpg'},
    ]

    calibration_file = 'Logi_calibration.json'
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
    dist_coeff    = np.array(calib_data['dist_coeff'], dtype=np.float32)

    # Explicit pair associations requested
    pairs_to_process = [
        {'cam_a': camera_configs[0], 'cam_b': camera_configs[1], 'json_out': 'matched_C1_C2.json'},
        {'cam_a': camera_configs[2], 'cam_b': camera_configs[3], 'json_out': 'matched_C3_C4.json'}
    ]

    for pair in pairs_to_process:
        cam_a = pair['cam_a']
        cam_b = pair['cam_b']
        output_file = pair['json_out']

        # Skip if you don't want to accidentally overwrite a freshly made set
        if os.path.exists(output_file):
            print(f"\nExisting file {output_file} found. Skipping to avoid overwrite.")
            continue

        # 1. Collect points interactively
        pts_a, pts_b = match_points_between_images(cam_a, cam_b)

        # Essential Matrix requires a mathematical minimum of 5 points (7-8+ strongly advised)
        if len(pts_a) < 5:
            print(f"ERROR: Found {len(pts_a)} points. At least 5 pairs are required for Essential Matrix computation.")
            continue

        # 2. Mathematical normalization (Undistortion step)
        pts_a_undist = cv2.undistortImagePoints(pts_a[:, np.newaxis, :], camera_matrix, dist_coeff).squeeze()
        pts_b_undist = cv2.undistortImagePoints(pts_b[:, np.newaxis, :], camera_matrix, dist_coeff).squeeze()

        # 3. Calculate Essential Matrix
        E, mask = cv2.findEssentialMat(
            points1=pts_a_undist,
            points2=pts_b_undist,
            cameraMatrix=camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        # 4. Package and Save Data structure
        output_data = {
            'camera_pair': f"{cam_a['name']}_{cam_b['name']}",
            'points_count': len(pts_a),
            'ransac_inliers': int(np.sum(mask)),
            'pts_camera_a': pts_a.tolist(),
            'pts_camera_b': pts_b.tolist(),
            'essential_matrix': E.tolist() if E is not None else None
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"SUCCESS: Saved new points data structure and Essential Matrix to → {output_file}\n")
#%% Cell 3 PnP
import os
import json
import numpy as np
import cv2
import open3d as o3d
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from Dev.pickers import (
    visualize_initial_scene_with_bounds, 
    visualize_scene_geometry, 
    visualize_mesh_error_vectors, 
    report_alignment_errors, 
    build_fixture_bounds
)

def triangulate_match_group(file_path, cam_idx_a, cam_idx_b):
    if not os.path.exists(file_path):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    with open(file_path, 'r') as f:
        m_data = json.load(f)
    pts_a = np.array(m_data['pts_camera_a'], dtype=np.float64)
    pts_b = np.array(m_data['pts_camera_b'], dtype=np.float64)
    
    pts_a_u = cv2.undistortImagePoints(pts_a[:, np.newaxis, :], camera_matrix, dist_coeff).squeeze()
    pts_b_u = cv2.undistortImagePoints(pts_b[:, np.newaxis, :], camera_matrix, dist_coeff).squeeze()
    
    R_a, _ = cv2.Rodrigues(camera_poses[cam_idx_a]['rvec'])
    R_b, _ = cv2.Rodrigues(camera_poses[cam_idx_b]['rvec'])
    
    P_a = camera_matrix @ np.hstack((R_a, camera_poses[cam_idx_a]['tvec'].reshape(3,1)))
    P_b = camera_matrix @ np.hstack((R_b, camera_poses[cam_idx_b]['tvec'].reshape(3,1)))
    
    pts4D = cv2.triangulatePoints(P_a, P_b, pts_a_u.T, pts_b_u.T)
    pts3D = (pts4D[:3, :] / pts4D[3, :]).T
    return pts3D.astype(np.float64), pts_a, pts_b

def pack_state_vector(poses, fids_3d, m12_3d, m34_3d):
    components = []
    for i in range(N_cams):
        components.extend(poses[i]['rvec'].flatten())
        components.extend(poses[i]['tvec'].flatten())
    components.extend(fids_3d.flatten())
    components.extend(m12_3d.flatten())
    components.extend(m34_3d.flatten())
    return np.array(components, dtype=np.float64)

def unpack_state_vector(vector):
    poses = {}
    idx = 0
    for i in range(N_cams):
        poses[i] = {'rvec': vector[idx:idx+3], 'tvec': vector[idx+3:idx+6]}
        idx += 6

    fids_3d = vector[idx:idx + N_fids*3].reshape((N_fids, 3))
    idx += N_fids*3
    m12_3d = vector[idx:idx + N_m12*3].reshape((N_m12, 3))
    idx += N_m12*3
    m34_3d = vector[idx:idx + N_m34*3].reshape((N_m34, 3))
    return poses, fids_3d, m12_3d, m34_3d

def optimization_residuals(vector, lambda_mesh_weight=1.5):
    poses, fids_3d, m12_3d, m34_3d = unpack_state_vector(vector)
    residuals = []

    for obs in fiducial_observations:
        cam = poses[obs['cam_idx']]
        pt3d = fids_3d[obs['point_id']].reshape(1, 3)
        uv_proj, _ = cv2.projectPoints(pt3d, cam['rvec'], cam['tvec'], camera_matrix, dist_coeff)
        residuals.extend((obs['uv'] - uv_proj.squeeze()).flatten())
        
    for i in range(N_m12):
        pt3d = m12_3d[i].reshape(1, 3)
        uv_p1, _ = cv2.projectPoints(pt3d, poses[0]['rvec'], poses[0]['tvec'], camera_matrix, dist_coeff)
        residuals.extend((match_uv_1[i] - uv_p1.squeeze()).flatten())
        uv_p2, _ = cv2.projectPoints(pt3d, poses[1]['rvec'], poses[1]['tvec'], camera_matrix, dist_coeff)
        residuals.extend((match_uv_2[i] - uv_p2.squeeze()).flatten())

    for i in range(N_m34):
        pt3d = m34_3d[i].reshape(1, 3)
        uv_p3, _ = cv2.projectPoints(pt3d, poses[2]['rvec'], poses[2]['tvec'], camera_matrix, dist_coeff)
        residuals.extend((match_uv_3[i] - uv_p3.squeeze()).flatten())
        uv_p4, _ = cv2.projectPoints(pt3d, poses[3]['rvec'], poses[3]['tvec'], camera_matrix, dist_coeff)
        residuals.extend((match_uv_4[i] - uv_p4.squeeze()).flatten())

    if N_fids > 0:
        query_points = o3d.core.Tensor(fids_3d, dtype=o3d.core.float32)
        closest_lookup = scene.compute_closest_points(query_points)
        closest_mesh_pts = closest_lookup['points'].numpy().astype(np.float64)
        mesh_displacements = fids_3d - closest_mesh_pts
        residuals.extend((lambda_mesh_weight * mesh_displacements).flatten())

    return np.array(residuals, dtype=np.float64)


if __name__ == '__main__':
    # ==============================================================================
    # 1. SETUP LOGISTICS AND CONVENTIONS
    # ==============================================================================
    calibration_file = 'Logi_calibration.json'
    mesh_file        = 'meshes/1613_LDMesh_260109.stl'

    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float64)
    dist_coeff    = np.array(calib_data['dist_coeff'], dtype=np.float64)

    print("Loading low-density STL mesh topology...")
    o3d_mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh_t   = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    scene    = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    bbox = o3d_mesh.get_axis_aligned_bounding_box()
    print(f"Mesh bounds: min={np.round(bbox.get_min_bound(),1)}  max={np.round(bbox.get_max_bound(),1)}")
    print(f"Mesh center: {np.round(bbox.get_center(),1)}")
    print(f"Mesh extent: {np.round(bbox.get_extent(),1)}")
    POS_TOLERANCE_X = 20.0 
    POS_TOLERANCE_Y = 10.0  
    POS_TOLERANCE_Z = 60.0 
    ROT_TOLERANCE_X = 0.17
    ROT_TOLERANCE_Y = 0.17
    ROT_TOLERANCE_Z = 0.17
    
    global nominal_fixture_poses
    nominal_fixture_poses = {
        1: (np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64), np.array([0.0, -600.0, 0.0])),
        2: (np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64), np.array([-1400.0, -600.0, 0.0])),  
        3: (np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64), np.array([-1400.0, 0.0, 0.0]))
    }
    
    # ==============================================================================
    # 2. DATA INGESTION & COGNITIVE TRIANGULATION
    # ==============================================================================
    camera_names = ['Camera_1', 'Camera_2', 'Camera_3', 'Camera_4']
    camera_poses = {} 
    fiducial_observations = [] 

    unique_fiducials = set()
    fiducial_data_per_cam = {}

    for idx, name in enumerate(camera_names):
        file_path = f'points_{name.replace("amera_", "")}.json' 
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            camera_poses[idx] = {'rvec': np.array(data['rvec'], dtype=np.float64), 'tvec': np.array(data['tvec'], dtype=np.float64)}
            fiducial_data_per_cam[idx] = data
            unique_fiducials.update(data['fiducial_names'])

    fiducial_list = sorted(list(unique_fiducials))
    fid_name_to_idx = {name: i for i, name in enumerate(fiducial_list)}

    for idx, data in fiducial_data_per_cam.items():
        for uv, f_name in zip(data['pts_2d'], data['fiducial_names']):
            fiducial_observations.append({
                'cam_idx': idx,
                'point_id': fid_name_to_idx[f_name],
                'uv': np.array(uv, dtype=np.float64)
            })

    init_fiducial_3d = np.zeros((len(fiducial_list), 3), dtype=np.float64)
    for idx, data in fiducial_data_per_cam.items():
        for f_name, pt3d in zip(data['fiducial_names'], data['pts_3d']):
            init_fiducial_3d[fid_name_to_idx[f_name]] = np.array(pt3d, dtype=np.float64)

    print("Triangulating structural pair matches...")
    pts3d_12, match_uv_1, match_uv_2 = triangulate_match_group('matched_C1_C2.json', 0, 1)
    pts3d_34, match_uv_3, match_uv_4 = triangulate_match_group('matched_C3_C4.json', 2, 3)
    cam_obs_counts = Counter(obs['cam_idx'] for obs in fiducial_observations)
    for i, name in enumerate(camera_names):
        n_matches = len(pts3d_12) if i < 2 else len(pts3d_34)
        print(f"  {name}: {cam_obs_counts.get(i, 0)} fiducial observations, {n_matches} match points")

    # ==============================================================================
    # 3. GLOBAL BUNDLE ADJUSTMENT ENGINE
    # ==============================================================================
    N_cams = 4
    N_fids = len(fiducial_list)
    N_m12  = len(pts3d_12)
    N_m34  = len(pts3d_34)

    x0 = pack_state_vector(camera_poses, init_fiducial_3d, pts3d_12, pts3d_34)
    
    # Render scene structure diagnostic check
    visualize_scene_geometry(camera_poses, o3d_mesh, init_fiducial_3d)

    low_b, up_b = build_fixture_bounds(
        camera_poses,
        init_fiducial_3d,
        pts3d_12,
        pts3d_34,
        pos_tolerances=(POS_TOLERANCE_X, POS_TOLERANCE_Y, POS_TOLERANCE_Z),
        rot_tolerances=(ROT_TOLERANCE_X, ROT_TOLERANCE_Y, ROT_TOLERANCE_Z),
    )
        # Safe clip operation matching bounds setup layout 
    violations = np.where((x0 < low_b) | (x0 > up_b))[0]
    print(f"Out-of-bounds indices: {violations}")
    for i in violations:
        print(f"  idx {i}: x0={x0[i]:.6f}  low={low_b[i]:.6f}  up={up_b[i]:.6f}")
    initial_vector = np.clip(x0, low_b + 1e-5, up_b - 1e-5)
    print(f"Starting Joint Bundle Adjustment Optimization over {len(x0)} independent parameters...")
   
    res = least_squares(
        optimization_residuals, 
        x0, 
        bounds=(low_b, up_b),
        method='trf', 
        x_scale='jac',
        verbose=2
    )   
    
    opt_poses, opt_fids, opt_m12, opt_m34 = unpack_state_vector(res.x)
    report_alignment_errors(
        opt_poses, opt_fids, opt_m12, opt_m34,
        N_cams, N_fids, N_m12, N_m34,
        fiducial_observations,
        match_uv_1, match_uv_2, match_uv_3, match_uv_4,
        camera_matrix, dist_coeff, scene
    )
    # Step 1: Per-fiducial reprojection errors
    print("\n--- PER-FIDUCIAL REPROJECTION ERRORS ---")
    for obs in fiducial_observations:
        cam = opt_poses[obs['cam_idx']]
        pt3d = opt_fids[obs['point_id']].reshape(1, 3)
        uv_proj, _ = cv2.projectPoints(pt3d, cam['rvec'], cam['tvec'], camera_matrix, dist_coeff)
        err = np.linalg.norm(obs['uv'] - uv_proj.squeeze())
        fid_name = fiducial_list[obs['point_id']]
        print(f"  Camera_{obs['cam_idx']+1} / {fid_name}: {err:.1f} px")

    print(f"\nFiducial 3D positions:")
    for name, pt in zip(fiducial_list, opt_fids):
        print(f"  {name}: {np.round(pt,1)}")
    visualize_scene_geometry(opt_poses, o3d_mesh)
    visualize_mesh_error_vectors(opt_fids, scene)
    
    initial_res = optimization_residuals(x0)
    final_res = res.fun
    print("\n" + "="*50)
    print(f"Initial Total Residual Norm: {np.linalg.norm(initial_res):.2f}")
    print(f"Optimized Total Residual Norm: {np.linalg.norm(final_res):.2f}")
    print("="*50)

    with open('matched_C1_C2.json') as f:
        m = json.load(f)
    print(f"C1-C2: {m['ransac_inliers']}/{m['points_count']} inliers ({100*m['ransac_inliers']/m['points_count']:.0f}%)")

    with open('matched_C3_C4.json') as f:
        m = json.load(f)
    print(f"C3-C4: {m['ransac_inliers']}/{m['points_count']} inliers ({100*m['ransac_inliers']/m['points_count']:.0f}%)")
    
    # Check where your face-anchoring fiducials sit relative to mesh extent
    mesh_z_max = 70.3   # from your bounds output
    mesh_z_min = -122.8

    face_landmarks = ['nasion', 'brow', 'tip_nose', 'corner_eye_L', 'corner_eye_R']
    print("Face landmark Z positions (mesh Z range: {:.1f} to {:.1f})".format(mesh_z_min, mesh_z_max))
    for name, pt in zip(fiducial_list, opt_fids):
        if name in face_landmarks:
            z_pct = (pt[2] - mesh_z_min) / (mesh_z_max - mesh_z_min) * 100
            print(f"  {name}: Z={pt[2]:.1f}  ({z_pct:.0f}% up from bottom)")
    optimized_output = {}
    for i, name in enumerate(camera_names):
        optimized_output[name] = {
            'rvec': opt_poses[i]['rvec'].tolist(),
            'tvec': opt_poses[i]['tvec'].tolist()
        }
    with open('all_optimized_poses.json', 'w') as f:
        json.dump(optimized_output, f, indent=4)
    print("Saved optimized camera geometries globally → all_optimized_poses.json")

    all_optode_clicks = {}
    for idx, data in fiducial_data_per_cam.items():
        cam_name = camera_names[idx]
        all_optode_clicks[cam_name] = {
            name: pixel for pixel, name in zip(data['pts_2d'], data['fiducial_names'])
        }

    with open('all_optode_clicks.json', 'w') as f:
        json.dump(all_optode_clicks, f, indent=4)
    print("Saved → all_optode_clicks.json")
# %% --- CELL 4: MULTI-CAMERA OPTODE PROJECTION & VALIDATION ---
import open3d as o3d
import numpy as np
import cv2
import json
import itertools
from scipy.optimize import minimize
# ==========================================
# 1. RAYCASTING HELPER
# ==========================================
def cast_ray(pixel_uv, rvec, tvec):
    """
    Cast a ray from the camera centre through pixel_uv.
    Returns the 3-D mesh intersection point, or None on a miss.
    """
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
    t    = np.array(tvec, dtype=np.float32).reshape(3, 1)

    cam_centre  = (-R.T @ t).flatten()
    ray_dir     = (R.T @ (np.linalg.inv(camera_matrix.astype(np.float32))
                          @ np.array([pixel_uv[0], pixel_uv[1], 1.0])
                          ).reshape(3, 1)).flatten()
    ray_dir    /= np.linalg.norm(ray_dir)

    ans   = scene.cast_rays(o3d.core.Tensor(
                [np.hstack([cam_centre, ray_dir])], dtype=o3d.core.float32))
    t_hit = ans['t_hit'].numpy()[0]

    return None if np.isinf(t_hit) else cam_centre + t_hit * ray_dir


# ==========================================
# 2. POSE REFINEMENT USING SEQUENTIAL GRAPH TRAVERSAL
# ==========================================

def reprojection_loss(params, pts_3d_ref, pts_2d_src, cam_matrix, dist_coeff):
    """
    Objective: given a candidate (rvec, tvec) for the source camera,
    project pts_3d_ref and measure pixel distance to pts_2d_src.
    """
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:].reshape(3, 1)
    proj, _ = cv2.projectPoints(pts_3d_ref.astype(np.float32),
                                rvec, tvec, cam_matrix, dist_coeff)
    proj = proj.squeeze()
    # Handle single point array shaping edge case
    if proj.ndim == 1:
        proj = proj.reshape(1, 2)
    return np.mean(np.linalg.norm(proj - pts_2d_src, axis=1))


def refine_pose(src_cam, ref_cam, shared_optodes, current_poses):
    """
    Refine src_cam's pose using ref_cam's CURRENT (potentially already refined) pose.
    Lowered minimum constraint to 2 points to allow cross-face bridging.
    """
    if len(shared_optodes) < 2:
        print(f"  [refine] Only {len(shared_optodes)} shared points between "
              f"{src_cam} and {ref_cam} — skipping (need ≥2)")
        return None

    # CRITICAL: Use the active tracking dictionary, not the raw file data
    ref_pose   = current_poses[ref_cam]
    ref_clicks = multi_cam_clicks[ref_cam]

    # Get 3-D points for each shared optode using the REFERENCE camera's active pose
    pts_3d, pts_2d = [], []
    for name in shared_optodes:
        p3d = cast_ray(ref_clicks[name], ref_pose['rvec'], ref_pose['tvec'])
        if p3d is None:
            continue
        pts_3d.append(p3d)
        pts_2d.append(multi_cam_clicks[src_cam][name])

    if len(pts_3d) < 2:
        print(f"  [refine] Too many ray misses for {src_cam}/{ref_cam} pair.")
        return None

    pts_3d = np.array(pts_3d, dtype=np.float32)
    pts_2d = np.array(pts_2d, dtype=np.float32)

    src_pose    = current_poses[src_cam]
    init_rvec   = np.array(src_pose['rvec'], dtype=np.float32).flatten()
    init_tvec   = np.array(src_pose['tvec'], dtype=np.float32).flatten()
    init_params = np.hstack([init_rvec, init_tvec])

    loss_before = reprojection_loss(init_params, pts_3d, pts_2d, camera_matrix, dist_coeff)

    res = minimize(reprojection_loss, init_params,
                   args=(pts_3d, pts_2d, camera_matrix, dist_coeff),
                   method='Powell',
                   options={'disp': False, 'xtol': 1e-5, 'ftol': 1e-5})

    loss_after = res.fun
    print(f"  [refine] {src_cam} vs {ref_cam} ({len(pts_3d)} pts): "
          f"reprojection {loss_before:.2f} → {loss_after:.2f} px "
          f"({'improved' if loss_after < loss_before else 'no improvement'})")

    if loss_after >= loss_before:
        return None   # Don't save adjustments if they degrade the fit

    return res.x[:3].reshape(3, 1), res.x[3:].reshape(3, 1)

if __name__ == '__main__':
    # ==========================================
    # CONFIGURATION
    # ==========================================
    OUTLIER_THRESHOLD_MM  = 15.0   # pairs further apart than this are flagged
    CONSENSUS_THRESHOLD_MM = 10.0  # a camera is "in consensus" if its point is within
                                    # this distance of the group median

    # ==========================================
    # 0. LOAD SHARED DATA
    # (assumes mesh, camera_matrix, dist_coeff already in scope from Cell 1)
    # ==========================================
    with open('all_optimized_poses.json', 'r') as f:
        camera_poses = json.load(f)

    with open('all_optode_clicks.json', 'r') as f:
        multi_cam_clicks = json.load(f)

    mesh_legacy = o3d.io.read_triangle_mesh(mesh_file)
    mesh_legacy.compute_vertex_normals()

    # 2. Now convert the legacy object to the new tensor-based mesh
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

    scene  = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    print("Raycasting scene ready.")

    # --- Find shared optodes between every camera pair ---
    cam_names = list(multi_cam_clicks.keys())
    shared_map = {}   # { (cam_a, cam_b): [optode_names] }

    for ca, cb in itertools.combinations(cam_names, 2):
        shared = sorted(set(multi_cam_clicks[ca]) & set(multi_cam_clicks[cb]))
        if shared:
            shared_map[(ca, cb)] = shared

    # --- Initialize tracking dictionaries ---
    REFERENCE_CAM = "Camera_4"   # Anchor point
    refined_poses = {cam: {'rvec': np.array(data['rvec'], dtype=np.float32),
                        'tvec': np.array(data['tvec'], dtype=np.float32)}
                    for cam, data in camera_poses.items()}

    refined_set   = [REFERENCE_CAM]
    unrefined_set = [c for c in cam_names if c != REFERENCE_CAM]

    print(f"\nStarting sequential graph refinement from anchor: {REFERENCE_CAM}")

    # Step through the network, resolving cameras with the strongest remaining connections first
    while unrefined_set:
        best_src = None
        best_ref = None
        best_shared = []
        max_shared_count = 0

        # Look for the unrefined camera holding the most connections to the already stabilized network
        for src_cam in unrefined_set:
            for ref_cam in refined_set:
                key = tuple(sorted([src_cam, ref_cam]))
                shared = shared_map.get(key, [])
                if len(shared) > max_shared_count:
                    max_shared_count = len(shared)
                    best_src = src_cam
                    best_ref = ref_cam
                    best_shared = shared

        # If the network breaks down below 2 points, we can't mathematically calculate adjustments
        if max_shared_count < 2:
            print(f"\n⚠️ Graph disconnected! Remaining cameras {unrefined_set} lack sufficient overlapping points.")
            break

        print(f"\nNext Queue Target: {best_src} (via {best_ref} with {max_shared_count} shared landmarks)")
        result = refine_pose(best_src, best_ref, best_shared, refined_poses)
        
        if result is not None:
            refined_poses[best_src]['rvec'], refined_poses[src_cam]['tvec'] = result
            print(f"  → Successfully updated pose for {best_src}")
        else:
            print(f"  → Retaining original baseline pose for {best_src}")

        # Move target camera to the stabilized set so it can act as a bridge for remaining cameras
        refined_set.append(best_src)
        unrefined_set.remove(best_src)

# %% --- CELL 5: VISUAL SANITY CHECK FOR 2D LABELS (UPDATED) ---
import json
import cv2
import matplotlib.pyplot as plt
import os
if __name__ == '__main__':
    camera_configs = [
        {'name': 'Camera_1', 'photo_path': 'Camera1_2023-01-15_10-14-42.jpg', 'json_out': 'points_C1.json'},
        {'name': 'Camera_2', 'photo_path': 'Camera2_2023-01-15_10-14-46.jpg', 'json_out': 'points_C2.json'},
        {'name': 'Camera_3', 'photo_path': 'Camera3_2023-01-15_10-14-43.jpg', 'json_out': 'points_C3.json'},
        {'name': 'Camera_4', 'photo_path': 'Camera4_2023-01-15_10-14-44.jpg', 'json_out': 'points_C4.json'},
    ]
    # Create a quick mapping lookup: {'Camera_1': 'Camera1_2023-01-15_10-14-42.jpg', ...}
    photo_lookup = {cam['name']: cam['photo_path'] for cam in camera_configs}

    clicks_file = 'all_optode_clicks.json'

    if not os.path.exists(clicks_file):
        print(f"❌ Error: {clicks_file} not found. Please run your pipeline first.")
    else:
        with open(clicks_file, 'r') as f:
            multi_cam_clicks = json.load(f)

        print("--- Displaying 2D Clicks with Explicit Filenames ---")
        
        for cam_name, clicks in multi_cam_clicks.items():
            # Look up the specific filename from your configuration array
            img_filename = photo_lookup.get(cam_name)
            if not img_filename:
                print(f"⚠️ Skipping {cam_name}: No photo_path mapping found in camera_configs.")
                continue
                
            img_path = img_filename
            
            if not os.path.exists(img_path):
                print(f"⚠️ Skipping {cam_name}: Image file not found at '{img_path}'")
                continue
                
            # Load and prep image for Matplotlib display
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 9))
            plt.imshow(img)
            
            # Overlay the labeled coordinates
            for label, pixel_uv in clicks.items():
                u, v = pixel_uv
                
                # Plot a point marker
                plt.scatter(u, v, color='cyan', marker='o', s=40, edgecolors='black', zorder=3)
                
                # Draw the text overlay box
                plt.text(u + 5, v - 5, label, color='white', fontsize=10, weight='bold',
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='cyan', boxstyle='round,pad=0.3'),
                        zorder=4)
                        
            plt.title(f"Reviewing: {img_filename} ({cam_name})", fontsize=14, weight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

# %% --- CELL 6: POST-ADJUSTMENT ALIGNMENT CHECK ---
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
if __name__ == '__main__':
    # ==========================================
    # CONFIGURATION — Matches your setups
    # ==========================================
    camera_configs = [
        {'name': 'Camera_1', 'photo_path': 'Camera1_2023-01-15_10-14-42.jpg'},
        {'name': 'Camera_2', 'photo_path': 'Camera2_2023-01-15_10-14-46.jpg'},
        {'name': 'Camera_3', 'photo_path': 'Camera3_2023-01-15_10-14-43.jpg'},
        {'name': 'Camera_4', 'photo_path': 'Camera4_2023-01-15_10-14-44.jpg'},
    ]

    camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)  # Assumes calib_data is in scope
    dist_coeff    = np.array(calib_data['dist_coeff'], dtype=np.float32)

    # Load data computed from the alignment cell
    with open('all_refined_poses.json', 'r') as f:
        refined_poses = json.load(f)

    with open('final_3d_optodes.json', 'r') as f:
        final_3d_optodes = json.load(f)

    with open('all_optode_clicks.json', 'r') as f:
        multi_cam_clicks = json.load(f)

    photo_lookup = {cam['name']: cam['photo_path'] for cam in camera_configs}

    print("--- Plotting Post-Refinement Alignments ---")

    for cam_name, clicks in multi_cam_clicks.items():
        img_filename = photo_lookup.get(cam_name)
        img_path = img_filename if img_filename else ""
        
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping {cam_name}: Image missing.")
            continue
            
        # Get the refined camera extrinsics
        pose = refined_poses[cam_name]
        rvec = np.array(pose['rvec'], dtype=np.float32).reshape(3, 1)
        tvec = np.array(pose['tvec'], dtype=np.float32).reshape(3, 1)
        
        # Load and prepare image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(14, 10))
        plt.imshow(img)
        
        # Track items we successfully plot for a clean legend
        has_obs, has_proj = False, False
        
        for label, pixel_uv in clicks.items():
            u_obs, v_obs = pixel_uv
            
            # Plot the original user click (Observed)
            plt.scatter(u_obs, v_obs, color='red', marker='x', s=60, lw=2, zorder=3, label='Observed (Click)' if not has_obs else "")
            has_obs = True
            
            # Check if this landmark has a computed consensus 3D coordinate
            if label in final_3d_optodes:
                pt3d = np.array(final_3d_optodes[label], dtype=np.float32).reshape(1, 3)
                
                # Project the consensus 3D point into this camera's refined frame
                px_proj, _ = cv2.projectPoints(pt3d, rvec, tvec, camera_matrix, dist_coeff)
                u_proj, v_proj = px_proj.squeeze()
                
                # Plot the mathematically calculated projection point
                plt.scatter(u_proj, v_proj, color='lime', marker='o', s=50, edgecolors='black', zorder=4, label='Refined Projection' if not has_proj else "")
                has_proj = True
                
                # Draw a displacement vector line between the click and the math projection
                plt.plot([u_obs, u_proj], [v_obs, v_proj], color='yellow', linestyle='--', lw=1.5, zorder=2)
                
                # Print label text near the refined point
                plt.text(u_proj + 15, v_proj - 15, label, color='white', fontsize=9, weight='bold',
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='lime', boxstyle='round,pad=0.2'),
                        zorder=5)
        
        plt.title(f"Refinement Alignment Check: {cam_name} ({img_filename})", fontsize=14, weight='bold')
        plt.legend(loc='upper right', framealpha=0.9, facecolor='black', edgecolor='white', labelcolor='white')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# %% --- CELL 7: 2D PHOTO OVERLAY WITH PROJECTED MESH WIREFRAME ---
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os

if __name__ == '__main__':

    # ==========================================
    # CONFIGURATION — Matches your setups
    # ==========================================
    camera_configs = [
        {'name': 'Camera_1', 'photo_path': 'Camera1_2023-01-15_10-14-42.jpg'},
        {'name': 'Camera_2', 'photo_path': 'Camera2_2023-01-15_10-14-46.jpg'},
        {'name': 'Camera_3', 'photo_path': 'Camera3_2023-01-15_10-14-43.jpg'},
        {'name': 'Camera_4', 'photo_path': 'Camera4_2023-01-15_10-14-44.jpg'},
    ]

    mesh_path = 'meshes/1613_LDMesh_260109.stl'
    calibration_file = 'Logi_calibration.json'

    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)  # Assumes calib_data is in scope
    dist_coeff    = np.array(calib_data['dist_coeff'], dtype=np.float32)

    # Load data computed from your sequential alignment script
    with open('all_optimized_poses.json', 'r') as f:
        refined_poses = json.load(f)

    photo_lookup = {cam['name']: cam['photo_path'] for cam in camera_configs}

    # ==========================================
    # 1. EXTRACT WIREFRAME EDGES FROM MESH
    # ==========================================
    print("Loading mesh and extracting structural wireframe edges...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles)

    # To avoid drawing thousands of internal lines and freezing the notebook, 
    # we extract unique edges instead of drawing every full triangle.
    edges = set()
    for tri in triangles:
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            if v1 > v2:
                v1, v2 = v2, v1
            edges.add((v1, v2))

    edges = list(edges)
    print(f"Extracted {len(edges)} unique geometric edges for projection.")

    # Downsample edges slightly for faster rendering performance if the mesh is dense
    # (Taking every 3rd edge keeps the structure perfectly recognizable but runs 3x faster)
    DOWNSAMPLE_FACTOR = 3
    render_edges = edges[::DOWNSAMPLE_FACTOR]

    # ==========================================
    # 2. PROJECT AND OVERLAY MESH FOR EACH VIEW
    # ==========================================
    print("Loaded poses:")
    for cam_name, pose in refined_poses.items():
        rvec = np.array(pose['rvec']).flatten()
        tvec = np.array(pose['tvec']).flatten()
        R, _ = cv2.Rodrigues(rvec)
        cam_center = (-R.T @ tvec.reshape(3,1)).flatten()
        print(f"  {cam_name}: tvec={np.round(tvec,1)}  world_center={np.round(cam_center,1)}")
    print("\n--- Projecting 3D Mesh Outlines Onto Source Photos ---")

    for cam_name, pose in refined_poses.items():
        img_filename = photo_lookup.get(cam_name)
        img_path =  img_filename if img_filename else ""
        
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping {cam_name}: Image file missing at '{img_path}'")
            continue

        # Load image asset
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        rvec = np.array(pose['rvec'], dtype=np.float32).reshape(3, 1)
        tvec = np.array(pose['tvec'], dtype=np.float32).reshape(3, 1)

        # Project all 3D vertices into this camera's perspective frame
        projected_pts, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeff)
        projected_pts = projected_pts.squeeze().astype(np.int32)

        # Create a transparent overlay layer for the wireframe mask
        mesh_overlay = img.copy()

        # Draw the projected wireframe lines onto the overlay canvas
        for edge in render_edges:
            pt1 = projected_pts[edge[0]]
            pt2 = projected_pts[edge[1]]
            
            # Guard check to ensure lines stay reasonably within image clipping boundaries
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h) or (0 <= pt2[0] < w and 0 <= pt2[1] < h):
                # Drawing neon green wireframe lines (BGR: [0, 255, 0])
                cv2.line(mesh_overlay, tuple(pt1), tuple(pt2), (0, 255, 0), 1, cv2.LINE_AA)

        # Blend the original photo with the mesh wireframe overlay (70% photo, 30% wireframe)
        final_blend = cv2.addWeighted(img, 0.7, mesh_overlay, 0.3, 0)
        final_blend = cv2.cvtColor(final_blend, cv2.COLOR_BGR2RGB)

        # Plot results inline inside your notebook
        plt.figure(figsize=(14, 10))
        plt.imshow(final_blend)
        plt.title(f"3D Mesh Silhouette Alignment: {cam_name} ({img_filename})", fontsize=14, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# %% Cell 8: Run SAM on each image and save distance transform maps for optimization
import os
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

if __name__ == '__main__':
    # 1. Setup Execution Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sam_checkpoint = "sam_vit_h_4b8939.pth"  # Ensure this file is downloaded locally!
    model_type = "vit_h"

    try:
        print("Loading SAM ViT-H Model Weights into memory...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
    except FileNotFoundError:
        print(f"ERROR: Could not find '{sam_checkpoint}'. Please download it before running segmentations.")
        raise

    # 2. Iterate Over Every Configured Camera View Dynamically
    for cam in camera_configs:
        cam_number = cam['name'].split('_')[-1] # Extracts "1", "2", etc.
        cam_suffix = f"C{cam_number}"           # Becomes "C1", "C2", etc.
        
        json_filename = f"points_{cam_suffix}.json"
        target_mask_name = f"mask_{cam_suffix}.png"
        
        print(f"\n" + "-"*50)
        print(f"Processing SAM Segmentation for: {cam['name']} ({cam['photo_path']})")
        print(f"Target Output Mask: {target_mask_name}")
        print("-"*50)
        
        # Check if we have the required 2D click point primitives from Cell 1
        if not os.path.exists(json_filename):
            print(f"⚠️ Skipping {cam['name']}: No underlying '{json_filename}' file found. Run picking first.")
            continue
            
        with open(json_filename, 'r') as f:
            picked_data = json.load(f)
            
        # Extract the user-clicked 2D coordinates to serve as positive foreground hints
        input_points = np.array(picked_data['pts_2d'], dtype=np.float32)
        if len(input_points) == 0:
            print(f"⚠️ Skipping {cam['name']}: No points were saved inside the JSON file.")
            continue

        # 3. Ingest Image & Synchronize Spaces
        image_bgr = cv2.imread(cam['photo_path'])
        if image_bgr is None:
            print(f"❌ Error: Image file '{cam['photo_path']}' could not be loaded from disk.")
            continue
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        # 4. Prompt SAM using point geometries (Using your original stable math)
        input_labels = np.ones(len(input_points), dtype=np.int32)

        print(f"Running SAM inference using {len(input_points)} alignment coordinates as anchors...")
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False  # Reverted: Forces the tightest, single best mask
        )

        best_mask = masks[0]  # Restored your original exact array target
        best_score = scores[0]

        # 5. Visual Diagnostics & Verification
        plt.figure(figsize=(12, 6))

        # Left Panel: Raw Sensor Feed
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title(f"{cam['name']} Raw Image")
        plt.axis('off')

        # Right Panel: Semi-transparent Overlay of the extracted mask boundary
        plt.subplot(1, 2, 2)
        plt.imshow(image_rgb)

        # Build a blue mask alpha layer
        mask_overlay = np.zeros((*best_mask.shape, 4))
        mask_overlay[best_mask] = [0.0, 0.5, 1.0, 0.4]  # Soft Cyan/Blue with 40% Opacity
        plt.imshow(mask_overlay)

        # Display original prompt locations to verify anchor points
        plt.scatter(input_points[:, 0], input_points[:, 1], color='red', marker='*', s=120, label="Anchor Prompts")
        plt.legend(loc='upper right')
        plt.title(f"Extracted Silhouette (Confidence Score: {best_score:.3f})")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # 6. Save Structural Grayscale Output to Hard Drive
        binary_mask_img = (best_mask * 255).astype(np.uint8)
        cv2.imwrite(target_mask_name, binary_mask_img)
        print(f"✅ Successfully written binary map → {target_mask_name}")

    print("\n" + "="*50)
    print("SAM Mask Generation Step Completed across all detected camera frames.")
    print("="*50)

# %% Cell 9: SAM included Optimization solver
import os
import json
import numpy as np
import cv2
import open3d as o3d
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
if __name__ == '__main__':
    # ==============================================================================
    # 1. SETUP LOGISTICS AND CONVENTIONS
    # ==============================================================================
    calibration_file = 'Logi_calibration.json'
    mesh_file        = 'meshes/1613_LDMesh_260109.stl'

    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float64)
    dist_coeff    = np.array(calib_data['dist_coeff'], dtype=np.float64)

    print("Loading low-density STL mesh topology...")
    o3d_mesh = o3d.io.read_triangle_mesh(mesh_file)
    # Compute vertex normals if they don't exist, required for horizon math
    if not o3d_mesh.has_vertex_normals():
        o3d_mesh.compute_vertex_normals()

    # Extract float64 vertex and normal arrays for differentiable math
    mesh_vertices = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    mesh_normals  = np.asarray(o3d_mesh.vertex_normals, dtype=np.float64)

    # Initialize raycasting scene for elastic mesh prior
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    scene  = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    # ==============================================================================
    # 2. DATA INGESTION & COGNITIVE TRIANGULATION
    # ==============================================================================
    camera_names = ['Camera_1', 'Camera_2', 'Camera_3', 'Camera_4']
    camera_poses = {} 
    fiducial_observations = [] 
    masks_dt = {} # Dictionary to hold precomputed distance transforms per camera

    unique_fiducials = set()
    fiducial_data_per_cam = {}

    for idx, name in enumerate(camera_names):
        cam_suffix = name.replace("Camera_", "C")
        
        # 2A. Load PnP Point Alignments
        file_path = f'points_{cam_suffix}.json' 
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            camera_poses[idx] = {'rvec': np.array(data['rvec'], dtype=np.float64), 'tvec': np.array(data['tvec'], dtype=np.float64)}
            fiducial_data_per_cam[idx] = data
            unique_fiducials.update(data['fiducial_names'])

        # 2B. Load SAM Binary Masks & Precompute Distance Transform Maps
        mask_path = f'mask_{cam_suffix}.png' # Expects e.g., mask_C1.png
        # In your main optimization script (Section 2B), replace the distance field generation with this:
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # 1. Cleanly isolate ONLY the 1-pixel wide outer perimeter line
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            edge_canvas = np.zeros_like(mask_img)
            cv2.drawContours(edge_canvas, contours, -1, 255, thickness=1)
            
            # 2. Invert so the line itself is 0, and moving away from it scales up
            inverted_edges = 255 - edge_canvas
            dt = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)
            masks_dt[idx] = dt.astype(np.float64)

    fiducial_list = sorted(list(unique_fiducials))
    fid_name_to_idx = {name: i for i, name in enumerate(fiducial_list)}

    for idx, data in fiducial_data_per_cam.items():
        for uv, f_name in zip(data['pts_2d'], data['fiducial_names']):
            fiducial_observations.append({
                'cam_idx': idx,
                'point_id': fid_name_to_idx[f_name],
                'uv': np.array(uv, dtype=np.float64)
            })

    init_fiducial_3d = np.zeros((len(fiducial_list), 3), dtype=np.float64)
    for idx, data in fiducial_data_per_cam.items():
        for f_name, pt3d in zip(data['fiducial_names'], data['pts_3d']):
            init_fiducial_3d[fid_name_to_idx[f_name]] = np.array(pt3d, dtype=np.float64)

    def triangulate_match_group(file_path, cam_idx_a, cam_idx_b):
        if not os.path.exists(file_path):
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
        with open(file_path, 'r') as f:
            m_data = json.load(f)
        pts_a = np.array(m_data['pts_camera_a'], dtype=np.float64)
        pts_b = np.array(m_data['pts_camera_b'], dtype=np.float64)
        
        pts_a_u = cv2.undistortImagePoints(pts_a[:, np.newaxis, :], camera_matrix, dist_coeff).squeeze()
        pts_b_u = cv2.undistortImagePoints(pts_b[:, np.newaxis, :], camera_matrix, dist_coeff).squeeze()
        
        R_a, _ = cv2.Rodrigues(camera_poses[cam_idx_a]['rvec'])
        R_b, _ = cv2.Rodrigues(camera_poses[cam_idx_b]['rvec'])
        
        P_a = camera_matrix @ np.hstack((R_a, camera_poses[cam_idx_a]['tvec'].reshape(3,1)))
        P_b = camera_matrix @ np.hstack((R_b, camera_poses[cam_idx_b]['tvec'].reshape(3,1)))
        
        pts4D = cv2.triangulatePoints(P_a, P_b, pts_a_u.T, pts_b_u.T)
        pts3D = (pts4D[:3, :] / pts4D[3, :]).T
        return pts3D.astype(np.float64), pts_a, pts_b

    print("Triangulating structural pair matches...")
    pts3d_12, match_uv_1, match_uv_2 = triangulate_match_group('matched_C1_C2.json', 0, 1)
    pts3d_34, match_uv_3, match_uv_4 = triangulate_match_group('matched_C3_C4.json', 2, 3)

    # ==============================================================================
    # 3. GLOBAL BUNDLE ADJUSTMENT ENGINE
    # ==============================================================================
    N_cams = 4
    N_fids = len(fiducial_list)
    N_m12  = len(pts3d_12)
    N_m34  = len(pts3d_34)

    def pack_state_vector(poses, fids_3d, m12_3d, m34_3d):
        components = []
        for i in range(N_cams):
            components.extend(poses[i]['rvec'].flatten())
            components.extend(poses[i]['tvec'].flatten())
        components.extend(fids_3d.flatten())
        components.extend(m12_3d.flatten())
        components.extend(m34_3d.flatten())
        return np.array(components, dtype=np.float64)

    def unpack_state_vector(vector):
        poses = {}
        idx = 0
        for i in range(N_cams):
            poses[i] = {'rvec': vector[idx:idx+3], 'tvec': vector[idx+3:idx+6]}
            idx += 6
        fids_3d = vector[idx:idx + N_fids*3].reshape((N_fids, 3))
        idx += N_fids*3
        m12_3d = vector[idx:idx + N_m12*3].reshape((N_m12, 3))
        idx += N_m12*3
        m34_3d = vector[idx:idx + N_m34*3].reshape((N_m34, 3))
        return poses, fids_3d, m12_3d, m34_3d

    x0 = pack_state_vector(camera_poses, init_fiducial_3d, pts3d_12, pts3d_34)

    def optimization_residuals(vector, lambda_mesh=1.5, lambda_silhouette=0.5):
        poses, fids_3d, m12_3d, m34_3d = unpack_state_vector(vector)
        residuals = []

        # --- Term 1: Reprojection Loss for Mesh Fiducials ---
        for obs in fiducial_observations:
            cam = poses[obs['cam_idx']]
            pt3d = fids_3d[obs['point_id']].reshape(1, 3)
            uv_proj, _ = cv2.projectPoints(pt3d, cam['rvec'], cam['tvec'], camera_matrix, dist_coeff)
            residuals.extend((obs['uv'] - uv_proj.squeeze()).flatten())

        # --- Term 2 & 3: Reprojection Loss for Pure Matches ---
        for i in range(N_m12):
            pt3d = m12_3d[i].reshape(1, 3)
            uv_p1, _ = cv2.projectPoints(pt3d, poses[0]['rvec'], poses[0]['tvec'], camera_matrix, dist_coeff)
            uv_p2, _ = cv2.projectPoints(pt3d, poses[1]['rvec'], poses[1]['tvec'], camera_matrix, dist_coeff)
            residuals.extend((match_uv_1[i] - uv_p1.squeeze()).flatten())
            residuals.extend((match_uv_2[i] - uv_p2.squeeze()).flatten())

        for i in range(N_m34):
            pt3d = m34_3d[i].reshape(1, 3)
            uv_p3, _ = cv2.projectPoints(pt3d, poses[2]['rvec'], poses[2]['tvec'], camera_matrix, dist_coeff)
            uv_p4, _ = cv2.projectPoints(pt3d, poses[3]['rvec'], poses[3]['tvec'], camera_matrix, dist_coeff)
            residuals.extend((match_uv_3[i] - uv_p3.squeeze()).flatten())
            residuals.extend((match_uv_4[i] - uv_p4.squeeze()).flatten())

        # --- Term 4: The Elastic Mesh Constraint ---
        if N_fids > 0:
            query_points = o3d.core.Tensor(fids_3d, dtype=o3d.core.float32)
            closest_lookup = scene.compute_closest_points(query_points)
            closest_mesh_pts = closest_lookup['points'].numpy().astype(np.float64)
            mesh_displacements = fids_3d - closest_mesh_pts
            residuals.extend((lambda_mesh * mesh_displacements).flatten())

        # --- Term 5: Apparent Contour (Silhouette) Distance Constraint (STABILIZED) ---
        for idx, cam in poses.items():
            if idx not in masks_dt:
                continue
                
            dt = masks_dt[idx]
            h, w = dt.shape
            
            # 1. Compute World-Space Camera Center
            R, _ = cv2.Rodrigues(cam['rvec'])
            t = cam['tvec'].reshape(3, 1)
            cam_center = (-R.T @ t).flatten()
            
            # 2. Compute viewing vectors and dot products for ALL vertices
            view_vectors = cam_center - mesh_vertices
            view_vectors /= np.linalg.norm(view_vectors, axis=1, keepdims=True)
            dot_products = np.sum(view_vectors * mesh_normals, axis=1)
            
            # 3. Project ALL mesh vertices to 2D up front to preserve array indexing sizes
            uv_proj, _ = cv2.projectPoints(mesh_vertices, cam['rvec'], cam['tvec'], camera_matrix, dist_coeff)
            uv_proj = uv_proj.squeeze()
            
            x = uv_proj[:, 0]
            y = uv_proj[:, 1]
            
            # 4. Continuous, fixed-size mathematical masking
            # Calculate a smooth weight based on how close to the horizon a vertex is.
            # Vertices perfectly on the horizon get weight ~ 1.0; far away get 0.0.
            horizon_weight = np.exp(-np.square(dot_products) / (2 * np.square(0.05)))
            
            # Check sensor boundaries safely using continuous clipping instead of filtering away indices
            # This keeps the final array size exactly equal to len(mesh_vertices) every single iteration!
            x_clipped = np.clip(x, 0, w - 1.001)
            y_clipped = np.clip(y, 0, h - 1.001)
            
            # 5. Differentiable Lookup
            raw_dists = map_coordinates(dt, [y_clipped, x_clipped], order=1, mode='nearest')
            
            # If a point actually projected off-screen, give it a fixed penalty instead of dropping it
            off_screen_mask = (x < 0) | (x >= w - 1) | (y < 0) | (y >= h - 1)
            raw_dists[off_screen_mask] = 100.0  # Safe static fallback error for off-screen movements
            
            # Apply our horizon weight cleanly
            weighted_dists = raw_dists * horizon_weight * lambda_silhouette
            
            # Append to residuals — this adds exactly len(mesh_vertices) entries every time
            residuals.extend(weighted_dists.flatten())

        return np.array(residuals, dtype=np.float64)

    # ==============================================================================
    # 4. EXECUTION PIPELINE
    # ==============================================================================
    print(f"Starting Joint Bundle Adjustment Optimization over {len(x0)} independent parameters...")

    # The magical inclusion: `loss='soft_l1'` and `f_scale=5.0` automatically
    # suppress the tractor-beam pull of physical glasses/accessories in the SAM masks.
    res = least_squares(
        optimization_residuals, 
        x0, 
        method='trf',       # Trust-region-reflective is required for robust loss functions
        loss='soft_l1',     # Switches quadratic error to linear when errors get too large
        f_scale=5.0,        # Outlier boundary threshold: 5 pixels
        x_scale='jac', 
        verbose=2
    )

    opt_poses, opt_fids, opt_m12, opt_m34 = unpack_state_vector(res.x)

    initial_res = optimization_residuals(x0)
    final_res = res.fun
    print("\n" + "="*50)
    print(f"Initial Total Residual Norm: {np.linalg.norm(initial_res):.2f}")
    print(f"Optimized Total Residual Norm: {np.linalg.norm(final_res):.2f}")
    print("="*50)

    optimized_output = {}
    for i, name in enumerate(camera_names):
        optimized_output[name] = {
            'rvec': opt_poses[i]['rvec'].tolist(),
            'tvec': opt_poses[i]['tvec'].tolist()
        }

    with open('all_optimized_poses.json', 'w') as f:
        json.dump(optimized_output, f, indent=4)
    print("Saved optimized camera geometries globally → all_optimized_poses.json")

    all_optode_clicks = {}
    for idx, data in fiducial_data_per_cam.items():
        cam_name = camera_names[idx]
        all_optode_clicks[cam_name] = {
            name: pixel for pixel, name in zip(data['pts_2d'], data['fiducial_names'])
        }

    with open('all_optode_clicks.json', 'w') as f:
        json.dump(all_optode_clicks, f, indent=4)
    print("Saved → all_optode_clicks.json")
# %% --- DIAGNOSTIC VISUALIZATION ENGINE ---
import matplotlib.pyplot as plt
if __name__ == '__main__':
    for idx, cam in camera_poses.items():
        if idx not in masks_dt:
            continue
            
        dt = masks_dt[idx]
        h, w = dt.shape
        
        # Extract camera parameters for iteration 0
        R, _ = cv2.Rodrigues(cam['rvec'])
        t = cam['tvec'].reshape(3, 1)
        cam_center = (-R.T @ t).flatten()
        
        # Calculate initial horizon vertices
        view_vectors = cam_center - mesh_vertices
        view_vectors /= np.linalg.norm(view_vectors, axis=1, keepdims=True)
        dot_products = np.sum(view_vectors * mesh_normals, axis=1)
        
        # Calculate the horizon weights
        horizon_weight = np.exp(-np.square(dot_products) / (2 * np.square(0.05)))
        
        # Project vertices
        uv_proj, _ = cv2.projectPoints(mesh_vertices, cam['rvec'], cam['tvec'], camera_matrix, dist_coeff)
        uv_proj = uv_proj.squeeze()
        
        # Plotting the diagnostic window
        plt.figure(figsize=(14, 6))
        
        # Panel 1: Check the Distance Field Map
        plt.subplot(1, 2, 1)
        plt.imshow(dt, cmap='jet')
        plt.colorbar(label='Pixel Distance Error')
        plt.title(f"Camera {idx+1} Distance Field\n(The SAM contour line MUST be dark blue/0)")
        
        # Panel 2: Check Horizon Line Overlay
        image_path = camera_configs[idx]['photo_path']
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 2)
        plt.imshow(img_rgb)
        
        # Only scatter points that have a strong horizon weight (> 0.2)
        strong_horizon = horizon_weight > 0.2
        plt.scatter(uv_proj[strong_horizon, 0], uv_proj[strong_horizon, 1], 
                    c=horizon_weight[strong_horizon], cmap='autumn', s=10, label='3D Horizon Line')
        
        plt.title(f"Camera {idx+1} Initial Horizon Overlap\n(Red/Yellow points should sit exactly on the head edge)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Print Residual Weight Check ---
    dummy_vector = pack_state_vector(camera_poses, init_fiducial_3d, pts3d_12, pts3d_34)
    # Run the loss function manually once to look at component scales
    print("\n--- INITIAL LOSS WEIGHT AUDIT ---")
    print(f"Total mesh vertices contributing to silhouette: {len(mesh_vertices)}")

# %% --- CELL: RAYCAST OPTODE POSITIONS ---
# %% --- CELL: INTERACTIVE OPTODE PICKER VIA RAYCASTING ---
import json
import numpy as np
import cv2
import open3d as o3d
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import multiprocessing as mp

with open('all_optimized_poses.json') as f:
    camera_poses_saved = json.load(f)

with open('Logi_calibration.json') as f:
    calib_data = json.load(f)

camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
dist_coeff    = np.array(calib_data['dist_coeff'], dtype=np.float32)

mesh_legacy = o3d.io.read_triangle_mesh('meshes/1613_LDMesh_260109.stl')
mesh_t      = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
scene       = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh_t)

def cast_ray(pixel_uv, rvec, tvec):
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
    t    = np.array(tvec, dtype=np.float32).reshape(3, 1)
    cam_centre = (-R.T @ t).flatten()
    ray_dir    = (R.T @ (np.linalg.inv(camera_matrix)
                         @ np.array([pixel_uv[0], pixel_uv[1], 1.0])
                         ).reshape(3, 1)).flatten()
    ray_dir   /= np.linalg.norm(ray_dir)
    ans   = scene.cast_rays(o3d.core.Tensor(
                [np.hstack([cam_centre, ray_dir])], dtype=o3d.core.float32))
    t_hit = ans['t_hit'].numpy()[0]
    return None if np.isinf(t_hit) else cam_centre + t_hit * ray_dir


def _run_optode_picker(photo_path, queue, cam_name, existing_clicks=None):
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt

    img = mpimg.imread(photo_path)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.set_title(f"{cam_name} — Left-click optodes | Right-click = delete last | Close when done")

    clicks  = []
    markers = []

    if existing_clicks:
        for i, uv in enumerate(existing_clicks):
            ax.plot(uv[0], uv[1], 'go', markersize=6)
            ax.text(uv[0]+5, uv[1]-5, str(i+1), color='lime', fontsize=8)

    def _redraw():
        for m in markers:
            for artist in m:
                artist.remove()
        markers.clear()
        for i, uv in enumerate(clicks):
            d, = ax.plot(uv[0], uv[1], 'ro', markersize=6)
            t  = ax.text(uv[0]+5, uv[1]-5, str(i+1), color='red', fontsize=9)
            markers.append((d, t))
        fig.canvas.draw()

    def on_click(event):
        if not event.inaxes:
            return
        if event.button == 1:
            clicks.append([event.xdata, event.ydata])
            _redraw()
        elif event.button == 3 and clicks:
            clicks.pop()
            _redraw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show(block=True)

    for uv in clicks:
        queue.put(uv)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    camera_configs = [
        {'name': 'Camera_1', 'photo_path': 'Camera1_2023-01-15_10-14-42.jpg'},
        {'name': 'Camera_2', 'photo_path': 'Camera2_2023-01-15_10-14-46.jpg'},
        {'name': 'Camera_3', 'photo_path': 'Camera3_2023-01-15_10-14-43.jpg'},
        {'name': 'Camera_4', 'photo_path': 'Camera4_2023-01-15_10-14-44.jpg'},
    ]

    all_2d_clicks = {}  # { cam_name: [[u,v], [u,v], ...] }

    for cam in camera_configs:
        print(f"\nPicking optodes for: {cam['name']} — close window when done")
        q = mp.Queue()
        p = mp.Process(target=_run_optode_picker,
                       args=(cam['photo_path'], q, cam['name'],
                             all_2d_clicks.get(cam['name'])))
        p.start()
        p.join()

        cam_clicks = []
        while not q.empty():
            cam_clicks.append(q.get())
        all_2d_clicks[cam['name']] = cam_clicks
        print(f"  {len(cam_clicks)} clicks recorded for {cam['name']}")

        with open('optode_clicks_2d.json', 'w') as f:
            json.dump(all_2d_clicks, f, indent=2)

    # ── Raycast onto mesh ─────────────────────────────────────────────────
    print("\n--- RAYCASTING ---")
    all_3d_positions = {}  # { cam_name: [[x,y,z], ...] }

    for cam_name, clicks in all_2d_clicks.items():
        pose = camera_poses_saved[cam_name]
        cam_3d = []
        for uv in clicks:
            pt3d = cast_ray(uv, pose['rvec'], pose['tvec'])
            if pt3d is not None:
                cam_3d.append(pt3d.tolist())
                print(f"  Hit:  {cam_name} point {len(cam_3d)} → {np.round(pt3d, 2)}")
            else:
                print(f"  Miss: {cam_name} point {len(cam_3d)+1}")
                cam_3d.append(None)
        all_3d_positions[cam_name] = cam_3d

    with open('final_3d_optodes.json', 'w') as f:
        json.dump(all_3d_positions, f, indent=2)
    print(f"\nSaved → final_3d_optodes.json")

# %%
# %% --- CELL: CONVERT TO OptLocs FORMAT ---
import json
import numpy as np

with open('final_3d_optodes.json') as f:
    data = json.load(f)

# Collect all non-None 3D points across all cameras into a flat list
all_points = []
for cam_name, points in data.items():
    for pt in points:
        if pt is not None:
            all_points.append(pt)

# Write in the same format as OptLocs_260320.txt
with open('OptLocs_autoOnly_260603.txt', 'w') as f:
    for pt in all_points:
        f.write(f"{pt[0]}, {pt[1]}, {pt[2]}\n")

print(f"Saved {len(all_points)} optode positions → OptLocs_autoOnly_260603.txt")
# %%
