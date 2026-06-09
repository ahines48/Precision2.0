import json
import os
import numpy as np
import cv2
import multiprocessing as mp

# ==============================================================================
# 1. SEQUENTIAL 2D PAIR PICKER
# ==============================================================================

def _run_pure_2d_picker(photo_path, result_list, window_title, current_point_idx):
    """
    Runs in its own process. Allows picking a single 2D coordinate 
    corresponding to the current active point index.
    """
    import matplotlib
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