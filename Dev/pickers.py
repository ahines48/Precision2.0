# pickers.py
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
def run_2d_picker(photo_path, queue, window_title,
                  existing_pts=None, highlight_idx=None, single_mode=False):
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(photo_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)

    pts, markers = [], []

    if existing_pts:
        for i, (x, y) in enumerate(existing_pts):
            color = 'yellow' if i == highlight_idx else 'red'
            ax.plot(x, y, 'o', color=color, markersize=5)
            ax.text(x + 5, y - 5, str(i + 1), color=color,
                    fontsize=11, fontweight='bold')

    if single_mode:
        ax.set_title(f"Click to replace 2D point {highlight_idx + 1} "
                     f"(yellow = current)\nClose window when done.")
    else:
        ax.set_title(f"{window_title}\n"
                     f"LEFT-CLICK = add  |  RIGHT-CLICK = delete last  |  Close when done")

    def _redraw():
        for d, t in markers:
            d.remove(); t.remove()
        markers.clear()
        offset = len(existing_pts) if existing_pts and not single_mode else 0
        for i, (x, y) in enumerate(pts):
            d, = ax.plot(x, y, 'ro', markersize=5)
            t  = ax.text(x + 5, y - 5, str(i + 1 + offset),
                         color='red', fontsize=11, fontweight='bold')
            markers.append((d, t))
        fig.canvas.draw()

    def on_click(event):
        if not event.inaxes:
            return
        if event.button == 1:
            pts.append([event.xdata, event.ydata])
            print(f"  [2D] point {len(pts)}: ({event.xdata:.1f}, {event.ydata:.1f})")
            if single_mode:
                ax.plot(event.xdata, event.ydata, 'g*', markersize=12)
                fig.canvas.draw()
                plt.close(fig)
            else:
                _redraw()
        elif event.button == 3 and pts and not single_mode:
            pts.pop()
            print(f"  [2D] deleted last ({len(pts)} remaining)")
            _redraw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show(block=True)

    for p in pts:
        queue.put(p)


def run_3d_picker(mesh_path, queue, window_title,
                  existing_pts=None, highlight_idx=None, single_mode=False):
    import numpy as np
    from vedo import Mesh, Plotter, Sphere, Text3D

    mesh  = Mesh(mesh_path)
    title = window_title if not single_mode else \
            f"Click to replace 3D point {highlight_idx + 1}"
    vplt  = Plotter(title=title, axes=1)
    pts, actors = [], []

    if existing_pts:
        for i, p in enumerate(existing_pts):
            c = 'yellow' if i == highlight_idx else 'red'
            vplt.add(Sphere(pos=p, r=3, c=c))
            vplt.add(Text3D(str(i + 1), pos=np.array(p) + [1, 1, 0],
                            s=5, c='black'))

    def on_left_click(event):
        if event.actor != mesh or event.picked3d is None:
            return
        if single_mode and pts:
            return
        idx     = mesh.closest_point(event.picked3d, return_point_id=True)
        point3d = mesh.points[idx].tolist()
        pts.append(point3d)
        n   = len(pts) + (len(existing_pts) if existing_pts and not single_mode else 0)
        sph = Sphere(pos=point3d, r=3, c='lime')
        lbl = Text3D(str(n), pos=np.array(point3d) + [1, 1, 0], s=5, c='black')
        actors.append((sph, lbl))
        vplt.add(sph, lbl)
        vplt.render()
        print(f"  [3D] point {len(pts)}: {np.round(point3d, 2)}")
        if single_mode:
            print("  [3D] point selected — close this window to continue.")

    def on_right_click(event):
        if single_mode or not pts:
            return
        pts.pop()
        sph, lbl = actors.pop()
        vplt.remove(sph, lbl)
        vplt.render()
        print(f"  [3D] deleted last ({len(pts)} remaining)")

    vplt.add_callback("LeftButtonPress",  on_left_click)
    vplt.add_callback("RightButtonPress", on_right_click)
    vplt.show(mesh, interactive=True)
    vplt.close()

    for p in pts:
        queue.put(p)

def visualize_initial_scene_with_bounds(init_poses, o3d_mesh, nominal_poses, init_fids=None):
    """
    Plots the Open3D mesh, the starting positions of the cameras (before bundle adjustment),
    and wireframe bounding boxes representing the fixture tolerances.
    """
    visualizer_geometries = [o3d_mesh]
    
    # Compute scene metrics for scaling markers dynamically
    bbox = o3d_mesh.get_axis_aligned_bounding_box()
    scene_extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    sphere_radius = scene_extent * 0.02
    axis_size     = scene_extent * 0.15
    optical_axis_len = scene_extent * 0.08

    # --- Helper to create axis arrows ---
    def make_axis_arrow(direction, color, size):
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=size * 0.03, cone_radius=size * 0.06,
            cylinder_height=size * 0.8, cone_height=size * 0.2
        )
        arrow.paint_uniform_color(color)
        arrow.compute_vertex_normals()
        if direction == 'x':
            R = np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=np.float64)
        elif direction == 'y':
            R = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)
        else:
            R = np.eye(3)
        arrow.rotate(R, center=[0,0,0])
        return arrow

    # Add Coordinate Axes at origin
    visualizer_geometries.append(make_axis_arrow('x', [1.0, 0.0, 0.0], axis_size))
    visualizer_geometries.append(make_axis_arrow('y', [0.0, 1.0, 0.0], axis_size))
    visualizer_geometries.append(make_axis_arrow('z', [0.0, 0.0, 1.0], axis_size))

    # Add White Origin Sphere
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.5)
    origin_sphere.paint_uniform_color([1.0, 1.0, 1.0])
    origin_sphere.compute_vertex_normals()
    visualizer_geometries.append(origin_sphere)

    # --- Plot Fiducials if provided ---
    if init_fids is not None and len(init_fids) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(init_fids)
        pcd.paint_uniform_color([1.0, 0.8, 0.0]) # Gold
        visualizer_geometries.append(pcd)

    # --- Define Tolerance Dimensions (from your layout) ---
    # Convert mm tolerances to whatever your mesh units are. 
    # (Assuming mesh is in mm because tolerances are 20, 10, 60)
    dx = 20.0
    dy = 10.0
    dz = 60.0

    print("\n" + "="*50)
    print("  INITIAL GEOMETRY & BOUNDING BOX VISUALIZER")
    print("="*50)

    # --- Render Cameras & Bounding Boxes ---
    for cam_idx, pose in init_poses.items():
        # Compute current camera center in world space: C = -R^T * t
        R, _ = cv2.Rodrigues(pose['rvec'])
        t = pose['tvec'].reshape(3, 1)
        cam_center = (-R.T @ t).flatten()

        # 1. Draw Starting Camera Position (Sphere)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        # Cam 1-2 Red, Cam 3-4 Green
        color = [1.0, 0.2, 0.2] if cam_idx < 2 else [0.2, 1.0, 0.2]
        sphere.paint_uniform_color(color)
        sphere.translate(cam_center)
        sphere.compute_vertex_normals()
        visualizer_geometries.append(sphere)

        # 2. Draw Starting Optical Axis Vector Line
        optical_tip = (R.T @ np.array([[0], [0], [optical_axis_len]])).flatten() + cam_center
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([cam_center, optical_tip]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line_set.paint_uniform_color([0.2, 0.4, 1.0]) # Blue axis vector
        visualizer_geometries.append(line_set)

        # 3. Create and Draw the Bounding Box for the Camera
        if cam_idx == 0:
            print(f"  Camera_1: Fixed Anchor Frame at [0, 0, 0]")
            # Camera 0 is fixed, no bounding box required (or a tiny 1mm marker box)
            continue
        
        # Look up nominal position for this camera from fixture config
        _, t_nominal = nominal_poses[cam_idx]
        cx, cy, cz = t_nominal.flatten()

        print(f"  Camera_{cam_idx+1} Box Center (Nominal): [{cx:+.1f}, {cy:+.1f}, {cz:+.1f}]")

        # Define 8 corners of the bounding box around nominal center
        min_bound = np.array([cx - dx, cy - dy, cz - dz])
        max_bound = np.array([cx + dx, cy + dy, cz + dz])
        
        o3d_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(o3d_bbox)
        
        # Color the bounding box magenta so it clearly pops out against the mesh
        bbox_lines.paint_uniform_color([1.0, 0.0, 1.0]) 
        visualizer_geometries.append(bbox_lines)

    print("="*50 + "\n")

    # Launch Open3D Window
    o3d.visualization.draw_geometries(
        visualizer_geometries,
        window_name="Initial Positions & Bounding Boxes  |  Boxes = Magenta",
        width=1280,
        height=720
    )

def visualize_mesh_error_vectors(opt_fids, scene):
    """
    3D quiver plot of fiducial→mesh error vectors.
    Arrow length and color both encode error magnitude so arrows are always visible.
    """
    # 1. Compute closest mesh points and error vectors
    query_points = o3d.core.Tensor(opt_fids, dtype=o3d.core.float32)
    closest_lookup = scene.compute_closest_points(query_points)
    closest_mesh_pts = closest_lookup['points'].numpy().astype(np.float64)
 
    vectors = closest_mesh_pts - opt_fids  # direction: point → mesh surface
    magnitudes = np.linalg.norm(vectors, axis=1)
 
    # 2. Normalize arrow display length to scene scale
    #    Each arrow is drawn at a fixed fraction of scene extent, colored by true magnitude
    scene_extent = np.linalg.norm(opt_fids.max(axis=0) - opt_fids.min(axis=0))
    display_scale = scene_extent * 0.06  # visual arrow length, independent of true error
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vectors = np.where(
            magnitudes[:, np.newaxis] > 1e-9,
            vectors / magnitudes[:, np.newaxis],
            0.0
        )
 
    # 3. Colormap: blue (small error) → red (large error)
    norm = plt.Normalize(vmin=magnitudes.min(), vmax=magnitudes.max())
    cmap = plt.cm.coolwarm
    arrow_colors = cmap(norm(magnitudes))
 
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
 
    # Plot fiducial points
    ax.scatter(
        opt_fids[:, 0], opt_fids[:, 1], opt_fids[:, 2],
        c='gold', s=40, zorder=5, label='Optimized Fiducials', depthshade=True
    )
 
    # Plot closest mesh points
    ax.scatter(
        closest_mesh_pts[:, 0], closest_mesh_pts[:, 1], closest_mesh_pts[:, 2],
        c='gray', s=15, alpha=0.5, label='Closest Mesh Points', depthshade=True
    )
 
    # Draw one arrow per point, colored by magnitude
    for i in range(len(opt_fids)):
        ax.quiver(
            opt_fids[i, 0], opt_fids[i, 1], opt_fids[i, 2],
            unit_vectors[i, 0] * display_scale,
            unit_vectors[i, 1] * display_scale,
            unit_vectors[i, 2] * display_scale,
            color=arrow_colors[i],
            arrow_length_ratio=0.25,
            linewidth=1.5
        )
 
    # Colorbar showing true error magnitude
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('True Error Magnitude (scene units)', fontsize=10)
 
    ax.set_title("Mesh Alignment Diagnostics — Error Vectors\n"
                 "(Arrow direction = toward mesh surface; color/length = error magnitude)",
                 fontsize=11)
    ax.set_xlabel("X")

    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()



def visualize_scene_geometry(opt_poses, o3d_mesh, opt_fids=None):
    visualizer_geometries = [o3d_mesh]

    bbox = o3d_mesh.get_axis_aligned_bounding_box()
    scene_extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    sphere_radius = scene_extent * 0.02
    axis_length   = scene_extent * 0.08
    axis_size     = scene_extent * 0.15  # bigger than before so labels are obvious

    # --- Axis arrows: one per direction, color-coded and thick ---
    def make_axis_arrow(direction, color, size):
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=size * 0.03,
            cone_radius=size * 0.06,
            cylinder_height=size * 0.8,
            cone_height=size * 0.2
        )
        arrow.paint_uniform_color(color)
        arrow.compute_vertex_normals()
        # create_arrow points along +Z by default, so rotate to target axis
        if direction == 'x':
            R = np.array([[0,0,1],[0,1,0],[-1,0,0]], dtype=np.float64)  # Z→X
        elif direction == 'y':
            R = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)  # Z→Y
        else:  # z — already aligned
            R = np.eye(3)
        arrow.rotate(R, center=[0,0,0])
        return arrow

    # Red=X, Green=Y, Blue=Z  (matches Open3D convention)
    visualizer_geometries.append(make_axis_arrow('x', [1.0, 0.0, 0.0], axis_size))
    visualizer_geometries.append(make_axis_arrow('y', [0.0, 1.0, 0.0], axis_size))
    visualizer_geometries.append(make_axis_arrow('z', [0.0, 0.0, 1.0], axis_size))

    # Small sphere at origin as anchor point
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 1.5)
    origin_sphere.paint_uniform_color([1.0, 1.0, 1.0])
    origin_sphere.compute_vertex_normals()
    visualizer_geometries.append(origin_sphere)

    # --- Print axis legend to console so you can cross-reference while rotating ---
    print("\n" + "="*40)
    print("  WORLD AXIS LEGEND (viewer colours)")
    print("="*40)
    print("  RED   arrow  →  +X axis")
    print("  GREEN arrow  →  +Y axis")
    print("  BLUE  arrow  →  +Z axis")
    print("  WHITE sphere →  World Origin [0,0,0]")

    # Print camera positions in world space so you know where they sit
    print("\n  Camera world positions (C = -R^T * t):")
    for cam_idx, pose in opt_poses.items():
        R, _ = cv2.Rodrigues(pose['rvec'])
        t = pose['tvec'].reshape(3, 1)
        C = (-R.T @ t).flatten()
        print(f"  Camera_{cam_idx+1}: [{C[0]:+.3f}, {C[1]:+.3f}, {C[2]:+.3f}]")
    print("="*40 + "\n")

    # --- Optional fiducials ---
    if opt_fids is not None and len(opt_fids) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(opt_fids)
        pcd.paint_uniform_color([1.0, 0.8, 0.0])
        visualizer_geometries.append(pcd)

    # --- Camera spheres + optical axis lines (unchanged from your original) ---
    for cam_idx, pose in opt_poses.items():
        R, _ = cv2.Rodrigues(pose['rvec'])
        t = pose['tvec'].reshape(3, 1)
        cam_center = (-R.T @ t).flatten()

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        color = [1.0, 0.2, 0.2] if cam_idx < 2 else [0.2, 1.0, 0.2]
        sphere.paint_uniform_color(color)
        sphere.translate(cam_center)
        sphere.compute_vertex_normals()
        visualizer_geometries.append(sphere)

        optical_tip = (R.T @ np.array([[0], [0], [axis_length]])).flatten() + cam_center
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([cam_center, optical_tip]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line_set.paint_uniform_color([0.2, 0.4, 1.0])
        visualizer_geometries.append(line_set)

    o3d.visualization.draw_geometries(
        visualizer_geometries,
        window_name="Scene Geometry — Cameras & Mesh  |  R=X  G=Y  B=Z",
        width=1280,
        height=720,
        point_show_normal=False
    )

def report_alignment_errors(opt_poses, opt_fids, opt_m12, opt_m34, N_cams, N_fids, N_m12, N_m34, fiducial_observations,
                            match_uv_1, match_uv_2, match_uv_3, match_uv_4, camera_matrix, dist_coeff, scene):
    print("\n" + "="*50)
    print("           PERFORMANCE METRICS REPORT")
    print("="*50)
    
    # --- 1. CAMERA REGISTRATION ERROR (Reprojection Error) ---
    cam_errors = {i: [] for i in range(N_cams)}
    
    # Gather fiducial reprojection errors
    for obs in fiducial_observations:
        cam = opt_poses[obs['cam_idx']]
        pt3d = opt_fids[obs['point_id']].reshape(1, 3)
        uv_proj, _ = cv2.projectPoints(pt3d, cam['rvec'], cam['tvec'], camera_matrix, dist_coeff)
        error_pixels = np.linalg.norm(obs['uv'] - uv_proj.squeeze())
        cam_errors[obs['cam_idx']].append(error_pixels)
        
    # Gather Cam 1-2 pure match errors
    for i in range(N_m12):
        pt3d = opt_m12[i].reshape(1, 3)
        for cam_idx, match_uv in [(0, match_uv_1), (1, match_uv_2)]:
            uv_proj, _ = cv2.projectPoints(pt3d, opt_poses[cam_idx]['rvec'], opt_poses[cam_idx]['tvec'], camera_matrix, dist_coeff)
            cam_errors[cam_idx].append(np.linalg.norm(match_uv[i] - uv_proj.squeeze()))

    # Gather Cam 3-4 pure match errors
    for i in range(N_m34):
        pt3d = opt_m34[i].reshape(1, 3)
        for cam_idx, match_uv in [(2, match_uv_3), (3, match_uv_4)]:
            uv_proj, _ = cv2.projectPoints(pt3d, opt_poses[cam_idx]['rvec'], opt_poses[cam_idx]['tvec'], camera_matrix, dist_coeff)
            cam_errors[cam_idx].append(np.linalg.norm(match_uv[i] - uv_proj.squeeze()))

    print("1. Camera Registration (Mean Reprojection Error):")
    all_pixel_errors = []
    for cam_idx, errors in cam_errors.items():
        if errors:
            mean_err = np.mean(errors)
            max_err = np.max(errors)
            all_pixel_errors.extend(errors)
            print(f"   - Camera_{cam_idx+1}: Mean = {mean_err:.2f} px (Max = {max_err:.2f} px)")
    print(f"   >> GLOBAL CAMERA REGISTRATION ERROR: {np.mean(all_pixel_errors):.2f} px")
    print("-"*50)

    # --- 2. MESH ALIGNMENT ERROR ---
    if N_fids > 0:
        query_points = o3d.core.Tensor(opt_fids, dtype=o3d.core.float32)
        closest_lookup = scene.compute_closest_points(query_points)
        closest_mesh_pts = closest_lookup['points'].numpy().astype(np.float64)
        
        # Calculate standard Euclidean distance (3D) for each point
        mesh_distances = np.linalg.norm(opt_fids - closest_mesh_pts, axis=1)
        
        print("2. Mesh Alignment Performance:")
        print(f"   - Mean Distance to Mesh: {np.mean(mesh_distances):.3f} units")
        print(f"   - Max Distance to Mesh:  {np.max(mesh_distances):.3f} units")
    else:
        print("2. Mesh Alignment Performance: No fiducials available to evaluate.")
    print("="*50)


# Inside ~/Precision2.0/pickers.py

def build_fixture_bounds(poses_initial, fids_initial, m12_initial, m34_initial,
                         pos_tolerances, rot_tolerances, n_cams=4):
    """
    Bounds cameras 1-3 relative to camera 0's initial solvePnP pose.
    Camera 0 is fixed (tight bounds). Cameras 1-3 can deviate from their
    initial pose by pos_tolerances / rot_tolerances — but since all are
    anchored to the same starting poses, this preserves relative spacing
    while letting the rig float freely toward the mesh.
    """
    lower_bounds = []
    upper_bounds = []

    dx, dy, dz = pos_tolerances
    drx, dry, drz = rot_tolerances
    pos_tol = np.array([dx, dy, dz], dtype=np.float64)
    rot_tol = np.array([drx, dry, drz], dtype=np.float64)

    for cam_idx in range(n_cams):
        rvec_init = np.array(poses_initial[cam_idx]['rvec'], dtype=np.float64).flatten()
        tvec_init = np.array(poses_initial[cam_idx]['tvec'], dtype=np.float64).flatten()

        if cam_idx == 0:
            # Fix camera 0 as the rigid anchor
            lower_bounds.extend(rvec_init - 1e-6)
            upper_bounds.extend(rvec_init + 1e-6)
            lower_bounds.extend(tvec_init - 1e-6)
            upper_bounds.extend(tvec_init + 1e-6)
        else:
            # Allow deviation from solvePnP starting pose within tolerances
            lower_bounds.extend(rvec_init - rot_tol)
            upper_bounds.extend(rvec_init + rot_tol)
            lower_bounds.extend(tvec_init - pos_tol)
            upper_bounds.extend(tvec_init + pos_tol)

    # 3D points float freely
    n_pts = (len(fids_initial) + len(m12_initial) + len(m34_initial)) * 3
    lower_bounds.extend([-np.inf] * n_pts)
    upper_bounds.extend([ np.inf] * n_pts)

    return np.array(lower_bounds, dtype=np.float64), np.array(upper_bounds, dtype=np.float64)