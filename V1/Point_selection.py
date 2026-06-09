import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from vedo import Mesh, Plotter, Sphere, Text3D

def pick_points(mesh_file, photo_path, json_file):
    # --- PHASE 1: Show the Photo as a Reference (Non-blocking) ---
    plt.ion()  
    img = mpimg.imread(photo_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Reference Photo (Pick 3D points in Vedo first)")
    ax.set_navigate(True)
    fig.canvas.toolbar_visible = True
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    plt.show()
    plt.pause(0.1) 

    # --- PHASE 2: Pick 3D Points in Vedo (Blocking) ---
    with open(json_file, "w") as f:
        json.dump([], f)

    mesh = Mesh(mesh_file)
    picked_3d_points = []
    vplt = Plotter(title="3D Mesh Point Picker", axes=1)

    def on_3d_click(event):
        if event.actor == mesh:
            pt = mesh.picked3d
            idx = mesh.closest_point(pt, return_point_id=True)
            point3d = mesh.points[idx]
            picked_3d_points.append(point3d.tolist())
            
            point_num = len(picked_3d_points)
            print(f"Picked 3D point {point_num}: {point3d}")
            
            # Add a sphere
            vplt.add(Sphere(pos=point3d, r=3, c="red"))
            
            # Add the number slightly offset so it doesn't clip into the sphere
            # You may need to adjust the offset/scale 's' depending on your mesh size
            offset = np.array([1, 1, 0])
            vplt.add(Text3D(str(point_num), pos=point3d + offset, s=5, c='black'))

    vplt.add_callback("mouse click", on_3d_click)

    print("Vedo window open. Pick your 3D points.")
    print("When finished with 3D points, press 'q' to switch to the photo.")
    
    # Script pauses here until you press 'q'
    vplt.show(mesh, interactive=True) 

    with open(json_file, "w") as f:
        json.dump(picked_3d_points, f)

    num_points = len(picked_3d_points)

    # --- PHASE 3: Pick 2D Points on the Photo (Blocking) ---
    # The Vedo window stays visibly open on your screen, but interaction shifts to Matplotlib
    plt.ioff() 
    ax.set_title(f"Now pick {num_points} points on this image")
    fig.canvas.draw()

    image_points = []
    def on_image_click(event):
        if event.inaxes:
            if len(image_points) < num_points:
                x, y = event.xdata, event.ydata
                image_points.append([x, y])
                point_num = len(image_points)
                
                # Mark and number the 2D point
                ax.plot(x, y, 'ro', markersize=3)
                ax.text(x + 3, y - 3, str(point_num), color='red', fontsize=12, fontweight='bold')
                fig.canvas.draw()
                print(f"Picked image point {point_num}: ({x:.2f}, {y:.2f})")
                
            if len(image_points) == num_points:
                plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', on_image_click)
    print(f"Switching focus to the photo. Please pick {num_points} matching points.")
    
    # Script pauses here until you pick the required number of 2D points
    plt.show() 

    # --- PHASE 4: Cleanup and Save ---
    vplt.close() # Now we finally close the 3D window

    with open(json_file, "r+") as f:
        data = json.load(f)
        data.append(image_points)
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    print("Saved all points to JSON.")
    return np.array(picked_3d_points, dtype=np.float32), np.array(image_points, dtype=np.float32)