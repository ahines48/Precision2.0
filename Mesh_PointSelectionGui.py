from vedo import *
import numpy as np
import json


def select_3d_points(mesh_path, json_path):
    # Clear JSON file
    with open(json_path, "w") as f:
        json.dump([], f)   
        
    # Load your mesh
    mesh = Mesh(mesh_path)
    picked_points = []

    # Plotter setup
    vplt = Plotter(title="3D Mesh Point Picker", axes=1)

    def on_click(event):
        if event.actor == mesh:
            # Get closest point index
            pt = mesh.picked3d
            idx = mesh.closest_point(pt, return_point_id=True)
            point3d = mesh.points[idx]
            picked_points.append(point3d.tolist())
            print(f"Picked point {idx}: {point3d}")

            # Show a small sphere on the selected point
            vplt.add(Sphere(pos=point3d, r=.001, c="red"))

    # Add mesh and interactor
    vplt.show(mesh, __doc__, interactive=False)

    # Add click callback
    vplt.add_callback("mouse click", on_click)

    # Add key binding for saving
    def on_key_press(evt):
        if evt.keypress == 's':
            with open(json_path, "w") as f:
                json.dump(picked_points, f)
            print("Saved selected 3D points to JSON.")

    vplt.add_callback("key press", on_key_press)

    # Start interaction
    vplt.interactive().close()
    return picked_points