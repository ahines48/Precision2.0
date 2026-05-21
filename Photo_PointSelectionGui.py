import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def select_2d_points(image_path, json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
            
    num_points = len(data)

    image_points = []

    def on_image_click(event):
        if event.inaxes:
            if len(image_points) < num_points:
                x, y = event.xdata, event.ydata
                image_points.append([x, y])
                plt.plot(x, y, 'ro', markersize=10)
                plt.draw()
                print(f"Picked image point: ({x:.2f}, {y:.2f})")
            if len(image_points) == num_points:
                plt.close()
                
    img = mpimg.imread(image_path) 
    fig, ax = plt.subplots()
    ax.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', on_image_click)
    plt.title(f"Pick {num_points} points on the image")
    plt.show()
    ax.set_navigate(True)
    fig.canvas.toolbar_visible = True
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    # Save the 2D points as the next row in the JSON file
    with open(json_path, "r+") as f:
        data = json.load(f)
        data.append(image_points)
        f.seek(0)
        json.dump(data, f)
        f.truncate()
    print("Saved 2D image points to JSON.")
    return image_points