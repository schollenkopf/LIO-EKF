import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# OG flight:
# video_path = "../../data/a107ac6c-305a-4208-a5a4-b4ef41c68c60/video/standalone_session_a107ac6c-305a-4208-a5a4-b4ef41c68c60.mp4"
# csv_path = "./world_camera_poses_first6_2.csv"
# out_folder = "first6"
# video_start_time = 1717332837.74 - 4.16  # based on first big up motion in slam output

# Esbjerg
video_path = "../../data/flight-esbjerg/video.mp4"
csv_path = "./world_camera_poses_esbjerg_preprocessed.csv"
out_folder = "scenes_colmap/esbjerg"
video_start_time = 1741440389.22 - 4.500003  # esbjerg based on json file


output_image_dir = "../output/" + out_folder + "/images"
points3D_path = "../output/" + out_folder + "/sparse/0/points3D.txt"
output_txt_path = "../output/" + out_folder + "/sparse/0/images.txt"
output_plot_path = "../output/" + out_folder + "/camera_poses_plot.png"


# Create the output directory if it doesn't exist
os.makedirs(output_image_dir, exist_ok=True)

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(csv_path)

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)
import numpy as np
from scipy.spatial.transform import Rotation as R

# Set up a 3D plot for camera poses with orientation
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-10, 0])

camera_id = 1
# Define arrow scale
arrow_length = 0.4  # Adjust based on your scene size
last_frametime = 0
max_fps = 0.5
frame_count = 0
with open(output_txt_path, "w") as txt_file:
    for index, row in data.iterrows():
        # if index < 164:  # skip static initial pos for esbjerg
        #     continue
        # if index % 10 != 0:
        #     continue
        # if index != 14 and (index > 1 and index < 70 or index % 4 != 0):
        #     continue

        timestamp = row["timestamp"]
        frame_time = timestamp - video_start_time

        if not frame_time - last_frametime > (1 / max_fps):
            continue
        frame_count += 1
        print(frame_count)
        last_frametime = frame_time
        print(frame_time)
        frame_index = int(frame_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at {frame_index}")
            continue

        image_filename = f"{output_image_dir}/frame_{index+1}.jpg"
        cv2.imwrite(image_filename, frame)

        bx, by, bz = row["bx"], row["by"], row["bz"]
        bqx, bqy, bqz, bqw = row["bqx"], row["bqy"], row["bqz"], row["bqw"]
        cx, cy, cz = row["cx"], row["cy"], row["cz"]
        cqx, cqy, cqz, cqw = row["cqx"], row["cqy"], row["cqz"], row["cqw"]

        tx, ty, tz = row["x"], row["y"], row["z"]

        t_world_base = [bx, by, bz]
        R_world_base = R.from_quat([bqx, bqy, bqz, bqw]).as_matrix().T
        t_base_camera = [cx, cy, cz]
        R_base_camera = R.from_quat([cqx, cqy, cqz, cqw]).as_matrix()
        R_world_camera = R_base_camera @ R_world_base

        t_camera_world = -R_world_camera @ t_world_base - R_base_camera @ t_base_camera

        qx, qy, qz, qw = R.from_matrix(R_world_camera).as_quat()
        tx, ty, tz = t_camera_world

        txt_file.write(
            f"{index+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} frame_{index+1}.jpg\n\n"
        )

        # Plot position
        # ax.scatter(tx, ty, tz, c="r", marker="o")

        # Convert quaternion to rotation matrix

        rot_matrix = R_world_camera

        tx, ty, tz = -R_world_camera.T @ t_camera_world

        rot_matrix = rot_matrix.T

        # Extract basis vectors
        x_axis = rot_matrix[:, 0] * arrow_length  # Right (X direction)
        y_axis = rot_matrix[:, 1] * arrow_length  # Up (Y direction)
        z_axis = rot_matrix[:, 2] * arrow_length  # Forward (Z direction)

        # Plot orientation arrows
        ax.quiver(
            tx,
            ty,
            tz,
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="r",
            label="X-axis" if index == 0 else "",
        )
        ax.quiver(
            tx,
            ty,
            tz,
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="g",
            label="Y-axis" if index == 0 else "",
        )
        ax.quiver(
            tx,
            ty,
            tz,
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="b",
            label="Z-axis" if index == 0 else "",
        )

        ax.text(tx, ty, tz, f"{index+1}", color="black", fontsize=8, weight="bold")

        rot_matrix = R_world_base.T

        # Extract basis vectors
        x_axis = rot_matrix[:, 0] * arrow_length  # Right (X direction)
        y_axis = rot_matrix[:, 1] * arrow_length  # Up (Y direction)
        z_axis = rot_matrix[:, 2] * arrow_length  # Forward (Z direction)

        # Plot orientation arrows
        ax.quiver(
            tx,
            ty,
            tz,
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="r",
            alpha=0.2,
        )
        ax.quiver(
            tx,
            ty,
            tz,
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="g",
            alpha=0.2,
        )
        ax.quiver(
            tx,
            ty,
            tz,
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="b",
            alpha=0.2,
        )


# Release the video capture object
cap.release()

point_cloud = []
if os.path.exists(points3D_path):
    with open(points3D_path, "r") as f:
        for i, line in enumerate(f):
            if line.startswith("#") or not line.strip() or i % 100 != 0:
                continue
            parts = line.split()
            x, y, z = map(float, parts[1:4])  # Extract X, Y, Z
            point_cloud.append([x, y, z])

# Convert to numpy array
point_cloud = np.array(point_cloud)

# Plot point cloud
ax.scatter(
    point_cloud[:, 0],
    point_cloud[:, 1],
    point_cloud[:, 2],
    c="k",
    marker=".",
    s=1,
    alpha=0.5,
    label="3D Points",
)

# Show and save the camera poses plot
ax.set_title("Camera Poses (Position & Orientation)")
plt.legend()
plt.savefig(output_plot_path)
plt.show()

print("Processing completed with orientation visualization.")
