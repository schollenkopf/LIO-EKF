import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

out_folder = "first6"
csv_path = "./world_camera_poses_first6_2.csv"
points3D_path = "../output/" + out_folder + "/sparse/0/points3D.txt"


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


# Define arrow scale
arrow_length = 0.6  # Adjust based on your scene size
poses = [
    [1, 1, -1],
    [1, -1, -1],
    [-1, 1, -1],
    [1, 1, 1],
    [1, -1, 1],
    [-1, 1, 1],
    [0, 0, 0],
]
data = pd.read_csv(csv_path)
count = 0
for index, row in data.iterrows():
    if index % 10 != 0:
        continue
    # if index != 14 and (index > 1 and index < 70 or index % 4 != 0):
    #     continue
    print(index)
    timestamp = row["timestamp"]

    qx, qy, qz, qw = row["qx"], row["qy"], row["qz"], row["qw"]
    bx, by, bz = row["bx"], row["by"], row["bz"]
    bqx, bqy, bqz, bqw = row["bqx"], row["bqy"], row["bqz"], row["bqw"]
    cx, cy, cz = row["cx"], row["cy"], row["cz"]
    cqx, cqy, cqz, cqw = row["cqx"], row["cqy"], row["cqz"], row["cqw"]

    tx, ty, tz = row["x"], row["y"], row["z"]
    camera_id = 1

    t_world_base = [bx, by, bz]
    R_world_base = R.from_quat([bqx, bqy, bqz, bqw]).as_matrix().T
    t_base_camera = [cx, cy, cz]
    R_base_camera = R.from_quat([cqx, cqy, cqz, cqw]).as_matrix()
    R_world_camera = R_base_camera @ R_world_base

    # tx, ty, tz = poses[count]
    # count += 1
    t_camera_world = -R_world_camera @ t_world_base - R_base_camera @ t_base_camera

    tx, ty, tz = -R_world_camera.T @ t_camera_world

    rot_matrix = R_world_camera.T  # R.from_quat([bqx, bqy, bqz, bqw]).as_matrix().T

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

    rot_matrix = R_world_base.T  # R.from_quat([bqx, bqy, bqz, bqw]).as_matrix().T

    # Extract basis vectors
    x_axis = rot_matrix[:, 0] * arrow_length  # Right (X direction)
    y_axis = rot_matrix[:, 1] * arrow_length  # Up (Y direction)
    z_axis = rot_matrix[:, 2] * arrow_length  # Forward (Z direction)

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
plt.show()
