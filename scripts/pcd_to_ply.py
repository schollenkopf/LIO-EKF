import open3d as o3d
import os
import numpy as np

input_folder = "../data/esbjerg/scans/"
output_folder = "../data/esbjerg/scansSubSample/"
os.makedirs(output_folder, exist_ok=True)

skip = 1000
for i, filename in enumerate(sorted(os.listdir(input_folder))):
    if i < 400 or i>2100:
        continue
    print(i)
    if filename.endswith(".pcd"):
        pcd_path = os.path.join(input_folder, filename)
        pcd = o3d.io.read_point_cloud(pcd_path)
        num_points = np.asarray(pcd.points).shape[0]
        gray = np.tile([0.5, 0.5, 0.5], (num_points, 1))
        pcd.colors = o3d.utility.Vector3dVector(gray)
        pcd.remove_non_finite_points()
        ply_filename = os.path.splitext(filename)[0] + ".ply"
        ply_path = os.path.join(output_folder, ply_filename)
        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)