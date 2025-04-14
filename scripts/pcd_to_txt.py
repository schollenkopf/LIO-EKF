# import numpy as np


# def convert_pcd_to_colmap(
#     pcd_file, output_file="../output/first6_calibrated_cleaned/points3D.txt"
# ):
#     with open(pcd_file, "r") as f:
#         lines = f.readlines()

#     # Find data start
#     data_start = (
#         next(i for i, line in enumerate(lines) if line.strip() == "DATA ascii") + 1
#     )
#     points = np.loadtxt(lines[data_start:], dtype=float)

#     with open(output_file, "w") as f:
#         for i, row in enumerate(points, start=1):

#             x, y, z = row[:3]  # XYZ
#             r, g, b = (
#                 (255, 255, 255) if len(row) < 6 else map(int, row[3:6])
#             )  # RGB if available
#             f.write(f"{i} {x} {y} {z} {r} {g} {b} 0.0 0\n")

#     print(f"Converted {len(points)} points to {output_file}")


# convert_pcd_to_colmap("../output/first6.pcd")
# conda activate master_thesis_311
import open3d as o3d
import numpy as np


def convert_pcd_to_colmap(
    pcd_file,
    output_file="../output/scenes_colmap/first6_calibrated_cleaned3/sparse/0/points3D.txt",
):
    # Load point cloud using Open3D
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Convert to NumPy arrays
    points = np.asarray(pcd.points)
    colors = (
        (np.asarray(pcd.colors) * 255).astype(int)
        if pcd.has_colors()
        else np.full_like(points, 255)
    )

    # Write to COLMAP format
    with open(output_file, "w") as f:
        for i, (xyz, rgb) in enumerate(zip(points, colors), start=1):
            x, y, z = xyz
            r, g, b = rgb
            f.write(f"{i} {x} {y} {z} {int(r)} {int(g)} {int(b)} 0.0 0\n")

    print(f"Converted {len(points)} points to {output_file}")


# Run conversion
convert_pcd_to_colmap("../output/first6_pre_cleaned3.ply")
