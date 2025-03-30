import numpy as np


def convert_pcd_to_colmap(
    pcd_file, output_file="../output/first6/sparse/0/points3D.txt"
):
    with open(pcd_file, "r") as f:
        lines = f.readlines()

    # Find data start
    data_start = (
        next(i for i, line in enumerate(lines) if line.strip() == "DATA ascii") + 1
    )
    points = np.loadtxt(lines[data_start:], dtype=float)

    with open(output_file, "w") as f:
        for i, row in enumerate(points, start=1):

            x, y, z = row[:3]  # XYZ
            r, g, b = (
                (255, 255, 255) if len(row) < 6 else map(int, row[3:6])
            )  # RGB if available
            f.write(f"{i} {x} {y} {z} {r} {g} {b} 0.0 0\n")

    print(f"Converted {len(points)} points to {output_file}")


convert_pcd_to_colmap("../output/first6.pcd")
