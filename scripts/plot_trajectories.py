import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob


def load_odom_file(filename, max_rows=None):
    data = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = list(map(float, line.strip().split()))
            tx, ty, tz = parts[1], parts[2], parts[3]
            data.append([tx, ty, tz])
            if max_rows is not None and len(data) >= max_rows:
                break
    return np.array(data)


def set_axes_equal(ax):
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )
    centers = np.mean(limits, axis=1)
    max_range = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    for center, set_lim in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        set_lim(center - max_range, center + max_range)


def plot_odom_files(filepaths, max_rows=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = ["black", "red", "green"]

    for idx, filepath in enumerate(filepaths):
        trajectory = load_odom_file(filepath, max_rows)
        label = filepath.split("/")[-2]

        color = colors[idx]  # Get a color for this trajectory

        # Plot the trajectory
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            label=label,
            color=color,
            alpha=0.5,
        )

        # Plot start and end points in same color
        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            trajectory[0, 2],
            marker="o",
            s=50,
            color=color,
            edgecolors="black",
            # label=f"{label} start",
        )
        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            trajectory[-1, 2],
            marker="x",
            s=50,
            color=color,
            edgecolors="black",
            # label=f"{label} end",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Odometry Trajectories")
    ax.legend()
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Modify this path pattern to match your odom files location
    odom_files = [
        # "../output/lioekf_ableviation/8_ladder_not_downsampled_cylinder_unevenly_downsampled/odo_tum.txt",
        # "../output/lioekf_ableviation/9_ransac_radius_tuned/odo_tum.txt",
        "../output/lioekf_ableviation/Full/odo_tum.txt",
        "../output/lioekf_ableviation/Full_no_map_cleanup/odo_tum.txt",
        "../output/lioekf_ableviation/Full_no_fixed_point_ratio/odo_tum.txt",
        # "../output/lioekf_ableviation/Full_with_laser_up/odo_tum.txt",
    ]
    # odom_files = glob.glob("../output/lioekf_ableviation/*/odo_tum.txt")
    max_rows = 3500
    plot_odom_files(odom_files, max_rows)
