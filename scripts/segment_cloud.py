import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import matplotlib as plt


def segment_cloud(points, normals, eps=0.1, min_samples=4):
    db = KMeans(n_clusters=50)
    labels = db.fit(points).labels_
    colors = np.zeros((points.shape[0], 3))
    unique_labels = np.unique(labels)
    print(len(unique_labels))
    new_points = np.zeros((points.shape[0], 3))
    for label in unique_labels:

        weak_axises, has_weakness = analise_cloud(
            points[labels == label], normals[labels == label]
        )
        if label == -1:
            color = np.array([0.5, 0.5, 0.5])
        elif has_weakness:
            color = np.array([0.0, 0.0, 0.0])
        else:
            new_points[labels == label] = points[labels == label]
            color = plt.cm.jet(label / len(unique_labels))[:3]

        colors[labels == label] = color

    return colors, new_points


def analise_cloud(points, normals, weak_thresh=0.04, dominant_thresh=1):
    point_cross_normal = np.cross(points, normals)
    weak_axises = []
    has_weakness = False
    for source in [normals, point_cross_normal]:
        C = source.T @ source
        eig_vals, eig_vecs = np.linalg.eigh(C)
        sorted_indices = np.argsort(eig_vals)
        eig_vals = eig_vals[sorted_indices]
        eig_vecs = eig_vecs[:, sorted_indices]
        # print(np.min(eig_vals) / np.max(eig_vals))
        if np.max(eig_vals) / np.sum(eig_vals) >= dominant_thresh:
            weak_axises += [eig_vecs[:, :2]]
            has_weakness = True
        elif np.min(eig_vals) / np.max(eig_vals) <= weak_thresh:
            weak_axises += [eig_vecs[:, :1]]
            has_weakness = True
    return weak_axises, has_weakness


def draw_axis(dominant_normal, scale=0.5):
    arrow_center = np.array([0, 0, 2])
    arrow_tip = arrow_center + dominant_normal * scale
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([arrow_center, arrow_tip])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    return line_set


def weighted_cloud(pcd):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=10)
    )

    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    weak_axises, has_weakness = analise_cloud(points, normals)
    print(has_weakness)
    colors, new_points = segment_cloud(points, normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.points = o3d.utility.Vector3dVector(new_points)
    return pcd, weak_axises


pcd = o3d.io.read_point_cloud(
    "scan_esb.pcd", remove_nan_points=True, remove_infinite_points=True
)
pcd, weak_axises = weighted_cloud(pcd)
print(weak_axises)


axises_o3d = []
for wa in weak_axises:
    for axis in wa.T:
        axises_o3d += [draw_axis(axis)]

o3d.visualization.draw_geometries([pcd, *axises_o3d], point_show_normal=False)
