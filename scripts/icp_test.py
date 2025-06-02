


import open3d as o3d
import numpy as np
import pyransac3d as pyrsc
import time


def pca_downsample(pcd, threshold=0.1, downsample_factor=20):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
    )
    # o3d.visualization.draw_geometries(
    #     [pcd], point_show_normal=True, window_name="target and source"
    # )
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    translation_C = normals.T @ normals
    eigenvalues, eigenvectors = np.linalg.eigh(translation_C)

    sorted_indices = np.argsort(eigenvalues)

    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    scaled_eigenvalues = eigenvalues / np.sum(eigenvalues)
    # print(scaled_eigenvalues)
    if scaled_eigenvalues[2] > (1 - threshold):
        # Flat surface (e.g., wall or floor): one dominant direction
        weak_axes = eigenvectors[:, :2]
    elif scaled_eigenvalues[0] < threshold:
        # Linear structure (e.g., column): one weak direction
        weak_axes = eigenvectors[:, :1]
    else:
        # no need to downsample
        return points

    filtered_points = points

    for axis in weak_axes.T:
        projections = filtered_points @ axis
        low = np.percentile(projections, 25)
        high = np.percentile(projections, 75)
        mask = (projections >= low) & (projections <= high)
        filtered_points = filtered_points[mask]

    return filtered_points[::downsample_factor]


def ransac_downsample(pcd2, downsample_factor=20):
    points2 = np.asarray(pcd2.points)
    cyl = pyrsc.Cylinder()
    center, axis, radius, inliers = cyl.fit(points2, thresh=0.15, maxIteration=1000)
    cylinder_points = points2[inliers]
    inside_points = points2[np.setdiff1d(np.arange(len(points2)), inliers)]
    cylinder_points = points2[inliers]
    store_ply(o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(inside_points)
    ),"source_ransac_inside")
    store_ply(o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(cylinder_points)
    ),"source_ransac_cylinder")
    z_vals = cylinder_points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min
    cylinder_points = cylinder_points[z_vals > (z_max - 0.5 * z_range)][
        ::downsample_factor
    ]
    pcd2_preprocessed = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(np.vstack([cylinder_points, inside_points]))
    )
    return pcd2_preprocessed

def store_ply(pcd,name,folder="icptest",file_type=".ply"):
    path = folder + "/" + name + file_type
    o3d.io.write_point_cloud(path, pcd, write_ascii=True)

def custom_downsample(points, downsample_factor=20):
    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min
    return points[z_vals > (z_max - 0.5 * z_range)][::downsample_factor]


def icp(source, target, method):
    T = o3d.pipelines.registration.registration_icp(
        source,
        target,
        1.0,
        np.eye(4),
        method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    ).transformation
    return o3d.geometry.PointCloud(source).transform(T)


# rotation example:
# pcd = o3d.io.read_point_cloud("../data/esbjerg/scans/1741440505887518.pcd")
# pcd2 = o3d.io.read_point_cloud("../data/esbjerg/scans/1741440510787694.pcd")

# translation example
pcd = o3d.io.read_point_cloud("../data/esbjerg/scans/1741440573289115.pcd")
pcd2 = o3d.io.read_point_cloud("../data/esbjerg/scans/1741440573788945.pcd")

pcd.remove_non_finite_points()
pcd2.remove_non_finite_points()

# pcd2_preprocessed = ransac_downsample(pcd2)

c1 = o3d.io.read_point_cloud("icptest/source_mask3d/source_cable1.ply")
c2 = o3d.io.read_point_cloud("icptest/source_mask3d/source_cable2.ply")
l = o3d.io.read_point_cloud("icptest/source_mask3d/source_ladder.ply")
r = o3d.io.read_point_cloud("icptest/source_mask3d/source_cylinder.ply")

# c1 = o3d.io.read_point_cloud("icptest/source_column1.ply")
# c2 = o3d.io.read_point_cloud("icptest/source_column2.ply")
# l = o3d.io.read_point_cloud("icptest/source_ladder.ply")
# r = o3d.io.read_point_cloud("icptest/source_cylinder.ply")

start = time.time()
c1p = pca_downsample(c1)
c2p = pca_downsample(c2)
lp = pca_downsample(l)
rp = pca_downsample(r)
print(f"Execution time: {time.time() - start:.5f} seconds")

pcd2_preprocessed = o3d.geometry.PointCloud(
    points=o3d.utility.Vector3dVector(np.vstack([c1p, c2p, lp, rp]))
)

# set colors
pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (len(pcd.points), 1)))
pcd2.colors = o3d.utility.Vector3dVector(
    np.tile([0.5, 0.0, 0.0], (len(pcd2.points), 1))
)
pcd2_preprocessed.colors = o3d.utility.Vector3dVector(
    np.tile([0.0, 0.5, 0.0], (len(pcd2_preprocessed.points), 1))
)


o3d.visualization.draw_geometries(
    [pcd, pcd2], point_show_normal=False, window_name="target and source"
)
# store_ply(pcd,"target")
# store_ply(pcd2,"raw_source")
o3d.visualization.draw_geometries(
    [pcd2_preprocessed],
    point_show_normal=False,
    window_name="target and source downsampled",
)


# store_ply(pcd2_preprocessed,"source_preprocessed")

# Point to point
pcd2_ptp = icp(
    pcd2, pcd, o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
pcd2_preprocessed_ptp = icp(
    pcd2_preprocessed,
    pcd,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
)

o3d.visualization.draw_geometries(
    [pcd, pcd2_ptp, pcd2_preprocessed_ptp], window_name="Point to Point comparison"
)
# store_ply(pcd2_ptp,"raw_source_p2point")
# store_ply(pcd2_preprocessed_ptp,"source_preprocessed_ptp")
# point to plane
pcd.estimate_normals()
pcd2.estimate_normals()
pcd2_preprocessed.estimate_normals()
pcd2_ptpl = icp(
    pcd2,
    pcd,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
)
pcd2_preprocessed_ptpl = icp(
    pcd2_preprocessed,
    pcd,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
)
o3d.visualization.draw_geometries(
    [pcd, pcd2_ptpl, pcd2_preprocessed_ptpl], window_name="Point to Plane comparison"
)
# store_ply(pcd2_ptpl,"raw_source_p2lane")
# store_ply(pcd2_preprocessed_ptpl,"source_preprocessed_ptpl")
# Generalised ICP
pcd2_gicp = icp(
    pcd2, pcd, o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
)
pcd2_preprocessed_gicp = icp(
    pcd2_preprocessed,
    pcd,
    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
)
o3d.visualization.draw_geometries(
    [pcd, pcd2_gicp, pcd2_preprocessed_gicp], window_name="GICP comparison"
)
# store_ply(pcd2_gicp,"raw_source_gicp")
# store_ply(pcd2_preprocessed_gicp,"source_preprocessed_gicp")
