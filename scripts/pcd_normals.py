import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the point cloud
# pcd = o3d.io.read_point_cloud("scan4.pcd")
# points = np.asarray(pcd.points)


pcd = o3d.geometry.PointCloud()
points = []

# floor/wall
# nr_points = 100
# xs = np.linspace(0, 1, nr_points)  # Angle around the cylinder
# ys = np.linspace(0, 1, nr_points)  # Height along the cylinder
# for x in xs:
#     for y in ys:
#         points.append(np.array([x, y, 0]))


# sphere
nr_points = 100
phi = np.linspace(0, np.pi, nr_points)  # Elevation angle
theta = np.linspace(0, 2 * np.pi, nr_points)  # Azimuth angle
for p in phi:
    for t in theta:
        x = np.sin(p) * np.cos(t)
        y = np.sin(p) * np.sin(t)
        z = np.cos(p)
        points.append(np.array([x, y, z]))
print(np.min(points))
print(np.max(points))

# cylinder with lid
# nr_points = 500
# theta = np.linspace(0, 2 * np.pi, nr_points)  # Angle around the cylinder
# z_vals = np.linspace(0, 1, 50)  # Height along the cylinder
# for z in z_vals:
#     for t in theta:
#         x = np.cos(t)  # Cylinder radius is 1
#         y = np.sin(t)
#         points.append(np.array([x, y, z]))
# for r in np.linspace(-1, 1, 50):
#     for t in theta:
#         x = r * np.cos(t)  # Cylinder radius is 1
#         y = r * np.sin(t)
#         points.append(np.array([x, y, 0]))

# smooth cylinder
# nr_points = 500
# theta = np.linspace(0, 2 * np.pi, nr_points)  # Angle around the cylinder
# z_vals = np.linspace(0, 1, 50)  # Height along the cylinder
# for z in z_vals:
#     for t in theta:
#         x = np.cos(t)  # Cylinder radius is 1
#         y = np.sin(t)
#         points.append(np.array([x, y, z]))

# corridor
# nr_points = 100
# for i in range(2):

#     xs = np.linspace(0, 1, nr_points)  # Angle around the cylinder
#     ys = np.linspace(0, 1, nr_points)  # Height along the cylinder
#     for x in xs:
#         for y in ys:
#             points.append(np.array([x, y, i]))
# for i in range(2):
#     xs = np.linspace(0, 1, nr_points)  # Angle around the cylinder
#     ys = np.linspace(0, 1, nr_points)  # Height along the cylinder
#     for x in xs:
#         for y in ys:
#             points.append(np.array([x, i, y]))
# points = np.array(points)


pcd.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd])

# o3d.visualization.draw_geometries([pcd], point_show_normal=False)
RELATIVE_EIGVAL_THRESH = 0.1
GRID_RESOLUTION = 0.3


def compute_pca_matrix(normals, points):
    print("paper approach")
    point_cross_normal = np.cross(points, normals)
    rotation_C = point_cross_normal.T @ point_cross_normal
    rot_eigenvalues, rot_eigenvectors = np.linalg.eigh(rotation_C)
    print(np.linalg.matrix_rank(rotation_C))
    translation_C = normals.T @ normals
    trans_eigenvalues, trans_eigenvectors = np.linalg.eigh(translation_C)
    print(np.linalg.matrix_rank(translation_C))

    eigenvalues = rot_eigenvalues
    eigenvectors = rot_eigenvectors
    # A = np.vstack((point_cross_normal.T, normals.T))

    # C = A @ A.T

    # eigenvalues, eigenvectors = np.linalg.eigh(C)
    # rank = np.linalg.matrix_rank(C)
    # print(rank)
    sorted_indices = np.argsort(eigenvalues)

    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    print(eigenvalues)
    print(eigenvectors)
    return eigenvalues, eigenvectors[:, :1]


def pca(normals):
    cov_matrix = np.cov(normals.T)
    print(cov_matrix)
    print(np.linalg.matrix_rank(cov_matrix))
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    print(eigenvalues)
    sorted_eigenval_indeces = eigenvalues.argsort()
    eigenvectors = eigenvectors[:, sorted_eigenval_indeces]
    eigenvalues = eigenvalues[sorted_eigenval_indeces]
    return eigenvalues, eigenvectors


def raytrace_normals(points, normals, grid_resolution=GRID_RESOLUTION):
    """Traces normals using actual ray tracing towards the point cloud center and visualizes voting space."""
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    grid_size = np.ceil((max_bound - min_bound) / grid_resolution).astype(int)
    grid_origin = min_bound
    vote_grid = np.zeros(grid_size, dtype=int)

    for ray_origin, ray_direction in zip(points, normals):
        step = grid_resolution / np.linalg.norm(ray_direction)
        ray = ray_origin.copy()

        while np.all(ray >= min_bound) and np.all(ray <= max_bound):
            grid_idx = np.floor((ray - grid_origin) / grid_resolution).astype(int)
            if np.any(grid_idx < 0) or np.any(grid_idx >= grid_size):
                break  # Stop if out of grid bounds
            vote_grid[tuple(grid_idx)] += 1
            ray += ray_direction * step  # Continue tracing

    # Find the voxel with the highest votes
    max_votes = np.max(vote_grid)

    intersection_voxel = np.argwhere(vote_grid == max_votes)
    intersection_point = grid_origin + intersection_voxel.mean(axis=0) * grid_resolution
    intersection_type = "whatever"
    # Determine intersection type based on vote distribution
    if max_votes < 5:
        intersection_type = "scattered"  # No clear intersection
    elif len(intersection_voxel) > 10:
        intersection_type = "line"  # Votes spread along a line
    else:
        intersection_type = "point"  # Clear intersection point

    # Visualize the voting space
    vote_pcd = o3d.geometry.PointCloud()
    # vote_threshold = np.percentile(vote_grid, 90)
    vote_threshold = max_votes / 2
    vote_points = [
        grid_origin + grid_resolution * idx
        for idx in np.argwhere(vote_grid > vote_threshold)
    ]
    # print([[(v / 200) * 255, 0, 0] for v in vote_grid[vote_grid > 10]])
    # print([idx for idx in np.argwhere(vote_grid > 0)])
    vote_pcd.points = o3d.utility.Vector3dVector(vote_points)
    vote_pcd.colors = o3d.utility.Vector3dVector(
        [[(v / max_votes), 0, 0] for v in vote_grid[vote_grid > vote_threshold]]
    )

    return intersection_type, intersection_point, vote_pcd


def cloud_normal_pca(pcd):
    position_weakness = False
    position_spanning_vectors = None
    intersection_type = False
    intersection_point = None
    rotation_weakness = False
    # pcd.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    # )
    pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.points))
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    # cloud_center = np.mean(points, axis=0)
    # oriented_normals = []
    # for ray_origin, normal in zip(points, normals):
    #     direction = cloud_center - ray_origin
    #     ray_direction = -normal if np.dot(normal, direction) >= 0 else normal
    #     oriented_normals += [ray_direction]
    # pcd.normals = o3d.utility.Vector3dVector(oriented_normals)
    # normals = np.asarray(pcd.normals)

    start_time = time.time()
    eigenvalues, position_spanning_vectors = compute_pca_matrix(normals, points)
    end_time = time.time()
    print(f"paper execution time: {end_time - start_time:.6f} seconds")

    # start_time = time.time()
    # eigenvalues, eigenvectors = pca(normals)
    # end_time = time.time()
    # print(f"pca execution time: {end_time - start_time:.6f} seconds")

    # scaled_eigenvalues = eigenvalues / np.sum(eigenvalues)

    # if scaled_eigenvalues[0] > RELATIVE_EIGVAL_THRESH:
    #     # check volume spun using determinant?
    #     position_weakness = False  # no eigenvalue below threshold, therefore sufficient points with normals to span 3d space
    #     position_spanning_vectors = eigenvectors

    # elif scaled_eigenvalues[2] > (1 - RELATIVE_EIGVAL_THRESH):
    #     position_weakness = True  # one eigenvalue above threshold, therefore too dominant normal direction, e.g. floor
    #     rotation_weakness = True
    #     position_spanning_vectors = np.array(eigenvectors[:, 2])
    #     weak_axis = np.array(eigenvectors[:, 2])
    # else:
    #     position_weakness = True
    #     position_spanning_vectors = eigenvectors[:, 1:]
    vote_pcd = None
    # if not rotation_weakness:

    #     intersection_type, intersection_point, vote_pcd = raytrace_normals(
    #         points, normals
    #     )

    return (
        position_weakness,
        position_spanning_vectors,
        intersection_type,
        intersection_point,
        vote_pcd,
    )


def draw_axis(dominant_normal, scale=0.5):
    arrow_center = np.array([0, 0, 2])
    arrow_tip = arrow_center + dominant_normal * scale
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([arrow_center, arrow_tip])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    return line_set


(
    position_weakness,
    position_spanning_vectors,
    intersection_type,
    intersection_point,
    vote_pcd,
) = cloud_normal_pca(pcd)
print("Position Weakness:", position_weakness)
print("Position Vectors", position_spanning_vectors)
print("Intersection type :", intersection_type)
print("Intersection point", intersection_point)


# # normals = np.asarray(pcd.normals)
# # vertical_threshold = 0.8  # Adjust if needed (closer to 1 = stricter filtering)
# # valid_normals = np.abs(normals[:, 2]) > vertical_threshold
# # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[valid_normals])
# # pcd.normals = o3d.utility.Vector3dVector(normals[valid_normals])

# # voxel_size = 0.05  # Adjust this value as needed
# # pcd = pcd.voxel_down_sample(voxel_size)
# og_pcd = pcd.points

# pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)
# )
# normal_arrows = []
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)
# selected_points = []
# selected_normals = []
# for i in range(2):
#     dominant_normal = compute_dominant_normal(pcd)
#     normal_arrows += [draw_dominant_normal(pcd, dominant_normal)]
#     normals = np.asarray(pcd.normals)
#     dot_products = np.abs(np.dot(normals, dominant_normal))
#     # cosine_similarity = np.abs(
#     #     normals @ dominant_normal
#     # )  # Dot product with absolute value
#     # sorted_similarity = np.argsort(cosine_similarity)
#     # top_50_indices = np.argsort(dot_products)[-100:]

#     threshold = 0.1
#     # low_similarity_indices = np.where(cosine_similarity < threshold)[0]
#     indices_not_on_plane = np.where(dot_products > threshold)[0]
#     selected_points += list(
#         np.asarray(pcd.points)[np.where(dot_products < threshold)[0]]
#     )
#     selected_normals += list(
#         np.asarray(pcd.normals)[np.where(dot_products < threshold)[0]]
#     )

#     # print("Number points left: ", np.size(low_similarity_indices))

#     pcd.points = o3d.utility.Vector3dVector(
#         np.asarray(pcd.points)[indices_not_on_plane]
#     )
#     pcd.normals = o3d.utility.Vector3dVector(normals[indices_not_on_plane])
#     o3d.visualization.draw_geometries([pcd, *normal_arrows], point_show_normal=True)

# pcd.points = o3d.utility.Vector3dVector(np.array(selected_points))
# pcd.normals = o3d.utility.Vector3dVector(np.array(selected_normals))
# # Visualize point cloud with normals

# pcd.normals = o3d.utility.Vector3dVector(np.cross(points, np.asarray(pcd.normals)) * 10)

axises_o3d = []
for axis in position_spanning_vectors.T:
    axises_o3d += [draw_axis(axis)]
if vote_pcd:
    o3d.visualization.draw_geometries(
        [pcd, *axises_o3d, vote_pcd], point_show_normal=True
    )
else:
    o3d.visualization.draw_geometries([pcd, *axises_o3d], point_show_normal=True)
