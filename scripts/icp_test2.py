import open3d as o3d
import numpy as np
import copy


def icp(source, target, method):
    T = o3d.pipelines.registration.registration_icp(
        source,
        target,
        1.0,
        np.eye(4),
        method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    ).transformation
    t_est = T[:3, 3]
    print(t_est)
    t_gt = np.array(TGT)
    print(t_est-t_gt)
    trans_error = np.linalg.norm(t_est - t_gt)
    print("trans_error:",trans_error)
    return o3d.geometry.PointCloud(source).transform(T)

def pca_downsample(pcd, threshold=0.1, downsample_factor=20,voxelsize = 0.15):
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

    source_points = points

    for axis in weak_axes.T:
        projections = source_points @ axis
        low = np.percentile(projections, 25)
        high = np.percentile(projections, 75)
        mask = (projections >= low) & (projections <= high)
        source_points = source_points[mask]


    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    
    
    source_pcd = source_pcd.voxel_down_sample(voxel_size=voxelsize)

    return np.asarray(source_pcd.points)

def draw_cube(cube_center):
    cube_size = 0.05
    points = [] 
    cube_range = np.linspace(-cube_size / 2, cube_size / 2, 5)
    for dx in cube_range:
        for dy in cube_range:
            for dz in cube_range:
                is_surface = (
                    abs(dx) == cube_size / 2 or
                    abs(dy) == cube_size / 2 or
                    abs(dz) == cube_size / 2
                )
                if is_surface:
                    noise = np.random.normal(scale=0.002, size=3)
                    point = cube_center + np.array([dx, dy, dz]) + noise
                    points.append(point)
    return points


def create_corridor_with_cube(cube_center):
    pcd = o3d.geometry.PointCloud()
    points = []

    nr_points = 50
    for i in [-1,1]:
        xs = np.linspace(0, 1, nr_points)  
        ys = np.linspace(-1, 1, nr_points) 
        for x in xs:
            for y in ys:
                noise = np.random.normal(scale=0.002, size=3)
                points.append(np.array([x, y, i])+noise)
    xs = np.linspace(0, 1, nr_points)  
    zs = np.linspace(-1, 1, nr_points)  
    for i in [-1,1]:
        for x in xs:
            for z in zs:
                noise = np.random.normal(scale=0.002, size=3)
                points.append(np.array([x, i, z])+noise)
    floor = points
    cube = draw_cube(cube_center)
    points += cube

    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    floor_pcd = o3d.geometry.PointCloud()
    floor_pcd.points = o3d.utility.Vector3dVector(np.array(floor))

    cube_pcd = o3d.geometry.PointCloud()
    cube_pcd.points = o3d.utility.Vector3dVector(np.array(cube))
    return pcd,floor_pcd,cube_pcd




def create_floor_with_cube(cube_center):
    pcd = o3d.geometry.PointCloud()
    points = []

    nr_points = 50
    xs = np.linspace(-0.5, 0.5, nr_points)
    ys = np.linspace(-0.5, 0.5, nr_points)
    for x in xs:
        for y in ys:
            noise = np.random.normal(scale=0.004, size=3)
            point = np.array([x, y, -0.5]) + noise
            points.append(point)
    floor = points
    cube = draw_cube(cube_center)
    points += cube

    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    floor_pcd = o3d.geometry.PointCloud()
    floor_pcd.points = o3d.utility.Vector3dVector(np.array(floor))

    cube_pcd = o3d.geometry.PointCloud()
    cube_pcd.points = o3d.utility.Vector3dVector(np.array(cube))
    return pcd,floor_pcd,cube_pcd

# pcd,floor_pcd,cube_pcd = create_corridor_with_cube(np.array([0.1, 0.1,-0.95]))
# pcd2,floor_pcd,cube_pcd = create_corridor_with_cube(np.array([0.15, 0.1, -0.95]))
# TGT = [-0.05,0,0]
pcd,floor_pcd,cube_pcd = create_floor_with_cube(np.array([0.1, 0.1,-0.45]))
pcd2,floor_pcd,cube_pcd = create_floor_with_cube(np.array([0.12, 0.13, -0.45]))
TGT = [-0.02,-0.03,0]


pcd2_preprocessed = o3d.geometry.PointCloud(
    points=o3d.utility.Vector3dVector(np.vstack([pca_downsample(floor_pcd), pca_downsample(cube_pcd)]))
)

pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 0.0], (len(pcd.points), 1)))
pcd2.colors = o3d.utility.Vector3dVector(
    np.tile([0.5, 0.0, 0.0], (len(pcd2.points), 1))
)
pcd2_preprocessed.colors = o3d.utility.Vector3dVector(
    np.tile([0.0, 0.5, 0.0], (len(pcd2_preprocessed.points), 1))
)

o3d.visualization.draw_geometries(
    [pcd2_preprocessed], window_name="Preprocessed"
)

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


print("Point to Plane")


pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# pcd2_preprocessed.estimate_normals()
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


print("GICP")
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