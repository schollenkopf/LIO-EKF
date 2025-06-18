import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time


c1 = o3d.io.read_point_cloud("icptest/source_mask3d/source_cable1.ply")
c2 = o3d.io.read_point_cloud("icptest/source_mask3d/source_cable2.ply")
l = o3d.io.read_point_cloud("icptest/source_mask3d/source_ladder.ply")
r = o3d.io.read_point_cloud("icptest/source_mask3d/source_cylinder.ply")


def pca(normals, points):
    print("paper approach")
    point_cross_normal = np.cross(points, normals)
    rotation_C = point_cross_normal.T @ point_cross_normal
    rot_eigenvalues, rot_eigenvectors = np.linalg.eigh(rotation_C)
    print(np.sum(rot_eigenvalues))
    rot_eigenvalues = rot_eigenvalues/ np.sum(rot_eigenvalues)
    translation_C = normals.T @ normals
    trans_eigenvalues, trans_eigenvectors = np.linalg.eigh(translation_C)
    trans_eigenvalues = trans_eigenvalues/ np.sum(trans_eigenvalues)
    return rot_eigenvalues,rot_eigenvectors,trans_eigenvalues,trans_eigenvectors

def pca_downsample_step1(pcd, eigenvalues,eigenvectors,threshold=0.1,voxelsize = 0.15):

    points = np.asarray(pcd.points)
   
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
    return source_pcd

def pca_downsample_step2(pcd, eigenvalues,eigenvectors,threshold=0.1,voxelsize = 0.15):
    points = np.asarray(pcd.points)
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

    
    source_pcd = source_pcd.voxel_down_sample(voxel_size=voxelsize*1.5)

    return source_pcd


def draw_axis(axis,weight,rot=True):
    arrow_center = np.array([0, 0, 0])
    axis = axis / np.linalg.norm(axis)
    arrow_tip = arrow_center + axis * 0.5
    # weight = 1 if weight > 1/3 else  weight * 3
    color = [1, 0, 0] if weight < 0.1 else [0, 1, 0]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([arrow_center, arrow_tip])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([color])
    return line_set

def create_origin_dot(radius=0.05):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color([0, 0, 1])  # Blue
    sphere.translate([0, 0, 0])  # Centered at origin
    return sphere


for i,pcd in enumerate([c1,c2,l,r]):
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)


    rot_eigenvalues,rot_eigenvectors,trans_eigenvalues,trans_eigenvectors = pca(normals,points)




    axises_o3d = []
    for weight,axis in zip(trans_eigenvalues,trans_eigenvectors.T):
        axises_o3d += [draw_axis(axis,weight,False)]


    classesColors = [np.array([1.0, 0.55, 0.0]),np.array([0, 1, 0.0]),np.array([0.0, 0.0, 1.0]),np.array([1, 185/255, 185/255])]
    pcd.colors = o3d.utility.Vector3dVector(np.tile(classesColors[i], (len(pcd.points), 1)))
    o3d.visualization.draw_geometries([ *axises_o3d,pcd])
    if i ==2:
        continue
    pcd = pca_downsample_step1(pcd,trans_eigenvalues,trans_eigenvectors)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(classesColors[i], (len(pcd.points), 1)))
    o3d.visualization.draw_geometries([ *axises_o3d,pcd])
    pcd = pca_downsample_step2(pcd,trans_eigenvalues,trans_eigenvectors)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(classesColors[i], (len(pcd.points), 1)))
    o3d.visualization.draw_geometries([ *axises_o3d,pcd])





