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
nr_points = 50
xs = np.linspace(-0.5, 0.5, nr_points) 
ys = np.linspace(-0.5, 0.5, nr_points) 
for x in xs:
    for y in ys:
        points.append(np.array([x, y, -0.5]))
center_rot = np.array([-0.2, 0, 0])
center_trans = np.array([0.2, 0,0])

# sphere
# nr_points = 50
# phi = np.linspace(0, np.pi-0.5, nr_points)  # Elevation angle
# theta = np.linspace(0, 1 * np.pi, nr_points)  # Azimuth angle
# for p in phi:
#     for t in theta:
#         x = np.sin(p) * np.cos(t)
#         y = np.sin(p) * np.sin(t)
#         z = np.cos(p)
#         points.append(np.array([x, y, z]))
# print(np.min(points))
# print(np.max(points))

# cylinder with lid
# nr_points = 50
# theta = np.linspace(0, 1 * np.pi, nr_points)  # Angle around the cylinder
# z_vals = np.linspace(-0.5, 0.5, 50)  # Height along the cylinder
# for z in z_vals:
#     for t in theta:
#         x = np.cos(t)  # Cylinder radius is 1
#         y = np.sin(t)
#         points.append(np.array([x, y, z]))
# for t in theta:
#     for r in np.linspace(0, 1, 50):
#         x = r * np.cos(t)  # Cylinder radius is 1
#         y = r * np.sin(t)
#         points.append(np.array([x, y, 0.5]))

# smooth cylinder
# nr_points = 50
# theta = np.linspace(0, 1 * np.pi, nr_points)  # Angle around the cylinder
# z_vals = np.linspace(-0.5, 0.5, 50)  # Height along the cylinder
# for z in z_vals:
#     for t in theta:
#         x = np.cos(t)  # Cylinder radius is 1
#         y = np.sin(t)
#         points.append(np.array([x, y, z]))

# corridor
# nr_points = 50
# for i in [-1,1]:
#     xs = np.linspace(0, 1, nr_points)  # Angle around the cylinder
#     ys = np.linspace(-1, 1, nr_points)  # Height along the cylinder
#     for x in xs:
#         for y in ys:
#             points.append(np.array([x, y, i]))
# xs = np.linspace(0, 1, nr_points)  # Angle around the cylinder
# zs = np.linspace(-1, 1, nr_points)  # Height along the cylinder
# for i in [-1,1]:
#     for x in xs:
#         for z in zs:
#             points.append(np.array([x, i, z]))
# points = np.array(points)
# center_rot = np.array([0, -0.2, 0])
# center_trans = np.array([0, 0.2,0])

pcd.points = o3d.utility.Vector3dVector(points)
pcd.estimate_normals()
    # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)

# o3d.visualization.draw_geometries([pcd])

# o3d.visualization.draw_geometries([pcd], point_show_normal=True)

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



def draw_axis(axis,weight,rot=True):
    arrow_center = center_rot if rot else center_trans
    axis = axis / np.linalg.norm(axis)
    arrow_tip = arrow_center + axis * 0.1
    weight = 1 if weight > 1/3 else  weight * 3
    color = [1 - weight, weight, 0]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([arrow_center, arrow_tip])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([color])
    return line_set


normals = np.asarray(pcd.normals)
points = np.asarray(pcd.points)


rot_eigenvalues,rot_eigenvectors,trans_eigenvalues,trans_eigenvectors = pca(normals,points)

def create_origin_dot(radius=0.05):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color([0, 0, 1])  # Blue
    sphere.translate([0, 0, 0])  # Centered at origin
    return sphere



axises_o3d = []
for weight,axis in zip(rot_eigenvalues,rot_eigenvectors.T):
    axises_o3d += [draw_axis(axis,weight)]
for weight,axis in zip(trans_eigenvalues,trans_eigenvectors.T):
     axises_o3d += [draw_axis(axis,weight,False)]

origin = o3d.geometry.PointCloud()
origin_points = np.array([0,0,0])
origin.points = o3d.utility.Vector3dVector([origin_points])
origin.colors = o3d.utility.Vector3dVector(np.tile([0,0, 1], (len(origin.points), 1)))

pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (len(pcd.points), 1)))
o3d.visualization.draw_geometries([ origin,*axises_o3d,pcd], point_show_normal=True)






w = np.linspace(0, 1, 500)
f = np.where(w > 1/3, 1, 3 * w)


fig, ax = plt.subplots()

# Create vertical gradient background from red to green
gradient = np.linspace(0, 1, 256).reshape(-1, 1)
ax.imshow(gradient, extent=[0, 1, 0, 1], aspect='auto', origin='lower',
          cmap='RdYlGn')

# Plot the function on top
ax.plot(w, f, color='black', linewidth=2)
ax.set_xlabel('scaled eigenvalue')
ax.set_ylabel('color')
plt.grid(True)
plt.show()