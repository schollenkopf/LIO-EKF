import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import time
class ConstrainedICP:
    def __init__(self, max_iterations=50, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def fit(self, source_points, target_points, source_constraints, initial_transform=None):
        """
        Fit constrained ICP alignment
        
        Args:
            source_points: Nx3 array of source points
            target_points: Mx3 array of target points  
            source_constraints: List of N constraint dicts, each containing:
                - 'translation_directions': Kx3 array of allowed translation directions
                - 'rotation_axes': Lx3 array of allowed rotation axes
            initial_transform: Initial 4x4 transformation matrix
        """
        if initial_transform is None:
            initial_transform = np.eye(4)
            
        current_transform = initial_transform.copy()
        
        for iteration in range(self.max_iterations):
            # Find correspondences
            
            start =  time.time()
            correspondences = self._find_correspondences(source_points, target_points, current_transform)
            print("correspondences time: ",time.time()-start)
            
            start =  time.time()
            # Optimize transform with constraints
            new_transform = self._optimize_constrained_transform(
                source_points, target_points, source_constraints, 
                correspondences, current_transform
            )
            print("optimise time: ",time.time()-start)
            # Check convergence
            transform_change = np.linalg.norm(new_transform - current_transform)
            print("Transform: ",new_transform)
            if transform_change < self.tolerance:
                print(f"Converged after {iteration+1} iterations")
                break
                
            current_transform = new_transform
            
        return current_transform
    
    def _find_correspondences(self, source_points, target_points, transform):
        """Find nearest neighbor correspondences"""
        # Transform source points
        transformed_source = self._apply_transform(source_points, transform)
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(target_points)
        distances, indices = nbrs.kneighbors(transformed_source)
        
        return indices.flatten()
    
    def _apply_transform(self, points, transform):
        """Apply 4x4 transformation to points"""
        homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (transform @ homogeneous.T).T
        return transformed[:, :3]
    
    def _optimize_constrained_transform(self, source_points, target_points, constraints, correspondences, initial_transform):
        """Optimize transformation with point-wise constraints"""
        
        def residual_function(params):
            # Convert parameters to transformation matrix
            transform = self._params_to_transform(params)
            
            residuals = []
            
            for i, target_idx in enumerate(correspondences):
                constraint = constraints[i]
                p_src = source_points[i]
                p_tgt = target_points[target_idx]
                
                # Compute constrained residual for this point
                point_residuals = self._compute_constrained_residual(
                    p_src, p_tgt, transform, constraint
                )
                residuals.extend(point_residuals)
            
            return np.array(residuals)
        

        
        # Convert initial transform to parameters
        initial_params = self._transform_to_params(initial_transform)
        
        
        # Optimize
        result = least_squares(residual_function, initial_params,method='lm')
        
        # Convert back to transformation matrix
        return self._params_to_transform(result.x)
    
    def _compute_constrained_residual(self, point_src, point_target, transform, constraint):
        """Compute residual for a single point with its constraints"""
        # Transform source point
        p_transformed = self._apply_transform(point_src.reshape(1, -1), transform)[0]
        
        # Translation error
        translation_error = point_target - p_transformed
        
        # Project translation error onto allowed directions
        T_dirs = constraint['translation_directions']
        if len(T_dirs) > 0:
            # Ensure T_dirs is 2D and transpose for proper matrix multiplication
            T_dirs = np.atleast_2d(T_dirs)
            if T_dirs.shape[1] != 3:
                T_dirs = T_dirs.T  # Transpose if needed
            # Project error: each row of T_dirs is a constraint direction
            return T_dirs @ translation_error
        else:
            return np.array([])
        

        
    def _transform_to_params(self, transform):
        """Convert 4x4 transform to 6-parameter representation [tx, ty, tz, rx, ry, rz]"""
        translation = transform[:3, 3]
        rotation_matrix = transform[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        rotation_vec = rotation.as_rotvec()
        
        return np.concatenate([translation, rotation_vec])
    
    def _params_to_transform(self, params):
        """Convert 6-parameter representation to 4x4 transform matrix"""
        translation = params[:3]
        rotation_vec = params[3:6]
        
        # Create rotation matrix from axis-angle
        rotation = R.from_rotvec(rotation_vec)
        rotation_matrix = rotation.as_matrix()
        
        # Build 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        
        return transform


def pca(normals, points, threshold=0.08):
    """
    Calculate constraint directions using PCA analysis
    """
    print("paper approach")
    point_cross_normal = np.cross(points, normals)
    rotation_C = point_cross_normal.T @ point_cross_normal
    rot_eigenvalues, rot_eigenvectors = np.linalg.eigh(rotation_C)
    print(f"Rotation eigenvalue sum: {np.sum(rot_eigenvalues)}")
    
    # Avoid division by zero
    if np.sum(rot_eigenvalues) > 1e-10:
        rot_eigenvalues = rot_eigenvalues / np.sum(rot_eigenvalues)
    else:
        rot_eigenvalues = np.ones_like(rot_eigenvalues) / len(rot_eigenvalues)
    
    
    translation_C = normals.T @ normals
    trans_eigenvalues, trans_eigenvectors = np.linalg.eigh(translation_C)
    
    # Avoid division by zero
    if np.sum(trans_eigenvalues) > 1e-10:
        trans_eigenvalues = trans_eigenvalues / np.sum(trans_eigenvalues)
    else:
        trans_eigenvalues = np.ones_like(trans_eigenvalues) / len(trans_eigenvalues)

    ret = []
    for eig, eigV in [(trans_eigenvalues, trans_eigenvectors), (rot_eigenvalues, rot_eigenvectors)]:
        sorted_indices = np.argsort(eig)
        eig = eig[sorted_indices]
        eigV = eigV[:, sorted_indices]
        scaled_eigenvalues = eig / np.sum(eig)
        
        if scaled_eigenvalues[2] > (1 - threshold):
            # Flat surface (e.g., wall or floor): one dominant direction
            strong_axes = eigV[:, 2:3].T  # Transpose to make it (1, 3)
        elif scaled_eigenvalues[0] < threshold:
            # Linear structure (e.g., column): one weak direction
            strong_axes = eigV[:, 1:3].T  # Transpose to make it (2, 3)
        else:
            # no need to downsample
            strong_axes = eigV.T  # Transpose to make it (3, 3)

        ret += [strong_axes]

    return ret


def compute_normals_for_segment(pcd, radius=0.1, max_nn=30):
    """Compute normals for a point cloud segment"""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return np.asarray(pcd.normals)


def segment_to_constraints(pcd, threshold=0.1):
    """Convert a point cloud segment to constraint directions"""
    points = np.asarray(pcd.points)
    
    # Compute normals if not available
    if not pcd.has_normals():
        normals = compute_normals_for_segment(pcd)
    else:
        normals = np.asarray(pcd.normals)
    
    # Get constraint directions using PCA
    translation_dirs, rotation_axes = pca(normals, points, threshold)
    
    print(f"  - Translation directions shape: {translation_dirs.shape}")
    print(f"  - Rotation axes shape: {rotation_axes.shape}")
    
    # Ensure directions are in the right format (N, 3) where N is number of constraint directions
    if translation_dirs.ndim == 1:
        translation_dirs = translation_dirs.reshape(1, -1)
    if rotation_axes.ndim == 1:
        rotation_axes = rotation_axes.reshape(1, -1)
    
    # Create constraints for each point in the segment
    constraints = []
    for i in range(len(points)):
        constraint = {
            'translation_directions': translation_dirs.copy(),
            'rotation_axes': rotation_axes.copy()
        }
        constraints.append(constraint)
    
    return constraints


def load_and_prepare_data():
    """Load point clouds and prepare constraint data"""
    print("Loading point clouds...")
    
    # Load target point cloud
    pcd_target = o3d.io.read_point_cloud("../data/esbjerg/scans/1741440573289115.pcd")
    pcd_target.remove_non_finite_points()
    # Load source segments
    segments = {
        'cable1': o3d.io.read_point_cloud("icptest/source_mask3d/source_cable1.ply"),
        'cable2': o3d.io.read_point_cloud("icptest/source_mask3d/source_cable2.ply"),
        'ladder': o3d.io.read_point_cloud("icptest/source_mask3d/source_ladder.ply"),
        'cylinder': o3d.io.read_point_cloud("icptest/source_mask3d/source_cylinder.ply")
    }
    
    # Combine all source points and compute constraints
    all_source_points = []
    all_constraints = []
    segment_colors = []
    
    # Color mapping for visualization
    colors = {
        'cable1': [1, 0, 0],    # Red
        'cable2': [0, 1, 0],    # Green  
        'ladder': [0, 0, 1],    # Blue
        'cylinder': [1, 1, 0]   # Yellow
    }
    
    for name, pcd in segments.items():
        print(f"Processing segment: {name}")
        
        # Get points
        points = np.asarray(pcd.points)
        all_source_points.append(points)
        
        # Compute constraints for this segment
        segment_constraints = segment_to_constraints(pcd, threshold=0.1)
        all_constraints.extend(segment_constraints)
        
        # Add colors for visualization
        segment_colors.extend([colors[name]] * len(points))
        
        print(f"  - Points: {len(points)}")
        print(f"  - Translation directions: {segment_constraints[0]['translation_directions'].shape}")
        print(f"  - Rotation axes: {segment_constraints[0]['rotation_axes'].shape}")
    
    # Combine all source points
    source_points = np.vstack(all_source_points)
    
    # Create combined source point cloud for visualization
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    source_pcd.colors = o3d.utility.Vector3dVector(segment_colors)
    
    return source_pcd, pcd_target, source_points, all_constraints


def visualize_alignment(source_pcd, target_pcd, transform=None, title="Point Cloud Alignment"):
    """Visualize the alignment result"""
    # Create visualization
    vis_pcds = []
    
    # Target point cloud (white/gray)
    target_vis = target_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    vis_pcds.append(target_vis)
    
    # Source point cloud (colored by segments)
    if transform is not None:
        source_vis = source_pcd.transform(transform)
    else:
        source_vis = source_pcd
    o3d.io.write_point_cloud("aligned_informed_icp.ply", source_vis, write_ascii=True)
    vis_pcds.append(source_vis)
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis_pcds.append(coord_frame)
    
    # Visualize
    print(f"Showing {title}")
    o3d.visualization.draw_geometries(vis_pcds, window_name=title)


def run_constrained_icp_pipeline():
    """Main pipeline for constrained ICP alignment"""
    # Load and prepare data
    source_pcd, target_pcd, source_points, constraints = load_and_prepare_data()
    target_points = np.asarray(target_pcd.points)
    
    print(f"\nData summary:")
    print(f"Source points: {len(source_points)}")
    print(f"Target points: {len(target_points)}")
    print(f"Constraints: {len(constraints)}")
    
    # Show initial alignment
    visualize_alignment(source_pcd, target_pcd, title="Initial Alignment")
    
    # Run constrained ICP
    print("\nRunning Constrained ICP...")
    icp = ConstrainedICP(max_iterations=50, tolerance=1e-4)
    
    # Initial transform (identity)
    initial_transform = np.eye(4)
    
    # Fit the alignment
    final_transform = icp.fit(source_points, target_points, constraints, initial_transform)
    
    print(f"\nFinal transformation matrix:")
    print(final_transform)
    
    # Show final alignment
    visualize_alignment(source_pcd, target_pcd, final_transform, "Constrained ICP Result")
    
    # Compute alignment error
    transformed_source = icp._apply_transform(source_points, final_transform)
    
    # Find correspondences for error calculation
    nbrs = NearestNeighbors(n_neighbors=1).fit(target_points)
    distances, indices = nbrs.kneighbors(transformed_source)
    
    mean_error = np.mean(distances)
    print(f"\nAlignment statistics:")
    print(f"Mean correspondence distance: {mean_error:.4f}")
    print(f"Max correspondence distance: {np.max(distances):.4f}")
    print(f"Min correspondence distance: {np.min(distances):.4f}")
    
    return final_transform, mean_error
        


if __name__ == "__main__":
    # Run the complete pipeline
    transform, error = run_constrained_icp_pipeline()