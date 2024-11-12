import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

# Function to perform Moving Least Squares smoothing for a single point
def mls_smooth_point(i, points, nbrs, n_neighbors):
    # print(i)
    point = points[i]
    distances, indices = nbrs.radius_neighbors([point])
    neighbors = points[indices[0]]
    
    # Compute the centroid of the neighbors
    centroid = np.mean(neighbors, axis=0)
    
    # Compute the covariance matrix of the neighbors
    cov_matrix = np.cov(neighbors - centroid, rowvar=False)
    
    # Compute the eigenvectors and eigenvalues of the covariance matrix
    _, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # The normal vector is the eigenvector corresponding to the smallest eigenvalue
    normal = eigenvectors[:, 0]
    
    # Project the point onto the plane defined by the neighbors
    point_on_plane = point - np.dot(point - centroid, normal) * normal
    
    return point_on_plane

# Function to perform MLS smoothing in parallel
def mls_smooth_parallel(pcd, radius=0.01, n_neighbors=30, n_jobs=-1):
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    
    # Create a NearestNeighbors object to find neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=radius)
    nbrs.fit(points)
    
    # Parallelize the MLS smoothing using joblib
    smoothed_points = Parallel(n_jobs=n_jobs)(
        delayed(mls_smooth_point)(i, points, nbrs, n_neighbors) for i in range(len(points))
    )
    
    # Convert the smoothed points back into a point cloud
    smoothed_pcd = o3d.geometry.PointCloud()
    smoothed_pcd.points = o3d.utility.Vector3dVector(np.array(smoothed_points))
    
    return smoothed_pcd

def estimate_normals_neighborhood_reconstruction(pcd, radius=0.5):
    # Create a KDTree for the point cloud
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Placeholder for normals
    normals = []

    # Iterate through each point in the point cloud
    for i in range(len(pcd.points)):
        # Find neighbors within the specified radius
        [_, idx, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if len(idx) < 3:
            normals.append([0, 0, 1])  # Default normal if less than 3 neighbors
            continue

        # Extract the neighbor points and get eigenvalues/vectors
        neighbor_points = np.asarray(pcd.points)[idx, :]
        covariance_matrix = np.cov(neighbor_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Normal is the eigenvector corresponding to the smallest eigenvalue
        normal = eigenvectors[:, np.argmin(eigenvalues)]

        # Ensure normal direction consistency
        if np.dot(normal, pcd.points[i]) < 0:
            normal = -normal

        normals.append(normal)
    normals = np.array(normals)
    pcd.normals = o3d.utility.Vector3dVector(normals)


# Load the point cloud
pcd = o3d.io.read_point_cloud("./_output/pcl.pcd")
print("Number of points in the point cloud:", len(pcd.points))


# Assume the original point cloud has color information
# Downsample the point cloud to create a reduced point cloud (e.g., using voxel downsampling)
voxel_size = 0.001  # Adjust voxel size for desired level of reduction
downsampled_pcd = pcd.voxel_down_sample(voxel_size)

# Convert the point clouds to numpy arrays
original_points = np.asarray(pcd.points)
reduced_points = np.asarray(downsampled_pcd.points)

# Get the colors from the original point cloud
original_colors = np.asarray(pcd.colors)

# Use NearestNeighbors to find the closest points between the original and reduced point clouds
nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(original_points)
distances, indices = nbrs.kneighbors(reduced_points)

# Assign the color of the closest point in the original point cloud to the reduced point cloud
reduced_colors = original_colors[indices.flatten()]

# Assign the colors to the downsampled point cloud
downsampled_pcd.colors = o3d.utility.Vector3dVector(reduced_colors)

# Perform MLS smoothing with parallelization
smoothed_pcd = mls_smooth_parallel(downsampled_pcd, radius=0.017, n_neighbors=120, n_jobs=-1)

smoothed_pcd.colors = downsampled_pcd.colors

print("Number of points in the point cloud:", len(smoothed_pcd.points))

# smoothed_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=500))
estimate_normals_neighborhood_reconstruction(smoothed_pcd, radius=0.05)

# Create the Poisson surface reconstruction and filter out any non dense areas
poisson_mesh, poisson_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(smoothed_pcd, depth=11)
# Convert densities to a NumPy array for easier manipulation
densities = np.asarray(poisson_densities)
threshold = np.percentile(densities, 5)
triangles_to_keep = densities > threshold
filtered_mesh = poisson_mesh.select_by_index(np.where(triangles_to_keep)[0])

coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
o3d.visualization.draw_geometries([filtered_mesh, coordinate_frame])



