"""
Iterative Closest Point (ICP) algorithm for point cloud alignment.

Finds the optimal rigid transformation (rotation + translation) between
two point clouds.
"""
import numpy as np


class PointCloudMatcher:
    """
    ICP point cloud matching for 2D and 3D data.

    Iteratively finds rotation and translation to align
    current points to previous points.
    """

    def __init__(self, tolerance=0.0001, max_iterations=100):
        """
        Initialize the ICP matcher.

        Parameters
        ----------
        tolerance : float
            Convergence tolerance for error reduction
        max_iterations : int
            Maximum number of iterations
        """
        self.tolerance = tolerance
        self.max_iter = max_iterations

    def match(self, previous_points, current_points):
        """
        Find transformation to align current points to previous points.

        Parameters
        ----------
        previous_points : ndarray
            Reference point cloud (D x N where D is dimension)
        current_points : ndarray
            Point cloud to transform (D x N)

        Returns
        -------
        tuple
            (rotation_matrix, translation_vector)
        """
        dim = previous_points.shape[0]
        curr = current_points.copy()
        H = None

        prev_error = float('inf')

        for iteration in range(self.max_iter):
            indices, error = self._find_correspondences(previous_points, curr)
            R, t = self._compute_transformation(previous_points[:, indices], curr)

            curr = R @ curr + t[:, np.newaxis]

            d_error = prev_error - error
            if d_error < 0:
                break

            H = self._update_transform(H, R, t)
            prev_error = error

            if d_error <= self.tolerance:
                break

        if H is None:
            R = np.eye(dim)
            t = np.zeros(dim)
        else:
            R = H[:dim, :dim]
            t = H[:dim, dim]

        return R, t

    def _find_correspondences(self, target, source):
        """Find nearest neighbor correspondences."""
        n_source = source.shape[1]
        n_target = target.shape[1]

        delta = target - source
        error = np.sum(np.linalg.norm(delta, axis=0))

        source_expanded = np.repeat(source, n_target, axis=1)
        target_tiled = np.tile(target, (1, n_source))

        distances = np.linalg.norm(source_expanded - target_tiled, axis=0)
        distances = distances.reshape(n_source, n_target)
        indices = np.argmin(distances, axis=1)

        return indices, error

    def _compute_transformation(self, target, source):
        """
        Compute optimal rotation and translation using SVD.

        This is the core of ICP - finds the rotation R and translation t
        that minimizes ||R*source + t - target||.
        """
        target_mean = np.mean(target, axis=1)
        source_mean = np.mean(source, axis=1)

        target_centered = target - target_mean[:, np.newaxis]
        source_centered = source - source_mean[:, np.newaxis]

        W = source_centered @ target_centered.T
        U, S, Vh = np.linalg.svd(W)

        R = (U @ Vh).T
        t = target_mean - R @ source_mean

        return R, t

    def _update_transform(self, H_prev, R, t):
        """Update cumulative homogeneous transformation."""
        dim = R.shape[0]
        H = np.zeros((dim + 1, dim + 1))
        H[:dim, :dim] = R
        H[:dim, dim] = t
        H[dim, dim] = 1.0

        if H_prev is None:
            return H
        return H_prev @ H


def transform_points(points, rotation, translation):
    """
    Apply rigid transformation to points.

    Parameters
    ----------
    points : ndarray
        Point cloud (D x N)
    rotation : ndarray
        Rotation matrix (D x D)
    translation : ndarray
        Translation vector (D,)

    Returns
    -------
    ndarray
        Transformed points (D x N)
    """
    return rotation @ points + translation[:, np.newaxis]


def generate_test_point_cloud(n_points=100, field_size=50.0, seed=None):
    """
    Generate a random 2D point cloud.

    Parameters
    ----------
    n_points : int
        Number of points
    field_size : float
        Size of the field
    seed : int, optional
        Random seed

    Returns
    -------
    ndarray
        Point cloud (2 x N)
    """
    if seed is not None:
        np.random.seed(seed)
    px = (np.random.rand(n_points) - 0.5) * field_size
    py = (np.random.rand(n_points) - 0.5) * field_size
    return np.vstack((px, py))
