"""
K-means clustering algorithm for object clustering.

Groups points into K clusters based on distance to cluster centroids.
"""
import math
import random


class KMeansClustering:
    """
    K-means clustering for 2D point data.

    Iteratively assigns points to clusters and updates centroids
    until convergence.
    """

    def __init__(self, n_clusters, max_iterations=100, convergence_threshold=0.1):
        """
        Initialize K-means clustering.

        Parameters
        ----------
        n_clusters : int
            Number of clusters
        max_iterations : int
            Maximum number of iterations
        convergence_threshold : float
            Convergence threshold for cost change
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iterations
        self.threshold = convergence_threshold

        self.centroids_x = []
        self.centroids_y = []
        self.labels = []

    def fit(self, points_x, points_y):
        """
        Cluster the given points.

        Parameters
        ----------
        points_x : list
            X coordinates of points
        points_y : list
            Y coordinates of points

        Returns
        -------
        list
            Cluster labels for each point
        """
        n_points = len(points_x)
        self.labels = [random.randint(0, self.n_clusters - 1) for _ in range(n_points)]
        self.centroids_x = [0.0] * self.n_clusters
        self.centroids_y = [0.0] * self.n_clusters

        self._update_centroids(points_x, points_y)

        prev_cost = float('inf')

        for _ in range(self.max_iter):
            cost = self._assign_clusters(points_x, points_y)
            self._update_centroids(points_x, points_y)

            d_cost = abs(cost - prev_cost)
            if d_cost < self.threshold:
                break
            prev_cost = cost

        return self.labels

    def _assign_clusters(self, points_x, points_y):
        """Assign each point to nearest centroid."""
        total_cost = 0.0

        for i in range(len(points_x)):
            px, py = points_x[i], points_y[i]

            distances = [
                math.hypot(px - cx, py - cy)
                for cx, cy in zip(self.centroids_x, self.centroids_y)
            ]

            min_dist = min(distances)
            min_idx = distances.index(min_dist)

            self.labels[i] = min_idx
            total_cost += min_dist

        return total_cost

    def _update_centroids(self, points_x, points_y):
        """Update centroid positions as mean of assigned points."""
        for k in range(self.n_clusters):
            cluster_x = [points_x[i] for i in range(len(points_x)) if self.labels[i] == k]
            cluster_y = [points_y[i] for i in range(len(points_y)) if self.labels[i] == k]

            if len(cluster_x) > 0:
                self.centroids_x[k] = sum(cluster_x) / len(cluster_x)
                self.centroids_y[k] = sum(cluster_y) / len(cluster_y)

    def get_centroids(self):
        """
        Get cluster centroids.

        Returns
        -------
        tuple
            (centroids_x, centroids_y)
        """
        return self.centroids_x, self.centroids_y

    def get_cluster_points(self, cluster_id, points_x, points_y):
        """
        Get points belonging to a specific cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster index
        points_x, points_y : list
            Original points

        Returns
        -------
        tuple
            (cluster_points_x, cluster_points_y)
        """
        cx = [points_x[i] for i in range(len(points_x)) if self.labels[i] == cluster_id]
        cy = [points_y[i] for i in range(len(points_y)) if self.labels[i] == cluster_id]
        return cx, cy


def generate_clustered_data(center_x, center_y, n_points_per_cluster, spread=3.0, seed=None):
    """
    Generate synthetic clustered data.

    Parameters
    ----------
    center_x : list
        X coordinates of cluster centers
    center_y : list
        Y coordinates of cluster centers
    n_points_per_cluster : int
        Number of points per cluster
    spread : float
        Random spread around centers
    seed : int, optional
        Random seed

    Returns
    -------
    tuple
        (all_points_x, all_points_y)
    """
    if seed is not None:
        random.seed(seed)

    all_x, all_y = [], []

    for cx, cy in zip(center_x, center_y):
        for _ in range(n_points_per_cluster):
            all_x.append(cx + spread * (random.random() - 0.5))
            all_y.append(cy + spread * (random.random() - 0.5))

    return all_x, all_y
