"""Tests for K-means clustering."""
import math
import random
import pytest

from robotics_lib.mapping.kmeans import KMeansClustering, generate_clustered_data


class TestKMeansClustering:
    """Tests for the KMeansClustering class."""

    def test_clusters_separable_data(self):
        """Test clustering clearly separable data."""
        random.seed(42)
        centers_x = [0.0, 10.0]
        centers_y = [0.0, 10.0]
        points_x, points_y = generate_clustered_data(
            centers_x, centers_y, n_points_per_cluster=20, spread=2.0
        )

        kmeans = KMeansClustering(n_clusters=2)
        labels = kmeans.fit(points_x, points_y)

        assert len(labels) == len(points_x)
        assert set(labels) <= {0, 1}

    def test_correct_number_of_clusters(self):
        """Test that requested number of clusters is created."""
        random.seed(42)
        centers_x = [0.0, 10.0, 20.0]
        centers_y = [0.0, 10.0, 0.0]
        points_x, points_y = generate_clustered_data(
            centers_x, centers_y, n_points_per_cluster=15, spread=2.0
        )

        kmeans = KMeansClustering(n_clusters=3)
        labels = kmeans.fit(points_x, points_y)

        assert max(labels) <= 2
        assert min(labels) >= 0

    def test_centroids_near_true_centers(self):
        """Test that centroids are near true cluster centers."""
        random.seed(42)
        true_cx = [0.0, 20.0]
        true_cy = [0.0, 20.0]
        points_x, points_y = generate_clustered_data(
            true_cx, true_cy, n_points_per_cluster=50, spread=2.0
        )

        kmeans = KMeansClustering(n_clusters=2)
        kmeans.fit(points_x, points_y)
        centroids_x, centroids_y = kmeans.get_centroids()

        # At least one centroid should be close to each true center
        for tcx, tcy in zip(true_cx, true_cy):
            min_dist = min(
                math.hypot(tcx - cx, tcy - cy)
                for cx, cy in zip(centroids_x, centroids_y)
            )
            assert min_dist < 5.0  # Within 5 units

    def test_convergence(self):
        """Test that algorithm converges."""
        random.seed(42)
        centers_x = [0.0, 10.0]
        centers_y = [0.0, 0.0]
        points_x, points_y = generate_clustered_data(
            centers_x, centers_y, n_points_per_cluster=30, spread=1.0
        )

        kmeans = KMeansClustering(n_clusters=2, max_iterations=100)
        labels = kmeans.fit(points_x, points_y)

        # Should have valid labels
        assert all(0 <= l < 2 for l in labels)

    def test_get_cluster_points(self):
        """Test retrieving points for a specific cluster."""
        random.seed(42)
        points_x = [0, 0, 0, 10, 10, 10]
        points_y = [0, 1, 2, 0, 1, 2]

        kmeans = KMeansClustering(n_clusters=2)
        labels = kmeans.fit(points_x, points_y)

        cluster_0_x, cluster_0_y = kmeans.get_cluster_points(0, points_x, points_y)
        cluster_1_x, cluster_1_y = kmeans.get_cluster_points(1, points_x, points_y)

        # Total points should equal original
        assert len(cluster_0_x) + len(cluster_1_x) == len(points_x)

    def test_single_cluster(self):
        """Test with single cluster."""
        random.seed(42)
        points_x = [1, 2, 3, 4, 5]
        points_y = [1, 2, 3, 4, 5]

        kmeans = KMeansClustering(n_clusters=1)
        labels = kmeans.fit(points_x, points_y)

        # All points should be in cluster 0
        assert all(l == 0 for l in labels)

    def test_empty_cluster_handling(self):
        """Test that algorithm handles potential empty clusters."""
        random.seed(123)
        # Tight cluster - might cause one centroid to be empty initially
        points_x = [0.0, 0.1, 0.2, 0.3, 0.4]
        points_y = [0.0, 0.1, 0.2, 0.3, 0.4]

        kmeans = KMeansClustering(n_clusters=2)
        labels = kmeans.fit(points_x, points_y)

        # Should not crash and return valid labels
        assert len(labels) == len(points_x)


class TestGenerateClusteredData:
    """Tests for data generation function."""

    def test_correct_number_of_points(self):
        """Test that correct number of points is generated."""
        centers_x = [0, 10]
        centers_y = [0, 10]

        points_x, points_y = generate_clustered_data(
            centers_x, centers_y, n_points_per_cluster=20
        )

        assert len(points_x) == 40  # 2 clusters * 20 points
        assert len(points_y) == 40

    def test_reproducible_with_seed(self):
        """Test that seed makes generation reproducible."""
        centers_x = [0, 10]
        centers_y = [0, 10]

        p1_x, p1_y = generate_clustered_data(
            centers_x, centers_y, n_points_per_cluster=10, seed=42
        )
        p2_x, p2_y = generate_clustered_data(
            centers_x, centers_y, n_points_per_cluster=10, seed=42
        )

        assert p1_x == p2_x
        assert p1_y == p2_y

    def test_points_near_centers(self):
        """Test that points are distributed around centers."""
        centers_x = [0.0]
        centers_y = [0.0]

        points_x, points_y = generate_clustered_data(
            centers_x, centers_y, n_points_per_cluster=100, spread=2.0, seed=42
        )

        # Mean should be close to center
        mean_x = sum(points_x) / len(points_x)
        mean_y = sum(points_y) / len(points_y)

        assert abs(mean_x - 0.0) < 1.0
        assert abs(mean_y - 0.0) < 1.0
