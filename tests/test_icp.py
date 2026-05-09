"""Tests for ICP (Iterative Closest Point) matching."""
import math
import numpy as np
import pytest

from robotics_lib.slam.icp import (
    PointCloudMatcher,
    transform_points,
    generate_test_point_cloud
)


class TestPointCloudMatcher:
    """Tests for the PointCloudMatcher class."""

    def test_identity_transformation(self):
        """Test matching identical point clouds."""
        np.random.seed(42)
        points = generate_test_point_cloud(n_points=50)

        matcher = PointCloudMatcher()
        R, t = matcher.match(points, points.copy())

        # Should be close to identity transformation
        assert np.allclose(R, np.eye(2), atol=0.1)
        assert np.allclose(t, np.zeros(2), atol=0.1)

    def test_translation_only(self):
        """Test matching with pure translation."""
        np.random.seed(42)
        prev_points = generate_test_point_cloud(n_points=50)
        translation = np.array([2.0, 3.0])
        curr_points = prev_points + translation[:, np.newaxis]

        matcher = PointCloudMatcher()
        R, t = matcher.match(prev_points, curr_points)

        # Apply transformation to current points
        aligned = R @ curr_points + t[:, np.newaxis]

        # Should align with previous points
        error = np.mean(np.linalg.norm(aligned - prev_points, axis=0))
        assert error < 1.0

    def test_rotation_only(self):
        """Test matching with pure rotation."""
        np.random.seed(42)
        prev_points = generate_test_point_cloud(n_points=50)
        angle = np.deg2rad(15)
        R_true = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        curr_points = R_true @ prev_points

        matcher = PointCloudMatcher()
        R, t = matcher.match(prev_points, curr_points)

        # Apply transformation
        aligned = R @ curr_points + t[:, np.newaxis]
        error = np.mean(np.linalg.norm(aligned - prev_points, axis=0))
        assert error < 1.0

    def test_rotation_and_translation(self):
        """Test matching with rotation and translation."""
        np.random.seed(42)
        prev_points = generate_test_point_cloud(n_points=100)

        angle = np.deg2rad(-10)
        R_true = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        t_true = np.array([0.5, 2.0])
        curr_points = R_true @ prev_points + t_true[:, np.newaxis]

        matcher = PointCloudMatcher()
        R, t = matcher.match(prev_points, curr_points)

        aligned = R @ curr_points + t[:, np.newaxis]
        error = np.mean(np.linalg.norm(aligned - prev_points, axis=0))
        assert error < 2.0

    def test_convergence(self):
        """Test that ICP converges."""
        np.random.seed(42)
        prev_points = generate_test_point_cloud(n_points=50)
        translation = np.array([1.0, 1.0])
        curr_points = prev_points + translation[:, np.newaxis]

        matcher = PointCloudMatcher(tolerance=0.001, max_iterations=50)
        R, t = matcher.match(prev_points, curr_points)

        # Should return valid transformation
        assert R.shape == (2, 2)
        assert t.shape == (2,)

    def test_svd_transformation(self):
        """Test SVD-based transformation computation."""
        np.random.seed(42)
        matcher = PointCloudMatcher()

        # Create known correspondence
        source = np.array([[0, 1, 2], [0, 0, 0]], dtype=float)
        angle = np.pi / 4
        R_true = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        t_true = np.array([1.0, 2.0])
        target = R_true @ source + t_true[:, np.newaxis]

        R, t = matcher._compute_transformation(target, source)

        # Verify transformation
        reconstructed = R @ source + t[:, np.newaxis]
        error = np.linalg.norm(reconstructed - target)
        assert error < 0.01


class TestHelperFunctions:
    """Tests for ICP helper functions."""

    def test_transform_points(self):
        """Test point transformation function."""
        points = np.array([[1, 2], [0, 0]], dtype=float)
        R = np.array([[0, -1], [1, 0]], dtype=float)  # 90 degree rotation
        t = np.array([1.0, 1.0])

        transformed = transform_points(points, R, t)

        # (1, 0) -> (0, 1) -> (1, 2) after rotation + translation
        assert np.isclose(transformed[0, 0], 1.0)
        assert np.isclose(transformed[1, 0], 2.0)

    def test_generate_point_cloud(self):
        """Test point cloud generation."""
        points = generate_test_point_cloud(n_points=100, field_size=10.0, seed=42)

        assert points.shape == (2, 100)
        assert np.all(points >= -5.0)
        assert np.all(points <= 5.0)

    def test_reproducible_generation(self):
        """Test that seed makes generation reproducible."""
        p1 = generate_test_point_cloud(n_points=50, seed=123)
        p2 = generate_test_point_cloud(n_points=50, seed=123)

        assert np.allclose(p1, p2)
