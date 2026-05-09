"""Tests for angle utilities."""
import numpy as np
import pytest

from robotics_lib.utils.angle import normalize_angle, rotation_matrix_2d


class TestNormalizeAngle:
    """Tests for the normalize_angle function."""

    def test_negative_angle_wraps_to_positive(self):
        result = normalize_angle(-4.0)
        expected = 2 * np.pi - 4.0  # approximately 2.283
        assert np.isclose(result, expected, atol=1e-6)

    def test_pi_boundary(self):
        result = normalize_angle(np.pi)
        # pi wraps to -pi in [-pi, pi) range
        assert np.isclose(abs(result), np.pi, atol=1e-10)

    def test_zero_unchanged(self):
        result = normalize_angle(0.0)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_degrees_mode(self):
        result = normalize_angle([370.0], degrees=True)
        expected = np.array([10.0])
        assert np.allclose(result, expected, atol=1e-6)

    def test_negative_to_positive_range(self):
        result = normalize_angle(-60.0, positive_only=True, degrees=True)
        assert np.isclose(result, 300.0, atol=1e-6)

    def test_array_input(self):
        angles = np.array([-150.0, 190.0, 350.0])
        result = normalize_angle(angles, degrees=True)
        expected = np.array([-150.0, -170.0, -10.0])
        assert np.allclose(result, expected, atol=1e-6)

    def test_positive_only_with_array(self):
        result = normalize_angle([90.0, -90.0], positive_only=True, degrees=True)
        expected = np.array([90.0, 270.0])
        assert np.allclose(result, expected, atol=1e-6)


class TestRotationMatrix2D:
    """Tests for the rotation_matrix_2d function."""

    def test_identity_at_zero(self):
        R = rotation_matrix_2d(0.0)
        expected = np.eye(2)
        assert np.allclose(R, expected, atol=1e-10)

    def test_90_degree_rotation(self):
        R = rotation_matrix_2d(np.pi / 2)
        # (1, 0) should rotate to (0, 1)
        point = np.array([1.0, 0.0])
        rotated = R @ point
        expected = np.array([0.0, 1.0])
        assert np.allclose(rotated, expected, atol=1e-10)

    def test_180_degree_rotation(self):
        R = rotation_matrix_2d(np.pi)
        point = np.array([1.0, 0.0])
        rotated = R @ point
        expected = np.array([-1.0, 0.0])
        assert np.allclose(rotated, expected, atol=1e-10)

    def test_orthogonal_matrix(self):
        R = rotation_matrix_2d(0.5)
        # R * R^T should equal identity
        assert np.allclose(R @ R.T, np.eye(2), atol=1e-10)

    def test_determinant_is_one(self):
        R = rotation_matrix_2d(1.23)
        det = np.linalg.det(R)
        assert np.isclose(det, 1.0, atol=1e-10)
