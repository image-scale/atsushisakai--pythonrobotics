"""Tests for Extended Kalman Filter localization."""
import math
import numpy as np
import pytest

from robotics_lib.localization.ekf import (
    ExtendedKalmanFilter,
    simulate_robot_motion,
    generate_observation
)


class TestExtendedKalmanFilter:
    """Tests for the ExtendedKalmanFilter class."""

    def test_prediction_step(self):
        """Test that prediction step updates state correctly."""
        ekf = ExtendedKalmanFilter(dt=0.1)
        state = np.zeros((4, 1))
        cov = np.eye(4)
        control = np.array([[1.0], [0.0]])  # v=1, yaw_rate=0

        x_pred, P_pred = ekf.predict(state, cov, control)

        # Robot should move forward in x direction
        assert x_pred[0, 0] > 0
        assert P_pred.shape == (4, 4)

    def test_update_step(self):
        """Test that update step corrects state estimate."""
        ekf = ExtendedKalmanFilter(dt=0.1)
        state = np.zeros((4, 1))
        cov = np.eye(4) * 10  # High uncertainty

        # Observation says robot is at (1, 1)
        observation = np.array([[1.0], [1.0]])

        x_upd, P_upd = ekf.update(state, cov, observation)

        # State should move toward observation
        assert x_upd[0, 0] > 0
        assert x_upd[1, 0] > 0
        # Uncertainty should decrease
        assert P_upd[0, 0] < cov[0, 0]
        assert P_upd[1, 1] < cov[1, 1]

    def test_full_estimation(self):
        """Test full predict-update cycle."""
        ekf = ExtendedKalmanFilter(dt=0.1)
        state = np.zeros((4, 1))
        cov = np.eye(4)
        control = np.array([[1.0], [0.1]])
        observation = np.array([[0.1], [0.0]])

        x_est, P_est = ekf.estimate(state, cov, control, observation)

        assert x_est.shape == (4, 1)
        assert P_est.shape == (4, 4)

    def test_simulation_tracking(self):
        """Test EKF tracking simulated robot motion."""
        np.random.seed(42)
        ekf = ExtendedKalmanFilter(dt=0.1)

        x_true = np.zeros((4, 1))
        x_est = np.zeros((4, 1))
        P_est = np.eye(4)

        for _ in range(100):
            control = np.array([[1.0], [0.1]])
            x_true = simulate_robot_motion(x_true, control, 0.1)
            observation = generate_observation(x_true, gps_noise_std=0.5)
            x_est, P_est = ekf.estimate(x_est, P_est, control, observation)

        # Estimate should be close to true state
        error = np.linalg.norm(x_est[:2] - x_true[:2])
        assert error < 2.0  # Within 2 meters

    def test_motion_jacobian(self):
        """Test that motion Jacobian has correct shape."""
        ekf = ExtendedKalmanFilter()
        state = np.array([[1.0], [2.0], [0.5], [1.0]])
        control = np.array([[1.0], [0.1]])

        J_f = ekf._motion_jacobian(state, control)

        assert J_f.shape == (4, 4)
        # Should be approximately identity plus perturbations
        assert np.isclose(J_f[0, 0], 1.0)
        assert np.isclose(J_f[1, 1], 1.0)

    def test_observation_jacobian(self):
        """Test that observation Jacobian has correct shape and values."""
        ekf = ExtendedKalmanFilter()
        J_h = ekf._observation_jacobian()

        assert J_h.shape == (2, 4)
        # H extracts x and y
        assert J_h[0, 0] == 1.0
        assert J_h[1, 1] == 1.0

    def test_custom_noise_parameters(self):
        """Test EKF with custom noise parameters."""
        Q = np.diag([0.5, 0.5, 0.1, 0.5]) ** 2
        R = np.diag([2.0, 2.0]) ** 2

        ekf = ExtendedKalmanFilter(dt=0.1, process_noise=Q, observation_noise=R)

        assert np.allclose(ekf.Q, Q)
        assert np.allclose(ekf.R, R)


class TestSimulationHelpers:
    """Tests for simulation helper functions."""

    def test_simulate_robot_motion(self):
        """Test robot motion simulation."""
        x = np.array([[0.0], [0.0], [0.0], [0.0]])
        control = np.array([[1.0], [0.0]])

        x_new = simulate_robot_motion(x, control, 0.1)

        # Robot should move in x direction
        assert x_new[0, 0] > 0
        assert np.isclose(x_new[1, 0], 0.0, atol=1e-6)

    def test_generate_observation(self):
        """Test observation generation."""
        np.random.seed(42)
        x = np.array([[5.0], [10.0], [0.5], [1.0]])

        observation = generate_observation(x, gps_noise_std=0.0)

        # With no noise, should match position
        assert np.isclose(observation[0, 0], 5.0)
        assert np.isclose(observation[1, 0], 10.0)

    def test_noisy_observation(self):
        """Test that observation has noise."""
        np.random.seed(42)
        x = np.array([[5.0], [10.0], [0.5], [1.0]])

        observations = [generate_observation(x, gps_noise_std=1.0) for _ in range(100)]
        obs_array = np.array(observations)

        # Mean should be close to true position
        mean_x = np.mean(obs_array[:, 0])
        mean_y = np.mean(obs_array[:, 1])
        assert abs(mean_x - 5.0) < 0.5
        assert abs(mean_y - 10.0) < 0.5
