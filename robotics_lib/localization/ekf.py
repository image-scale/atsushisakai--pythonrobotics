"""
Extended Kalman Filter for robot localization.

Uses linearized motion and observation models for state estimation.
"""
import math
import numpy as np


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for robot localization.

    State vector: [x, y, yaw, velocity]
    """

    def __init__(self, dt=0.1, process_noise=None, observation_noise=None):
        """
        Initialize the EKF.

        Parameters
        ----------
        dt : float
            Time step for prediction
        process_noise : ndarray, optional
            4x4 process noise covariance Q
        observation_noise : ndarray, optional
            2x2 observation noise covariance R
        """
        self.dt = dt

        if process_noise is None:
            self.Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
        else:
            self.Q = process_noise

        if observation_noise is None:
            self.R = np.diag([1.0, 1.0]) ** 2
        else:
            self.R = observation_noise

    def predict(self, state, covariance, control_input):
        """
        Predict step of the EKF.

        Parameters
        ----------
        state : ndarray
            Current state [x, y, yaw, v] (4x1)
        covariance : ndarray
            Current covariance matrix (4x4)
        control_input : ndarray
            Control input [v, yaw_rate] (2x1)

        Returns
        -------
        tuple
            (predicted_state, predicted_covariance)
        """
        x_pred = self._motion_model(state, control_input)
        J_f = self._motion_jacobian(state, control_input)
        P_pred = J_f @ covariance @ J_f.T + self.Q

        return x_pred, P_pred

    def update(self, state, covariance, observation):
        """
        Update step of the EKF.

        Parameters
        ----------
        state : ndarray
            Predicted state [x, y, yaw, v] (4x1)
        covariance : ndarray
            Predicted covariance matrix (4x4)
        observation : ndarray
            Observation [x, y] (2x1)

        Returns
        -------
        tuple
            (updated_state, updated_covariance)
        """
        J_h = self._observation_jacobian()
        z_pred = self._observation_model(state)

        innovation = observation - z_pred
        S = J_h @ covariance @ J_h.T + self.R
        K = covariance @ J_h.T @ np.linalg.inv(S)

        x_upd = state + K @ innovation
        P_upd = (np.eye(4) - K @ J_h) @ covariance

        return x_upd, P_upd

    def estimate(self, state, covariance, control_input, observation):
        """
        Perform full EKF step (predict + update).

        Parameters
        ----------
        state : ndarray
            Current state [x, y, yaw, v] (4x1)
        covariance : ndarray
            Current covariance matrix (4x4)
        control_input : ndarray
            Control input [v, yaw_rate] (2x1)
        observation : ndarray
            Observation [x, y] (2x1)

        Returns
        -------
        tuple
            (estimated_state, estimated_covariance)
        """
        x_pred, P_pred = self.predict(state, covariance, control_input)
        x_est, P_est = self.update(x_pred, P_pred, observation)
        return x_est, P_est

    def _motion_model(self, state, control):
        """Apply motion model to predict next state."""
        x, y, yaw, v = state.flatten()
        v_cmd, yaw_rate = control.flatten()

        F = np.array([
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 0]
        ])

        B = np.array([
            [self.dt * math.cos(yaw), 0],
            [self.dt * math.sin(yaw), 0],
            [0, self.dt],
            [1.0, 0]
        ])

        x_new = F @ state + B @ control
        return x_new

    def _motion_jacobian(self, state, control):
        """Compute Jacobian of motion model."""
        yaw = state[2, 0]
        v = control[0, 0]

        J_f = np.array([
            [1.0, 0, -self.dt * v * math.sin(yaw), self.dt * math.cos(yaw)],
            [0, 1.0, self.dt * v * math.cos(yaw), self.dt * math.sin(yaw)],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0]
        ])
        return J_f

    def _observation_model(self, state):
        """Apply observation model to get expected observation."""
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return H @ state

    def _observation_jacobian(self):
        """Compute Jacobian of observation model."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])


def simulate_robot_motion(x_true, control, dt):
    """
    Simulate true robot motion.

    Parameters
    ----------
    x_true : ndarray
        True state [x, y, yaw, v]
    control : ndarray
        Control input [v, yaw_rate]
    dt : float
        Time step

    Returns
    -------
    ndarray
        New true state
    """
    yaw = x_true[2, 0]

    F = np.array([
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 0]
    ])

    B = np.array([
        [dt * math.cos(yaw), 0],
        [dt * math.sin(yaw), 0],
        [0, dt],
        [1.0, 0]
    ])

    return F @ x_true + B @ control


def generate_observation(x_true, gps_noise_std=0.5):
    """
    Generate noisy GPS observation.

    Parameters
    ----------
    x_true : ndarray
        True state
    gps_noise_std : float
        Standard deviation of GPS noise

    Returns
    -------
    ndarray
        Noisy observation [x, y]
    """
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    noise = gps_noise_std * np.random.randn(2, 1)
    return H @ x_true + noise
