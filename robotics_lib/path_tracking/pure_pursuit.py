"""
Pure Pursuit path tracking controller.

Steers a vehicle to follow a path by pursuing a look-ahead point.
"""
import math
import numpy as np


class PurePursuitController:
    """
    Pure Pursuit path tracking controller.

    Uses a look-ahead point to compute steering for following a path.
    """

    def __init__(self, look_ahead_base=2.0, look_ahead_gain=0.1,
                 wheelbase=2.5, max_steer=math.pi/4):
        """
        Initialize the Pure Pursuit controller.

        Parameters
        ----------
        look_ahead_base : float
            Base look-ahead distance
        look_ahead_gain : float
            Velocity-dependent look-ahead gain
        wheelbase : float
            Vehicle wheelbase (distance between axles)
        max_steer : float
            Maximum steering angle (radians)
        """
        self.look_ahead_base = look_ahead_base
        self.look_ahead_gain = look_ahead_gain
        self.wheelbase = wheelbase
        self.max_steer = max_steer

    def compute_steering(self, state, path_x, path_y, last_target_idx=0):
        """
        Compute steering angle to follow the path.

        Parameters
        ----------
        state : VehicleState
            Current vehicle state
        path_x : list
            X coordinates of path waypoints
        path_y : list
            Y coordinates of path waypoints
        last_target_idx : int
            Previous target waypoint index

        Returns
        -------
        tuple
            (steering_angle, target_index)
        """
        look_ahead = self.look_ahead_base + self.look_ahead_gain * abs(state.velocity)
        target_idx = self._find_target_point(state, path_x, path_y, look_ahead, last_target_idx)

        tx = path_x[target_idx]
        ty = path_y[target_idx]

        alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.heading
        steering = math.atan2(2.0 * self.wheelbase * math.sin(alpha), look_ahead)

        steering = np.clip(steering, -self.max_steer, self.max_steer)

        return steering, target_idx

    def _find_target_point(self, state, path_x, path_y, look_ahead, last_idx):
        """Find the target waypoint at look-ahead distance."""
        idx = last_idx
        n_points = len(path_x)

        while idx < n_points - 1:
            dist = self._distance(state.rear_x, state.rear_y, path_x[idx], path_y[idx])
            if dist >= look_ahead:
                break
            idx += 1

        return idx

    def _distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance."""
        return math.hypot(x2 - x1, y2 - y1)


class VehicleState:
    """
    Vehicle state for path tracking.

    Uses bicycle kinematic model.
    """

    def __init__(self, x=0.0, y=0.0, heading=0.0, velocity=0.0, wheelbase=2.5):
        """
        Initialize vehicle state.

        Parameters
        ----------
        x, y : float
            Position coordinates
        heading : float
            Heading angle (yaw) in radians
        velocity : float
            Forward velocity
        wheelbase : float
            Vehicle wheelbase
        """
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = velocity
        self.wheelbase = wheelbase
        self._update_rear_position()

    def update(self, acceleration, steering, dt):
        """
        Update state using bicycle kinematic model.

        Parameters
        ----------
        acceleration : float
            Forward acceleration
        steering : float
            Steering angle
        dt : float
            Time step
        """
        self.x += self.velocity * math.cos(self.heading) * dt
        self.y += self.velocity * math.sin(self.heading) * dt
        self.heading += self.velocity / self.wheelbase * math.tan(steering) * dt
        self.heading = self._normalize_angle(self.heading)
        self.velocity += acceleration * dt
        self._update_rear_position()

    def _update_rear_position(self):
        """Update rear axle position."""
        self.rear_x = self.x - (self.wheelbase / 2) * math.cos(self.heading)
        self.rear_y = self.y - (self.wheelbase / 2) * math.sin(self.heading)

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi)."""
        return (angle + math.pi) % (2 * math.pi) - math.pi


class SpeedController:
    """
    PID speed controller.
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        """
        Initialize speed controller.

        Parameters
        ----------
        kp, ki, kd : float
            PID gains
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target_speed, current_speed, dt=0.1):
        """
        Compute acceleration command.

        Parameters
        ----------
        target_speed : float
            Desired speed
        current_speed : float
            Current speed
        dt : float
            Time step

        Returns
        -------
        float
            Acceleration command
        """
        error = target_speed - current_speed
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        acceleration = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        return acceleration


def simulate_tracking(path_x, path_y, target_speed=5.0, dt=0.1, max_time=100.0):
    """
    Simulate pure pursuit path tracking.

    Parameters
    ----------
    path_x, path_y : list
        Path waypoints
    target_speed : float
        Target velocity
    dt : float
        Time step
    max_time : float
        Maximum simulation time

    Returns
    -------
    tuple
        (trajectory_x, trajectory_y, trajectory_heading, trajectory_v)
    """
    state = VehicleState(x=path_x[0], y=path_y[0], heading=0.0, velocity=0.0)
    pursuit = PurePursuitController()
    speed_ctrl = SpeedController(kp=1.0)

    traj_x, traj_y = [state.x], [state.y]
    traj_heading, traj_v = [state.heading], [state.velocity]

    target_idx = 0
    time = 0.0

    while time < max_time and target_idx < len(path_x) - 1:
        steering, target_idx = pursuit.compute_steering(state, path_x, path_y, target_idx)
        acceleration = speed_ctrl.compute(target_speed, state.velocity, dt)

        state.update(acceleration, steering, dt)
        time += dt

        traj_x.append(state.x)
        traj_y.append(state.y)
        traj_heading.append(state.heading)
        traj_v.append(state.velocity)

    return traj_x, traj_y, traj_heading, traj_v
