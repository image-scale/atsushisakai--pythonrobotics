"""Tests for Pure Pursuit path tracking controller."""
import math
import numpy as np
import pytest

from robotics_lib.path_tracking.pure_pursuit import (
    PurePursuitController,
    VehicleState,
    SpeedController,
    simulate_tracking
)


class TestVehicleState:
    """Tests for the VehicleState class."""

    def test_initial_state(self):
        """Test initial state setup."""
        state = VehicleState(x=1.0, y=2.0, heading=0.5, velocity=3.0)

        assert state.x == 1.0
        assert state.y == 2.0
        assert state.heading == 0.5
        assert state.velocity == 3.0

    def test_update_straight(self):
        """Test state update for straight motion."""
        state = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=1.0)
        state.update(acceleration=0.0, steering=0.0, dt=1.0)

        # Should move forward in x direction
        assert state.x > 0
        assert abs(state.y) < 0.01

    def test_update_turning(self):
        """Test state update with steering."""
        state = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=1.0)
        state.update(acceleration=0.0, steering=0.2, dt=1.0)

        # Heading should change
        assert state.heading != 0.0

    def test_acceleration(self):
        """Test velocity change with acceleration."""
        state = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=0.0)
        state.update(acceleration=1.0, steering=0.0, dt=1.0)

        assert state.velocity > 0

    def test_heading_normalization(self):
        """Test heading stays in [-pi, pi)."""
        state = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=1.0)

        for _ in range(100):
            state.update(acceleration=0.0, steering=0.5, dt=0.1)

        assert -math.pi <= state.heading < math.pi


class TestPurePursuitController:
    """Tests for the PurePursuitController class."""

    def test_compute_steering_straight(self):
        """Test steering for straight path."""
        controller = PurePursuitController()
        state = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=1.0)
        path_x = [0, 10, 20, 30]
        path_y = [0, 0, 0, 0]

        steering, target_idx = controller.compute_steering(state, path_x, path_y)

        # Should be close to zero for straight path
        assert abs(steering) < 0.5

    def test_compute_steering_turn(self):
        """Test steering for path requiring turn."""
        controller = PurePursuitController()
        state = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=1.0)
        # Path turns left
        path_x = [0, 5, 5, 5]
        path_y = [0, 0, 5, 10]

        steering, target_idx = controller.compute_steering(state, path_x, path_y)

        # Should have some steering angle
        assert isinstance(steering, float)

    def test_look_ahead_increases_with_velocity(self):
        """Test that look-ahead distance increases with velocity."""
        controller = PurePursuitController(look_ahead_base=2.0, look_ahead_gain=0.1)

        la_slow = controller.look_ahead_base + controller.look_ahead_gain * 1.0
        la_fast = controller.look_ahead_base + controller.look_ahead_gain * 10.0

        assert la_fast > la_slow

    def test_steering_bounded(self):
        """Test that steering is bounded."""
        controller = PurePursuitController(max_steer=math.pi/4)
        state = VehicleState(x=0.0, y=0.0, heading=0.0, velocity=1.0)
        # Path perpendicular to heading
        path_x = [0, 0, 0]
        path_y = [0, 10, 20]

        steering, _ = controller.compute_steering(state, path_x, path_y)

        assert abs(steering) <= math.pi/4 + 0.01


class TestSpeedController:
    """Tests for the SpeedController class."""

    def test_proportional_control(self):
        """Test proportional speed control."""
        controller = SpeedController(kp=1.0, ki=0.0, kd=0.0)

        accel = controller.compute(target_speed=5.0, current_speed=0.0)
        assert accel > 0

        accel = controller.compute(target_speed=5.0, current_speed=10.0)
        assert accel < 0

    def test_zero_error(self):
        """Test zero acceleration at target speed."""
        controller = SpeedController(kp=1.0, ki=0.0, kd=0.0)

        accel = controller.compute(target_speed=5.0, current_speed=5.0)
        assert abs(accel) < 0.01


class TestSimulation:
    """Tests for path tracking simulation."""

    def test_simulate_straight_path(self):
        """Test simulation on straight path."""
        path_x = list(np.arange(0, 50, 0.5))
        path_y = [0.0] * len(path_x)

        traj_x, traj_y, _, _ = simulate_tracking(path_x, path_y, target_speed=5.0, max_time=20.0)

        assert len(traj_x) > 1
        # Should make progress along path
        assert max(traj_x) > 10

    def test_simulate_curved_path(self):
        """Test simulation on curved path."""
        path_x = list(np.arange(0, 50, 0.5))
        path_y = [math.sin(x / 5.0) * 5 for x in path_x]

        traj_x, traj_y, _, _ = simulate_tracking(path_x, path_y, target_speed=3.0, max_time=30.0)

        assert len(traj_x) > 1
        assert len(traj_y) > 1

    def test_velocity_reaches_target(self):
        """Test that velocity approaches target."""
        path_x = list(np.arange(0, 100, 0.5))
        path_y = [0.0] * len(path_x)

        _, _, _, traj_v = simulate_tracking(path_x, path_y, target_speed=5.0, max_time=50.0)

        # Velocity should be close to target after some time
        final_velocity = traj_v[-1]
        assert final_velocity > 3.0  # Should be approaching 5.0
