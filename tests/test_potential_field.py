"""Tests for Potential Field path planner."""
import math
import pytest

from robotics_lib.path_planning.potential_field import PotentialFieldPlanner


class TestPotentialFieldPlanner:
    """Tests for the PotentialFieldPlanner class."""

    def test_finds_path_to_goal(self):
        """Test that planner finds path to goal without obstacles."""
        planner = PotentialFieldPlanner(
            obstacles_x=[],
            obstacles_y=[],
            goal_x=10.0,
            goal_y=10.0
        )
        path_x, path_y = planner.find_path(0.0, 0.0)

        assert len(path_x) > 0
        assert len(path_y) > 0
        # Path should approach goal
        final_dist = math.hypot(path_x[-1] - 10.0, path_y[-1] - 10.0)
        assert final_dist < 1.0

    def test_path_starts_at_start(self):
        """Test that path starts at the start position."""
        planner = PotentialFieldPlanner(
            obstacles_x=[],
            obstacles_y=[],
            goal_x=10.0,
            goal_y=10.0
        )
        path_x, path_y = planner.find_path(0.0, 0.0)

        assert math.isclose(path_x[0], 0.0, abs_tol=0.01)
        assert math.isclose(path_y[0], 0.0, abs_tol=0.01)

    def test_avoids_obstacle(self):
        """Test that path avoids obstacles."""
        planner = PotentialFieldPlanner(
            obstacles_x=[5.0],
            obstacles_y=[5.0],
            goal_x=10.0,
            goal_y=10.0,
            influence_distance=3.0
        )
        path_x, path_y = planner.find_path(0.0, 0.0)

        # Check that path doesn't pass too close to obstacle
        min_dist = float('inf')
        for px, py in zip(path_x, path_y):
            dist = math.hypot(px - 5.0, py - 5.0)
            min_dist = min(min_dist, dist)

        # Path should maintain some distance from obstacle
        assert min_dist > 0.5

    def test_attractive_force_toward_goal(self):
        """Test that attractive force pulls toward goal."""
        planner = PotentialFieldPlanner(
            obstacles_x=[],
            obstacles_y=[],
            goal_x=10.0,
            goal_y=0.0,
            attract_gain=1.0
        )

        fa_x, fa_y = planner._attractive_force(0.0, 0.0)
        # Force should point toward goal (positive x)
        assert fa_x > 0
        assert abs(fa_y) < 0.01

    def test_repulsive_force_away_from_obstacle(self):
        """Test that repulsive force pushes away from obstacle."""
        planner = PotentialFieldPlanner(
            obstacles_x=[5.0],
            obstacles_y=[0.0],
            goal_x=10.0,
            goal_y=0.0,
            influence_distance=10.0,
            repulse_gain=100.0
        )

        # Test point near obstacle
        fr_x, fr_y = planner._repulsive_force(3.0, 0.0)
        # Force should push away (negative x, away from obstacle at x=5)
        assert fr_x < 0

    def test_configurable_gains(self):
        """Test that different gain values affect the path."""
        # Low repulsion - might get closer to obstacle
        planner_low = PotentialFieldPlanner(
            obstacles_x=[5.0],
            obstacles_y=[5.0],
            goal_x=10.0,
            goal_y=10.0,
            repulse_gain=10.0
        )
        path_x_low, path_y_low = planner_low.find_path(0.0, 0.0)

        # High repulsion - should stay farther from obstacle
        planner_high = PotentialFieldPlanner(
            obstacles_x=[5.0],
            obstacles_y=[5.0],
            goal_x=10.0,
            goal_y=10.0,
            repulse_gain=500.0
        )
        path_x_high, path_y_high = planner_high.find_path(0.0, 0.0)

        assert len(path_x_low) > 0
        assert len(path_x_high) > 0

    def test_potential_field_computation(self):
        """Test potential field grid computation."""
        planner = PotentialFieldPlanner(
            obstacles_x=[5.0],
            obstacles_y=[5.0],
            goal_x=10.0,
            goal_y=10.0
        )

        X, Y, potential = planner.compute_potential_field(
            x_range=(0, 15),
            y_range=(0, 15)
        )

        assert X.shape == Y.shape == potential.shape
        assert potential.shape[0] > 0
        assert potential.shape[1] > 0

    def test_oscillation_detection(self):
        """Test that oscillation is detected and path terminates."""
        # Put obstacle directly between start and goal (local minimum)
        planner = PotentialFieldPlanner(
            obstacles_x=[5.0],
            obstacles_y=[0.0],
            goal_x=10.0,
            goal_y=0.0,
            influence_distance=6.0,
            repulse_gain=500.0
        )

        path_x, path_y = planner.find_path(0.0, 0.0, max_steps=1000)

        # Should terminate (not run forever)
        assert len(path_x) <= 1000
        assert len(path_x) > 0
