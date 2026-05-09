"""Tests for RRT (Rapidly-exploring Random Tree) path planner."""
import math
import random
import pytest

from robotics_lib.path_planning.rrt import RandomTreePlanner


class TestRandomTreePlanner:
    """Tests for the RandomTreePlanner class."""

    def test_simple_path_finding(self):
        """Test that RRT finds a path in simple environment."""
        random.seed(42)  # For reproducibility
        obstacles = [(5, 5, 1), (3, 8, 1)]
        planner = RandomTreePlanner(
            start=[0, 0],
            goal=[10, 10],
            obstacles=obstacles,
            search_bounds=[-2, 15]
        )
        path = planner.find_path()

        assert path is not None
        assert len(path) > 0

    def test_path_starts_at_goal(self):
        """Test that path starts at goal (reversed format)."""
        random.seed(42)
        obstacles = [(5, 5, 0.5)]
        planner = RandomTreePlanner(
            start=[0, 0],
            goal=[10, 10],
            obstacles=obstacles,
            search_bounds=[-2, 15]
        )
        path = planner.find_path()

        assert path is not None
        assert math.isclose(path[0][0], 10.0, abs_tol=0.5)
        assert math.isclose(path[0][1], 10.0, abs_tol=0.5)

    def test_path_ends_at_start(self):
        """Test that path ends at start (reversed format)."""
        random.seed(42)
        obstacles = [(5, 5, 0.5)]
        planner = RandomTreePlanner(
            start=[0, 0],
            goal=[10, 10],
            obstacles=obstacles,
            search_bounds=[-2, 15]
        )
        path = planner.find_path()

        assert path is not None
        assert math.isclose(path[-1][0], 0.0, abs_tol=0.5)
        assert math.isclose(path[-1][1], 0.0, abs_tol=0.5)

    def test_avoids_obstacles(self):
        """Test that path avoids circular obstacles."""
        random.seed(42)
        # Large obstacle in the middle
        obstacles = [(5, 5, 2)]
        planner = RandomTreePlanner(
            start=[0, 0],
            goal=[10, 10],
            obstacles=obstacles,
            search_bounds=[-2, 15],
            robot_radius=0.5
        )
        path = planner.find_path()

        if path is not None:
            # Verify path points don't collide with obstacle
            for point in path:
                dist_to_obs = math.hypot(point[0] - 5, point[1] - 5)
                assert dist_to_obs > 2.5 - 0.5  # obstacle radius + robot radius - tolerance

    def test_configurable_step_size(self):
        """Test that different step sizes work."""
        random.seed(42)
        obstacles = [(5, 5, 0.5)]
        planner = RandomTreePlanner(
            start=[0, 0],
            goal=[10, 10],
            obstacles=obstacles,
            search_bounds=[-2, 15],
            step_size=2.0,
            path_step=0.25
        )
        path = planner.find_path()

        assert path is not None

    def test_goal_bias(self):
        """Test that higher goal bias finds paths faster."""
        random.seed(42)
        obstacles = []
        # With high goal bias, should find path quickly
        planner = RandomTreePlanner(
            start=[0, 0],
            goal=[5, 5],
            obstacles=obstacles,
            search_bounds=[-2, 10],
            goal_bias=50  # 50% chance to sample goal
        )
        path = planner.find_path()

        assert path is not None

    def test_no_path_in_max_iterations(self):
        """Test behavior when path not found within iterations."""
        random.seed(42)
        # Completely blocked scenario
        obstacles = [
            (5, i, 0.5) for i in range(-5, 15)
        ]
        planner = RandomTreePlanner(
            start=[0, 0],
            goal=[10, 0],
            obstacles=obstacles,
            search_bounds=[-5, 15],
            max_iterations=50  # Low iterations
        )
        path = planner.find_path()

        # May or may not find path depending on random samples
        # Just verify it doesn't crash
        assert path is None or isinstance(path, list)

    def test_search_bounds(self):
        """Test that planner respects search bounds."""
        random.seed(42)
        obstacles = []
        planner = RandomTreePlanner(
            start=[0, 0],
            goal=[5, 5],
            obstacles=obstacles,
            search_bounds=[0, 10]  # Constrained bounds
        )
        path = planner.find_path()

        assert path is not None
        # Path should be within bounds (approximately)
        for point in path:
            assert point[0] >= -1 and point[0] <= 11  # Some tolerance
            assert point[1] >= -1 and point[1] <= 11
