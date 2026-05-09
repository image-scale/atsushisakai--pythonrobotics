"""Tests for A* grid-based path planner."""
import math
import pytest

from robotics_lib.path_planning.astar import AStarPlanner


class TestAStarPlanner:
    """Tests for the AStarPlanner class."""

    def _create_boundary_obstacles(self, x_min, x_max, y_min, y_max):
        """Create obstacle lists that form a boundary."""
        ox, oy = [], []
        for i in range(x_min, x_max + 1):
            ox.append(float(i))
            oy.append(float(y_min))
        for i in range(x_min, x_max + 1):
            ox.append(float(i))
            oy.append(float(y_max))
        for i in range(y_min, y_max + 1):
            ox.append(float(x_min))
            oy.append(float(i))
        for i in range(y_min, y_max + 1):
            ox.append(float(x_max))
            oy.append(float(i))
        return ox, oy

    def test_simple_path_finding(self):
        """Test that A* finds a path in simple environment."""
        ox, oy = self._create_boundary_obstacles(0, 20, 0, 20)

        planner = AStarPlanner(ox, oy, grid_resolution=1.0, robot_radius=0.5)
        path_x, path_y = planner.find_path(5.0, 5.0, 15.0, 15.0)

        assert len(path_x) > 0
        assert len(path_y) > 0
        assert len(path_x) == len(path_y)

    def test_path_ends_at_goal(self):
        """Test that path starts at goal (path is reversed)."""
        ox, oy = self._create_boundary_obstacles(0, 20, 0, 20)

        planner = AStarPlanner(ox, oy, grid_resolution=1.0, robot_radius=0.5)
        path_x, path_y = planner.find_path(5.0, 5.0, 15.0, 15.0)

        assert math.isclose(path_x[0], 15.0, abs_tol=1.0)
        assert math.isclose(path_y[0], 15.0, abs_tol=1.0)

    def test_path_starts_at_start(self):
        """Test that path ends at start (path is reversed)."""
        ox, oy = self._create_boundary_obstacles(0, 20, 0, 20)

        planner = AStarPlanner(ox, oy, grid_resolution=1.0, robot_radius=0.5)
        path_x, path_y = planner.find_path(5.0, 5.0, 15.0, 15.0)

        assert math.isclose(path_x[-1], 5.0, abs_tol=1.0)
        assert math.isclose(path_y[-1], 5.0, abs_tol=1.0)

    def test_avoids_obstacles(self):
        """Test that path avoids obstacles."""
        ox, oy = self._create_boundary_obstacles(0, 20, 0, 20)
        # Add a wall obstacle in the middle
        for i in range(5, 15):
            ox.append(10.0)
            oy.append(float(i))

        planner = AStarPlanner(ox, oy, grid_resolution=1.0, robot_radius=0.5)
        path_x, path_y = planner.find_path(5.0, 10.0, 15.0, 10.0)

        # Path should exist and go around the obstacle
        assert len(path_x) > 0
        # Verify no path point is on the obstacle
        for px, py in zip(path_x, path_y):
            if 5 <= py <= 14:
                assert abs(px - 10.0) > 0.5

    def test_no_path_found(self):
        """Test handling when no path exists."""
        ox, oy = self._create_boundary_obstacles(0, 20, 0, 20)
        # Add a complete wall that blocks the path
        for i in range(0, 21):
            ox.append(10.0)
            oy.append(float(i))

        planner = AStarPlanner(ox, oy, grid_resolution=1.0, robot_radius=0.5)
        path_x, path_y = planner.find_path(5.0, 10.0, 15.0, 10.0)

        # Should return empty lists when no path found
        assert len(path_x) == 0
        assert len(path_y) == 0

    def test_heuristic_efficiency(self):
        """Test that A* uses heuristic effectively (optimal path for diagonal)."""
        ox, oy = self._create_boundary_obstacles(0, 30, 0, 30)

        planner = AStarPlanner(ox, oy, grid_resolution=1.0, robot_radius=0.5)
        path_x, path_y = planner.find_path(5.0, 5.0, 25.0, 25.0)

        # Calculate path length
        path_length = 0
        for i in range(len(path_x) - 1):
            dx = path_x[i+1] - path_x[i]
            dy = path_y[i+1] - path_y[i]
            path_length += math.hypot(dx, dy)

        # Straight-line distance
        direct_dist = math.hypot(25.0 - 5.0, 25.0 - 5.0)
        # Path should be close to optimal
        assert path_length < direct_dist * 1.2

    def test_different_resolution(self):
        """Test planner works with different grid resolutions."""
        ox, oy = self._create_boundary_obstacles(0, 40, 0, 40)

        planner = AStarPlanner(ox, oy, grid_resolution=2.0, robot_radius=1.0)
        path_x, path_y = planner.find_path(5.0, 5.0, 35.0, 35.0)

        assert len(path_x) > 0
        assert len(path_y) > 0
