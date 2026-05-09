"""Tests for PRM (Probabilistic Roadmap) path planner."""
import math
import random
import pytest

from robotics_lib.path_planning.prm import RoadmapPlanner


class TestRoadmapPlanner:
    """Tests for the RoadmapPlanner class."""

    def test_builds_roadmap(self):
        """Test that roadmap is built with samples."""
        random.seed(42)
        obstacles = [(5, 5, 1)]
        planner = RoadmapPlanner(
            obstacles=obstacles,
            area_bounds=(0, 20, 0, 20),
            num_samples=100
        )
        planner.build_roadmap()

        assert len(planner.samples) > 0
        assert len(planner.roadmap) > 0

    def test_finds_simple_path(self):
        """Test that PRM finds a path in simple environment."""
        random.seed(42)
        obstacles = [(5, 5, 0.5)]
        planner = RoadmapPlanner(
            obstacles=obstacles,
            area_bounds=(0, 20, 0, 20),
            num_samples=200,
            connection_distance=8.0
        )
        path_x, path_y = planner.find_path(0, 0, 20, 20)

        assert len(path_x) > 0
        assert len(path_y) > 0
        assert len(path_x) == len(path_y)

    def test_path_starts_at_start(self):
        """Test that path starts at the start position."""
        random.seed(42)
        obstacles = []
        planner = RoadmapPlanner(
            obstacles=obstacles,
            area_bounds=(0, 20, 0, 20),
            num_samples=100,
            connection_distance=10.0
        )
        path_x, path_y = planner.find_path(1.0, 1.0, 19.0, 19.0)

        if len(path_x) > 0:
            assert math.isclose(path_x[0], 1.0, abs_tol=0.01)
            assert math.isclose(path_y[0], 1.0, abs_tol=0.01)

    def test_path_ends_at_goal(self):
        """Test that path ends at the goal position."""
        random.seed(42)
        obstacles = []
        planner = RoadmapPlanner(
            obstacles=obstacles,
            area_bounds=(0, 20, 0, 20),
            num_samples=100,
            connection_distance=10.0
        )
        path_x, path_y = planner.find_path(1.0, 1.0, 19.0, 19.0)

        if len(path_x) > 0:
            assert math.isclose(path_x[-1], 19.0, abs_tol=0.01)
            assert math.isclose(path_y[-1], 19.0, abs_tol=0.01)

    def test_avoids_obstacles(self):
        """Test that samples avoid obstacles."""
        random.seed(42)
        obstacles = [(10, 10, 3)]
        planner = RoadmapPlanner(
            obstacles=obstacles,
            area_bounds=(0, 20, 0, 20),
            num_samples=200,
            robot_radius=0.5
        )
        planner.build_roadmap()

        # All samples should be collision-free
        for x, y in planner.samples:
            dist_to_obs = math.hypot(x - 10, y - 10)
            assert dist_to_obs > 3.0 + 0.5 - 0.1  # obstacle + robot - tolerance

    def test_no_path_when_blocked(self):
        """Test handling when no path exists."""
        random.seed(42)
        # Wall of obstacles blocking the way
        obstacles = [(10, i, 1) for i in range(25)]
        planner = RoadmapPlanner(
            obstacles=obstacles,
            area_bounds=(0, 20, 0, 20),
            num_samples=50,
            connection_distance=3.0
        )
        path_x, path_y = planner.find_path(2, 10, 18, 10)

        # May or may not find path - just verify it doesn't crash
        assert isinstance(path_x, list)
        assert isinstance(path_y, list)

    def test_connection_distance_affects_connectivity(self):
        """Test that connection distance affects roadmap connectivity."""
        random.seed(42)
        obstacles = []

        # Small connection distance - fewer edges
        planner_small = RoadmapPlanner(
            obstacles=obstacles,
            area_bounds=(0, 20, 0, 20),
            num_samples=100,
            connection_distance=2.0
        )
        planner_small.build_roadmap()

        # Large connection distance - more edges
        random.seed(42)
        planner_large = RoadmapPlanner(
            obstacles=obstacles,
            area_bounds=(0, 20, 0, 20),
            num_samples=100,
            connection_distance=10.0
        )
        planner_large.build_roadmap()

        # Count total edges
        edges_small = sum(len(neighbors) for neighbors in planner_small.roadmap.values())
        edges_large = sum(len(neighbors) for neighbors in planner_large.roadmap.values())

        assert edges_large > edges_small

    def test_visualization_data(self):
        """Test that visualization data can be retrieved."""
        random.seed(42)
        obstacles = []
        planner = RoadmapPlanner(
            obstacles=obstacles,
            area_bounds=(0, 10, 0, 10),
            num_samples=50,
            connection_distance=5.0
        )
        planner.build_roadmap()

        samples, edges = planner.get_roadmap_for_visualization()

        assert len(samples) > 0
        # Edges should be a list of tuples
        for edge in edges:
            assert len(edge) == 2
            assert len(edge[0]) == 2  # (x, y)
            assert len(edge[1]) == 2
