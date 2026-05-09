"""
Probabilistic Roadmap (PRM) path planning algorithm.

Builds a roadmap by sampling random points and connecting nearby samples.
"""
import math
import random


class RoadmapPlanner:
    """
    Probabilistic Roadmap path planner for 2D environments.

    Samples random configurations, connects nearby collision-free samples
    to build a roadmap, then searches the roadmap for paths.
    """

    def __init__(self, obstacles, area_bounds, robot_radius=0.0,
                 num_samples=500, connection_distance=10.0):
        """
        Initialize the PRM planner.

        Parameters
        ----------
        obstacles : list
            List of obstacles as (x, y, radius)
        area_bounds : tuple
            (min_x, max_x, min_y, max_y) search bounds
        robot_radius : float
            Robot radius for collision checking
        num_samples : int
            Number of random samples to generate
        connection_distance : float
            Maximum distance for connecting samples
        """
        self.obstacles = obstacles
        self.min_x = area_bounds[0]
        self.max_x = area_bounds[1]
        self.min_y = area_bounds[2]
        self.max_y = area_bounds[3]
        self.robot_radius = robot_radius
        self.num_samples = num_samples
        self.connect_dist = connection_distance

        self.samples = []
        self.roadmap = {}  # adjacency list

    def build_roadmap(self):
        """Build the probabilistic roadmap by sampling and connecting."""
        self.samples = []
        self.roadmap = {}

        for _ in range(self.num_samples):
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            if self._is_collision_free_point(x, y):
                self.samples.append((x, y))

        for i, sample in enumerate(self.samples):
            self.roadmap[i] = []
            for j, other in enumerate(self.samples):
                if i == j:
                    continue
                dist = self._distance(sample, other)
                if dist <= self.connect_dist:
                    if self._is_collision_free_edge(sample, other):
                        self.roadmap[i].append((j, dist))

    def find_path(self, start_x, start_y, goal_x, goal_y):
        """
        Find a path from start to goal using the roadmap.

        Parameters
        ----------
        start_x, start_y : float
            Start position
        goal_x, goal_y : float
            Goal position

        Returns
        -------
        tuple
            (path_x, path_y) or ([], []) if no path found
        """
        if not self.samples:
            self.build_roadmap()

        start = (start_x, start_y)
        goal = (goal_x, goal_y)

        start_idx = len(self.samples)
        goal_idx = len(self.samples) + 1
        self.samples.append(start)
        self.samples.append(goal)

        self.roadmap[start_idx] = []
        self.roadmap[goal_idx] = []

        for i, sample in enumerate(self.samples[:-2]):
            dist_s = self._distance(start, sample)
            dist_g = self._distance(goal, sample)

            if dist_s <= self.connect_dist:
                if self._is_collision_free_edge(start, sample):
                    self.roadmap[start_idx].append((i, dist_s))
                    self.roadmap[i].append((start_idx, dist_s))

            if dist_g <= self.connect_dist:
                if self._is_collision_free_edge(goal, sample):
                    self.roadmap[goal_idx].append((i, dist_g))
                    self.roadmap[i].append((goal_idx, dist_g))

        path_indices = self._dijkstra_search(start_idx, goal_idx)

        self.samples = self.samples[:-2]
        del self.roadmap[start_idx]
        del self.roadmap[goal_idx]
        for i in self.roadmap:
            self.roadmap[i] = [(j, d) for j, d in self.roadmap[i]
                               if j not in (start_idx, goal_idx)]

        if path_indices is None:
            return [], []

        path_x = [start_x]
        path_y = [start_y]
        for idx in path_indices[1:-1]:
            path_x.append(self.samples[idx][0])
            path_y.append(self.samples[idx][1])
        path_x.append(goal_x)
        path_y.append(goal_y)

        return path_x, path_y

    def _dijkstra_search(self, start_idx, goal_idx):
        """Search for shortest path using Dijkstra's algorithm."""
        costs = {start_idx: 0}
        parents = {start_idx: None}
        open_set = {start_idx}
        closed_set = set()

        while open_set:
            current = min(open_set, key=lambda x: costs[x])
            if current == goal_idx:
                path = []
                node = goal_idx
                while node is not None:
                    path.append(node)
                    node = parents[node]
                return list(reversed(path))

            open_set.remove(current)
            closed_set.add(current)

            for neighbor, dist in self.roadmap.get(current, []):
                if neighbor in closed_set:
                    continue
                new_cost = costs[current] + dist
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    parents[neighbor] = current
                    open_set.add(neighbor)

        return None

    def _is_collision_free_point(self, x, y):
        """Check if a point is collision-free."""
        for ox, oy, radius in self.obstacles:
            if self._distance((x, y), (ox, oy)) <= radius + self.robot_radius:
                return False
        return True

    def _is_collision_free_edge(self, p1, p2):
        """Check if an edge between two points is collision-free."""
        dist = self._distance(p1, p2)
        if dist < 1e-6:
            return True

        steps = int(dist / (self.robot_radius + 0.1)) + 1
        steps = max(steps, 10)

        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if not self._is_collision_free_point(x, y):
                return False
        return True

    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def get_roadmap_for_visualization(self):
        """Return samples and edges for visualization."""
        edges = []
        for i, neighbors in self.roadmap.items():
            for j, _ in neighbors:
                if i < j:
                    edges.append((self.samples[i], self.samples[j]))
        return self.samples, edges
