"""
Rapidly-exploring Random Tree (RRT) path planning algorithm.

Builds a tree by randomly sampling the space and connecting to nearest nodes.
"""
import math
import random


class RandomTreePlanner:
    """
    RRT path planner for 2D environments with circular obstacles.

    Grows a tree from start toward goal by random sampling and
    extending toward sampled points.
    """

    def __init__(self, start, goal, obstacles, search_bounds,
                 step_size=3.0, path_step=0.5, goal_bias=5, max_iterations=500,
                 robot_radius=0.0):
        """
        Initialize the RRT planner.

        Parameters
        ----------
        start : tuple/list
            Start position [x, y]
        goal : tuple/list
            Goal position [x, y]
        obstacles : list
            List of obstacles, each as (x, y, radius)
        search_bounds : tuple/list
            Random sampling bounds [min, max]
        step_size : float
            Maximum extension distance per step
        path_step : float
            Resolution for path interpolation
        goal_bias : int
            Percentage chance (0-100) to sample goal directly
        max_iterations : int
            Maximum tree expansion iterations
        robot_radius : float
            Robot radius for collision checking
        """
        self.start = _TreeNode(start[0], start[1])
        self.goal = _TreeNode(goal[0], goal[1])
        self.obstacles = obstacles
        self.min_bound = search_bounds[0]
        self.max_bound = search_bounds[1]
        self.step_size = step_size
        self.path_step = path_step
        self.goal_bias = goal_bias
        self.max_iter = max_iterations
        self.robot_radius = robot_radius
        self.nodes = []

    def find_path(self):
        """
        Search for a path from start to goal.

        Returns
        -------
        list or None
            List of [x, y] waypoints, or None if no path found
        """
        self.nodes = [self.start]

        for _ in range(self.max_iter):
            sample = self._random_sample()
            nearest_idx = self._find_nearest(sample)
            nearest = self.nodes[nearest_idx]

            new_node = self._extend(nearest, sample)

            if self._is_collision_free(new_node):
                self.nodes.append(new_node)

                if self._distance_to_goal(new_node) <= self.step_size:
                    final = self._extend(new_node, self.goal)
                    if self._is_collision_free(final):
                        return self._extract_path(len(self.nodes) - 1)

        return None

    def _random_sample(self):
        """Generate a random sample point, with goal bias."""
        if random.randint(0, 100) > self.goal_bias:
            x = random.uniform(self.min_bound, self.max_bound)
            y = random.uniform(self.min_bound, self.max_bound)
            return _TreeNode(x, y)
        return _TreeNode(self.goal.x, self.goal.y)

    def _find_nearest(self, sample):
        """Find the nearest node in the tree to the sample."""
        dists = [(n.x - sample.x)**2 + (n.y - sample.y)**2 for n in self.nodes]
        return dists.index(min(dists))

    def _extend(self, from_node, to_node):
        """Extend from a node toward another point."""
        new = _TreeNode(from_node.x, from_node.y)
        dist, angle = self._calc_dist_angle(from_node, to_node)

        extend_len = min(self.step_size, dist)
        steps = int(extend_len / self.path_step)

        new.path_x = [new.x]
        new.path_y = [new.y]

        for _ in range(steps):
            new.x += self.path_step * math.cos(angle)
            new.y += self.path_step * math.sin(angle)
            new.path_x.append(new.x)
            new.path_y.append(new.y)

        new.parent = from_node
        return new

    def _is_collision_free(self, node):
        """Check if node path is collision-free with all obstacles."""
        if node is None:
            return False

        for ox, oy, radius in self.obstacles:
            for px, py in zip(node.path_x, node.path_y):
                dist = math.hypot(ox - px, oy - py)
                if dist <= radius + self.robot_radius:
                    return False
        return True

    def _distance_to_goal(self, node):
        """Calculate distance from node to goal."""
        return math.hypot(node.x - self.goal.x, node.y - self.goal.y)

    def _calc_dist_angle(self, from_node, to_node):
        """Calculate distance and angle between nodes."""
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def _extract_path(self, goal_idx):
        """Extract path from tree by backtracking from goal."""
        path = [[self.goal.x, self.goal.y]]
        node = self.nodes[goal_idx]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path


class _TreeNode:
    """Node in the RRT tree."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None
