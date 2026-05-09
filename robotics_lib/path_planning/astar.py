"""
A* grid-based path planning algorithm.

Extends Dijkstra's algorithm with a heuristic for faster pathfinding.
"""
import math


class AStarPlanner:
    """
    A* path planner for 2D grids with obstacles.

    Uses Euclidean distance heuristic for optimal pathfinding.
    Supports 8-directional movement and obstacle avoidance.
    """

    def __init__(self, obstacles_x, obstacles_y, grid_resolution, robot_radius):
        """
        Initialize the A* planner.

        Parameters
        ----------
        obstacles_x : list
            X coordinates of obstacle positions
        obstacles_y : list
            Y coordinates of obstacle positions
        grid_resolution : float
            Size of each grid cell
        robot_radius : float
            Robot radius for collision checking
        """
        self.resolution = grid_resolution
        self.robot_radius = robot_radius

        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.width_x = None
        self.width_y = None
        self.obstacle_grid = None

        self._build_obstacle_grid(obstacles_x, obstacles_y)
        self.movements = self._get_movement_directions()

    def find_path(self, start_x, start_y, goal_x, goal_y):
        """
        Find a path from start to goal using A* algorithm.

        Parameters
        ----------
        start_x, start_y : float
            Start position coordinates
        goal_x, goal_y : float
            Goal position coordinates

        Returns
        -------
        tuple
            (path_x, path_y) lists of coordinates, or ([], []) if no path found
        """
        start = _AStarNode(
            self._position_to_index(start_x, self.min_x),
            self._position_to_index(start_y, self.min_y),
            0.0, -1
        )
        goal = _AStarNode(
            self._position_to_index(goal_x, self.min_x),
            self._position_to_index(goal_y, self.min_y),
            0.0, -1
        )

        open_nodes = {self._node_index(start): start}
        closed_nodes = {}

        while open_nodes:
            # Select node with lowest f = g + h
            current_key = min(
                open_nodes,
                key=lambda k: open_nodes[k].cost + self._heuristic(open_nodes[k], goal)
            )
            current = open_nodes[current_key]

            if current.gx == goal.gx and current.gy == goal.gy:
                goal.parent_idx = current.parent_idx
                goal.cost = current.cost
                break

            del open_nodes[current_key]
            closed_nodes[current_key] = current

            for dx, dy, move_cost in self.movements:
                neighbor = _AStarNode(
                    current.gx + dx,
                    current.gy + dy,
                    current.cost + move_cost,
                    current_key
                )
                neighbor_key = self._node_index(neighbor)

                if neighbor_key in closed_nodes:
                    continue

                if not self._is_valid_node(neighbor):
                    continue

                if neighbor_key not in open_nodes:
                    open_nodes[neighbor_key] = neighbor
                elif open_nodes[neighbor_key].cost > neighbor.cost:
                    open_nodes[neighbor_key] = neighbor
        else:
            return [], []

        return self._trace_path(goal, closed_nodes)

    def _heuristic(self, node, goal):
        """Calculate Euclidean distance heuristic."""
        dx = node.gx - goal.gx
        dy = node.gy - goal.gy
        return math.hypot(dx, dy)

    def _build_obstacle_grid(self, obs_x, obs_y):
        """Build the obstacle grid from obstacle positions."""
        self.min_x = round(min(obs_x))
        self.min_y = round(min(obs_y))
        self.max_x = round(max(obs_x))
        self.max_y = round(max(obs_y))

        self.width_x = round((self.max_x - self.min_x) / self.resolution)
        self.width_y = round((self.max_y - self.min_y) / self.resolution)

        self.obstacle_grid = [
            [False for _ in range(self.width_y)]
            for _ in range(self.width_x)
        ]

        for ix in range(self.width_x):
            x = self._index_to_position(ix, self.min_x)
            for iy in range(self.width_y):
                y = self._index_to_position(iy, self.min_y)
                for ox, oy in zip(obs_x, obs_y):
                    dist = math.hypot(ox - x, oy - y)
                    if dist <= self.robot_radius:
                        self.obstacle_grid[ix][iy] = True
                        break

    def _get_movement_directions(self):
        """Get the 8-directional movement model with costs."""
        diag = math.sqrt(2)
        return [
            (1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1),
            (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)
        ]

    def _position_to_index(self, pos, min_pos):
        """Convert world position to grid index."""
        return round((pos - min_pos) / self.resolution)

    def _index_to_position(self, idx, min_pos):
        """Convert grid index to world position."""
        return idx * self.resolution + min_pos

    def _node_index(self, node):
        """Calculate unique index for a node."""
        return (node.gy - self.min_y) * self.width_x + (node.gx - self.min_x)

    def _is_valid_node(self, node):
        """Check if node is within bounds and not in obstacle."""
        px = self._index_to_position(node.gx, self.min_x)
        py = self._index_to_position(node.gy, self.min_y)

        if px < self.min_x or py < self.min_y:
            return False
        if px >= self.max_x or py >= self.max_y:
            return False
        if self.obstacle_grid[node.gx][node.gy]:
            return False
        return True

    def _trace_path(self, goal_node, closed_nodes):
        """Trace back from goal to start to get the path."""
        path_x = [self._index_to_position(goal_node.gx, self.min_x)]
        path_y = [self._index_to_position(goal_node.gy, self.min_y)]

        parent_idx = goal_node.parent_idx
        while parent_idx != -1:
            node = closed_nodes[parent_idx]
            path_x.append(self._index_to_position(node.gx, self.min_x))
            path_y.append(self._index_to_position(node.gy, self.min_y))
            parent_idx = node.parent_idx

        return path_x, path_y


class _AStarNode:
    """Internal node class for A* search."""

    def __init__(self, gx, gy, cost, parent_idx):
        self.gx = gx
        self.gy = gy
        self.cost = cost  # g-value
        self.parent_idx = parent_idx
