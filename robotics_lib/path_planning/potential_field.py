"""
Potential Field path planning algorithm.

Uses artificial potential fields to guide navigation:
- Attractive potential pulls toward the goal
- Repulsive potential pushes away from obstacles
"""
import math
import numpy as np


class PotentialFieldPlanner:
    """
    Potential Field path planner for 2D environments.

    Combines attractive and repulsive potentials to create
    a gradient field for navigation.
    """

    def __init__(self, obstacles_x, obstacles_y, goal_x, goal_y,
                 grid_resolution=0.5, robot_radius=0.0,
                 attract_gain=5.0, repulse_gain=100.0,
                 influence_distance=5.0):
        """
        Initialize the potential field planner.

        Parameters
        ----------
        obstacles_x : list
            X coordinates of obstacle positions
        obstacles_y : list
            Y coordinates of obstacle positions
        goal_x, goal_y : float
            Goal position coordinates
        grid_resolution : float
            Resolution for potential field computation
        robot_radius : float
            Robot radius for collision checking
        attract_gain : float
            Attractive potential gain (strength toward goal)
        repulse_gain : float
            Repulsive potential gain (strength away from obstacles)
        influence_distance : float
            Distance at which obstacles start affecting the robot
        """
        self.obstacles_x = obstacles_x
        self.obstacles_y = obstacles_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.resolution = grid_resolution
        self.robot_radius = robot_radius
        self.ka = attract_gain
        self.kr = repulse_gain
        self.influence_dist = influence_distance

    def find_path(self, start_x, start_y, max_steps=1000, step_size=0.1):
        """
        Find a path from start to goal using potential field navigation.

        Parameters
        ----------
        start_x, start_y : float
            Start position coordinates
        max_steps : int
            Maximum number of steps
        step_size : float
            Step size for gradient descent

        Returns
        -------
        tuple
            (path_x, path_y) lists of coordinates
        """
        path_x = [start_x]
        path_y = [start_y]

        x, y = start_x, start_y
        prev_x, prev_y = None, None
        oscillation_count = 0

        for _ in range(max_steps):
            if self._distance_to_goal(x, y) < self.resolution:
                break

            fx, fy = self._compute_force(x, y)

            force_mag = math.hypot(fx, fy)
            if force_mag < 1e-6:
                break

            fx_norm = fx / force_mag
            fy_norm = fy / force_mag

            new_x = x + step_size * fx_norm
            new_y = y + step_size * fy_norm

            if prev_x is not None:
                if (abs(new_x - prev_x) < step_size * 0.1 and
                    abs(new_y - prev_y) < step_size * 0.1):
                    oscillation_count += 1
                    if oscillation_count > 10:
                        break
                else:
                    oscillation_count = 0

            prev_x, prev_y = x, y
            x, y = new_x, new_y
            path_x.append(x)
            path_y.append(y)

        return path_x, path_y

    def _compute_force(self, x, y):
        """Compute total force at position."""
        fa_x, fa_y = self._attractive_force(x, y)
        fr_x, fr_y = self._repulsive_force(x, y)
        return fa_x + fr_x, fa_y + fr_y

    def _attractive_force(self, x, y):
        """Compute attractive force toward goal."""
        dx = self.goal_x - x
        dy = self.goal_y - y
        dist = math.hypot(dx, dy)

        if dist < 1e-6:
            return 0.0, 0.0

        return self.ka * dx, self.ka * dy

    def _repulsive_force(self, x, y):
        """Compute total repulsive force from all obstacles."""
        fx, fy = 0.0, 0.0

        for ox, oy in zip(self.obstacles_x, self.obstacles_y):
            dx = x - ox
            dy = y - oy
            dist = math.hypot(dx, dy) - self.robot_radius

            if dist <= 0:
                dist = 0.01

            if dist <= self.influence_dist:
                repulsion = self.kr * (1.0/dist - 1.0/self.influence_dist) / (dist * dist)
                fx += repulsion * dx / math.hypot(dx, dy)
                fy += repulsion * dy / math.hypot(dx, dy)

        return fx, fy

    def _distance_to_goal(self, x, y):
        """Calculate distance to goal."""
        return math.hypot(x - self.goal_x, y - self.goal_y)

    def compute_potential_field(self, x_range, y_range):
        """
        Compute the potential field over a grid for visualization.

        Parameters
        ----------
        x_range : tuple
            (min_x, max_x) for grid
        y_range : tuple
            (min_y, max_y) for grid

        Returns
        -------
        tuple
            (X_grid, Y_grid, potential_grid) numpy arrays
        """
        x = np.arange(x_range[0], x_range[1], self.resolution)
        y = np.arange(y_range[0], y_range[1], self.resolution)
        X, Y = np.meshgrid(x, y)
        potential = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                px, py = X[i, j], Y[i, j]
                pot_a = self._attractive_potential(px, py)
                pot_r = self._repulsive_potential(px, py)
                potential[i, j] = pot_a + pot_r

        return X, Y, potential

    def _attractive_potential(self, x, y):
        """Compute attractive potential at position."""
        dist = self._distance_to_goal(x, y)
        return 0.5 * self.ka * dist * dist

    def _repulsive_potential(self, x, y):
        """Compute total repulsive potential from all obstacles."""
        potential = 0.0
        for ox, oy in zip(self.obstacles_x, self.obstacles_y):
            dist = math.hypot(x - ox, y - oy) - self.robot_radius
            if dist <= 0:
                dist = 0.01
            if dist <= self.influence_dist:
                potential += 0.5 * self.kr * (1.0/dist - 1.0/self.influence_dist)**2
        return potential
