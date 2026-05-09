# Acceptance Criteria

## Task 1: Angle utilities
- [x] `normalize_angle(-4.0)` returns approximately 2.283 (angle wrapped to [-pi, pi))
- [x] `normalize_angle(np.pi)` returns approximately -pi or pi (boundary case)
- [x] `normalize_angle([370], degrees=True)` returns approximately [10] degrees
- [x] `normalize_angle(-60, positive_only=True, degrees=True)` returns 300 degrees
- [x] `rotation_matrix_2d(np.pi/2)` returns a 2x2 matrix that rotates (1,0) to (0,1)
- [x] `rotation_matrix_2d(0)` returns the identity matrix
- [x] Both functions work with numpy arrays for vectorized operations

## Task 2: Dijkstra grid-based path planner
- [x] Planner finds shortest path from start to goal on a grid with obstacles
- [x] Returns the path as x and y coordinate lists
- [x] Correctly avoids obstacles (robot radius considered)
- [x] Supports 8-directional movement (including diagonals)
- [x] Handles different grid resolutions
- [x] Returns valid path for simple test case with obstacles
- [x] Path starts from start position and ends at goal position

## Task 3: A* grid-based path planner
- [x] A* planner finds path from start to goal using heuristic
- [x] Uses Euclidean distance heuristic for optimal pathfinding
- [x] Returns path as x and y coordinate lists
- [x] A* should generally explore fewer nodes than Dijkstra for the same problem
- [x] Correctly avoids obstacles considering robot radius
- [x] Handles empty open set gracefully (no path found)
- [x] Path endpoints match start and goal positions

## Task 4: RRT path planner
- [x] RRT finds path from start to goal by randomly sampling and connecting
- [x] Tree grows by extending from nearest node toward random samples
- [x] Checks collision with circular obstacles (x, y, radius)
- [x] Supports configurable expansion distance and path resolution
- [x] Goal sampling bias parameter speeds up convergence
- [x] Returns path as list of [x, y] waypoints, or None if no path found
- [x] Operates within configurable search bounds

## Task 5: Potential Field path planning
- [x] Attractive potential pulls robot toward goal
- [x] Repulsive potential pushes robot away from obstacles
- [x] Combined potential gradient provides motion direction
- [x] Returns path as x and y coordinate lists
- [x] Can get stuck in local minima (expected behavior)
- [x] Supports configurable attraction and repulsion strengths
- [x] Path generation stops at goal or when oscillating

## Task 6: PRM path planner
- [x] Samples random points in configuration space
- [x] Connects nearby samples that have collision-free paths
- [x] Builds a roadmap graph that can be queried
- [x] Finds path from start to goal using the roadmap
- [x] Uses Dijkstra/shortest path to search the graph
- [x] Checks collision with circular obstacles
- [x] Returns path as x and y coordinate lists or None if no path

## Task 7: Extended Kalman Filter
- [x] EKF predicts state using motion model
- [x] EKF updates state using observations
- [x] Computes Kalman gain for optimal fusion
- [x] Jacobian matrices used for linearization
- [x] Returns estimated state and covariance
- [x] Works with simulated robot motion and GPS observations
- [x] State includes position (x, y), heading (yaw), and velocity

## Task 8: ICP matching
- [x] Finds rotation and translation between two point sets
- [x] Iteratively refines alignment using nearest neighbor matching
- [x] Uses SVD for computing optimal transformation
- [x] Converges to solution within tolerance
- [x] Works with 2D point clouds
- [x] Returns rotation matrix and translation vector
- [x] Handles known correspondence case

## Task 9: Pure Pursuit path tracking
- [x] Controller computes steering angle to follow a path
- [x] Uses look-ahead distance to find target point
- [x] Look-ahead distance varies with velocity
- [x] Vehicle state includes position, heading, and velocity
- [x] State update follows bicycle kinematic model
- [x] Controller tracks waypoints from a given path
- [x] Includes PID speed control

## Task 10: K-means clustering
- [x] Clusters points into K groups
- [x] Iteratively updates cluster centroids
- [x] Assigns each point to nearest centroid
- [x] Converges when centroids stabilize
- [x] Returns cluster assignments and centroids
- [x] Works with 2D point data
- [x] Handles configurable number of clusters
