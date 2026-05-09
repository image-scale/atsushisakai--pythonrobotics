# Acceptance Criteria

## Task 1: Angle utilities

### Acceptance Criteria
- [x] `normalize_angle(-4.0)` returns approximately 2.283 (angle wrapped to [-pi, pi))
- [x] `normalize_angle(np.pi)` returns approximately -pi or pi (boundary case)
- [x] `normalize_angle([370], degrees=True)` returns approximately [10] degrees
- [x] `normalize_angle(-60, positive_only=True, degrees=True)` returns 300 degrees
- [x] `rotation_matrix_2d(np.pi/2)` returns a 2x2 matrix that rotates (1,0) to (0,1)
- [x] `rotation_matrix_2d(0)` returns the identity matrix
- [x] Both functions work with numpy arrays for vectorized operations

## Task 2: Dijkstra grid-based path planner

### Acceptance Criteria
- [x] Planner finds shortest path from start to goal on a grid with obstacles
- [x] Returns the path as x and y coordinate lists
- [x] Correctly avoids obstacles (robot radius considered)
- [x] Supports 8-directional movement (including diagonals)
- [x] Handles different grid resolutions
- [x] Returns valid path for simple test case with obstacles
- [x] Path starts from start position and ends at goal position

## Task 3: A* grid-based path planner

### Acceptance Criteria
- [x] A* planner finds path from start to goal using heuristic
- [x] Uses Euclidean distance heuristic for optimal pathfinding
- [x] Returns path as x and y coordinate lists
- [x] A* should generally explore fewer nodes than Dijkstra for the same problem
- [x] Correctly avoids obstacles considering robot radius
- [x] Handles empty open set gracefully (no path found)
- [x] Path endpoints match start and goal positions

## Task 4: RRT path planner

### Acceptance Criteria
- [ ] RRT finds path from start to goal by randomly sampling and connecting
- [ ] Tree grows by extending from nearest node toward random samples
- [ ] Checks collision with circular obstacles (x, y, radius)
- [ ] Supports configurable expansion distance and path resolution
- [ ] Goal sampling bias parameter speeds up convergence
- [ ] Returns path as list of [x, y] waypoints, or None if no path found
- [ ] Operates within configurable search bounds
