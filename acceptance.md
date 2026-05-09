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
- [ ] Planner finds shortest path from start to goal on a grid with obstacles
- [ ] Returns the path as x and y coordinate lists
- [ ] Correctly avoids obstacles (robot radius considered)
- [ ] Supports 8-directional movement (including diagonals)
- [ ] Handles different grid resolutions
- [ ] Returns valid path for simple test case with obstacles
- [ ] Path starts from start position and ends at goal position
