# Acceptance Criteria

## Task 1: Angle utilities

### Acceptance Criteria
- [ ] `normalize_angle(-4.0)` returns approximately 2.283 (angle wrapped to [-pi, pi))
- [ ] `normalize_angle(np.pi)` returns approximately -pi or pi (boundary case)
- [ ] `normalize_angle([370], degrees=True)` returns approximately [10] degrees
- [ ] `normalize_angle(-60, positive_only=True, degrees=True)` returns 300 degrees
- [ ] `rotation_matrix_2d(np.pi/2)` returns a 2x2 matrix that rotates (1,0) to (0,1)
- [ ] `rotation_matrix_2d(0)` returns the identity matrix
- [ ] Both functions work with numpy arrays for vectorized operations
