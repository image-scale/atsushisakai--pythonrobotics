"""
Angle manipulation utilities for robotics applications.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def normalize_angle(angle, positive_only=False, degrees=False):
    """
    Normalize angle(s) to a standard range.

    By default, normalizes to [-pi, pi) range. With positive_only=True,
    normalizes to [0, 2*pi) range.

    Parameters
    ----------
    angle : float or array_like
        Angle or array of angles to normalize
    positive_only : bool, optional
        If True, normalize to [0, 2*pi) instead of [-pi, pi)
    degrees : bool, optional
        If True, input/output are in degrees

    Returns
    -------
    float or ndarray
        Normalized angle(s)
    """
    is_scalar = isinstance(angle, (int, float))
    arr = np.asarray(angle).flatten()

    if degrees:
        arr = np.deg2rad(arr)

    if positive_only:
        result = arr % (2 * np.pi)
    else:
        result = (arr + np.pi) % (2 * np.pi) - np.pi

    if degrees:
        result = np.rad2deg(result)

    if is_scalar:
        return float(result[0])
    return result


def rotation_matrix_2d(angle):
    """
    Create a 2D rotation matrix from an angle.

    Parameters
    ----------
    angle : float
        Rotation angle in radians

    Returns
    -------
    ndarray
        2x2 rotation matrix
    """
    rot = Rotation.from_euler('z', angle)
    return rot.as_matrix()[:2, :2]
