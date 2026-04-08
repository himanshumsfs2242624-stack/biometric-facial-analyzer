# src/core/utils/geometry_math.py

import numpy as np
from typing import List, Union


# -----------------------------
# Euclidean Distance
# -----------------------------
def calculate_euclidean_distance(
    pt1: Union[List[float], np.ndarray],
    pt2: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate Euclidean distance between two points (2D or 3D).
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)

    return float(np.linalg.norm(pt1 - pt2))


# -----------------------------
# Eye Aspect Ratio (EAR)
# -----------------------------
def calculate_ear(eye_landmarks: List[List[float]]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR).

    Expected 6 eye landmarks:
    [p1, p2, p3, p4, p5, p6]
    """

    if len(eye_landmarks) != 6:
        raise ValueError("EAR requires exactly 6 eye landmarks")

    p1, p2, p3, p4, p5, p6 = map(np.array, eye_landmarks)

    A = calculate_euclidean_distance(p2, p6)
    B = calculate_euclidean_distance(p3, p5)
    C = calculate_euclidean_distance(p1, p4)

    return (A + B) / (2.0 * C) if C > 0 else 0.0


# -----------------------------
# Mouth Aspect Ratio (MAR)
# -----------------------------
def calculate_mar(lip_landmarks: List[List[float]]) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR).

    Expected 4 key landmarks:
    [left_corner, right_corner, upper_lip, lower_lip]
    """

    if len(lip_landmarks) != 4:
        raise ValueError("MAR requires exactly 4 lip landmarks")

    left, right, upper, lower = map(np.array, lip_landmarks)

    vertical = calculate_euclidean_distance(upper, lower)
    horizontal = calculate_euclidean_distance(left, right)

    return vertical / horizontal if horizontal > 0 else 0.0


# -----------------------------
# Vector Normalization
# -----------------------------
def normalize_vector(v: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Normalize vector to unit length.
    Useful for gaze direction vectors.
    """

    v = np.array(v, dtype=np.float32)

    norm = np.linalg.norm(v)

    if norm == 0:
        return v

    return v / norm