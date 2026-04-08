# src/core/biometric/keypoint_matcher.py

import numpy as np
from typing import List, Dict


def _flatten_landmarks(landmarks: List[Dict]) -> np.ndarray:
    """
    Convert list of landmark dicts → flat vector
    Expected format:
    [
        {"x": ..., "y": ..., "z": ...},
        ...
    ]
    """
    return np.array(
        [coord for lm in landmarks for coord in (lm["x"], lm["y"], lm["z"])],
        dtype=np.float32
    )


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize vector:
    1. Center (remove mean)
    2. Scale to unit length
    """
    vec = vec - np.mean(vec)  # center
    norm = np.linalg.norm(vec)

    if norm == 0:
        return vec

    return vec / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def compare_faces(landmarks_a: List[Dict], landmarks_b: List[Dict]) -> Dict:
    """
    Compare two sets of 468 3D landmarks using cosine similarity.

    Returns:
        {
            "similarity": float (-1 to 1),
            "match_percentage": float (0 to 100)
        }
    """

    # Step 1: Flatten
    vec_a = _flatten_landmarks(landmarks_a)
    vec_b = _flatten_landmarks(landmarks_b)

    # Step 2: Normalize
    vec_a = _normalize_vector(vec_a)
    vec_b = _normalize_vector(vec_b)

    # Step 3: Cosine similarity
    similarity = _cosine_similarity(vec_a, vec_b)

    # Step 4: Convert to percentage (0–100)
    # Cosine range: [-1, 1] → map to [0, 100]
    match_percentage = (similarity + 1) / 2 * 100

    return {
        "similarity": similarity,
        "match_percentage": match_percentage
    }