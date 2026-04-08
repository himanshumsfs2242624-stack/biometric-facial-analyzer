# tests/test_forensics.py

import pytest
import numpy as np

from src.core.forensic.ela import perform_ela
from src.core.forensic.prnu import extract_sensor_noise


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def dummy_image():
    """
    Create a deterministic dummy RGB image
    """
    np.random.seed(42)
    return (np.random.rand(256, 256, 3) * 255).astype(np.uint8)


# -----------------------------
# ELA Test
# -----------------------------
def test_perform_ela(dummy_image):
    result = perform_ela(dummy_image)

    # Ensure no crash and correct structure
    assert isinstance(result, dict)
    assert "ela_image" in result
    assert "max_discrepancy" in result

    ela_img = result["ela_image"]

    # Check type
    assert isinstance(ela_img, np.ndarray)

    # Check spatial dimensions match input
    assert ela_img.shape[:2] == dummy_image.shape[:2]

    # Check discrepancy is numeric
    assert isinstance(result["max_discrepancy"], float)


# -----------------------------
# PRNU Test
# -----------------------------
def test_extract_sensor_noise(dummy_image):
    result = extract_sensor_noise(dummy_image)

    # Ensure no crash and correct structure
    assert isinstance(result, dict)
    assert "noise_residue" in result

    noise = result["noise_residue"]

    # Check type
    assert isinstance(noise, np.ndarray)

    # Check spatial dimensions match grayscale input
    assert noise.shape[:2] == dummy_image.shape[:2]

    # Ensure values are finite
    assert np.isfinite(noise).all()