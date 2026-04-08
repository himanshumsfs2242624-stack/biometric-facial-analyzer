# tests/test_biometrics.py

import pytest
import numpy as np

# Import your modules
from src.core.biometric.ocular import OcularTracker
from src.core.biometric.mouth import MouthTracker
from src.core.biometric.mesh_468 import FaceMeshTracker


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def dummy_image():
    """
    Create a deterministic dummy RGB image (face-like placeholder).
    """
    np.random.seed(42)  # ensure reproducibility
    return (np.random.rand(480, 640, 3) * 255).astype(np.uint8)


@pytest.fixture
def dummy_landmarks():
    """
    Create fake 468 landmarks (for mesh delta testing).
    """
    return [
        {"x": float(i), "y": float(i), "z": 0.0}
        for i in range(468)
    ]


# -----------------------------
# OcularTracker Test
# -----------------------------
def test_ocular_tracker_process_frame(dummy_image, mocker):
    tracker = OcularTracker()

    # Mock mediapipe output
    mock_landmarks = mocker.Mock()
    mock_landmarks.landmark = [
        mocker.Mock(x=0.5, y=0.5, z=0.0) for _ in range(478)
    ]

    mock_results = mocker.Mock()
    mock_results.multi_face_landmarks = [mock_landmarks]

    mocker.patch.object(tracker.face_mesh, "process", return_value=mock_results)

    result = tracker.process_frame(dummy_image)

    assert isinstance(result, dict)
    assert result["face_detected"] is True

    # Required keys
    assert "eyes" in result
    assert "left" in result["eyes"]
    assert "right" in result["eyes"]

    assert "EAR" in result["eyes"]["left"]
    assert "pupil_diameter" in result["eyes"]["left"]


# -----------------------------
# MouthTracker Test
# -----------------------------
def test_mouth_tracker_process_frame(dummy_image, mocker):
    tracker = MouthTracker()

    mock_landmarks = mocker.Mock()
    mock_landmarks.landmark = [
        mocker.Mock(x=0.5, y=0.5, z=0.0) for _ in range(468)
    ]

    mock_results = mocker.Mock()
    mock_results.multi_face_landmarks = [mock_landmarks]

    mocker.patch.object(tracker.face_mesh, "process", return_value=mock_results)

    result = tracker.process_frame(dummy_image, frame_id=1)

    assert isinstance(result, dict)
    assert result["face_detected"] is True

    # Required keys
    assert "mouth" in result
    assert "MAR" in result["mouth"]
    assert "state" in result["mouth"]


# -----------------------------
# FaceMeshTracker Test
# -----------------------------
def test_face_mesh_tracker_process_frame(dummy_image, dummy_landmarks, mocker):
    tracker = FaceMeshTracker()

    mock_landmarks = mocker.Mock()
    mock_landmarks.landmark = [
        mocker.Mock(x=0.5, y=0.5, z=0.0) for _ in range(468)
    ]

    mock_results = mocker.Mock()
    mock_results.multi_face_landmarks = [mock_landmarks]

    mocker.patch.object(tracker.face_mesh, "process", return_value=mock_results)

    result = tracker.process_frame(
        dummy_image,
        previous_frame_landmarks=dummy_landmarks
    )

    assert isinstance(result, dict)
    assert result["face_detected"] is True

    # Required keys
    assert "landmarks" in result
    assert "movement" in result

    assert "chin_delta" in result["movement"]
    assert "cheeks_delta" in result["movement"]