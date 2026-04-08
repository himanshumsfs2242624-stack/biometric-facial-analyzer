# src/core/biometric/ocular.py

import mediapipe as mp
import numpy as np
from typing import Dict


class OcularTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # REQUIRED for iris
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Eye landmark indices (MediaPipe standard)
        # Based on common mapping :contentReference[oaicite:0]{index=0}
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Iris indices (MediaPipe adds 468–477)
        # :contentReference[oaicite:1]{index=1}
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]

    # -----------------------------
    # Utility functions
    # -----------------------------
    def _landmarks_to_array(self, landmarks, image_shape):
        h, w, _ = image_shape
        return np.array([[lm.x * w, lm.y * h] for lm in landmarks])

    def _euclidean(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    # -----------------------------
    # EAR Calculation
    # -----------------------------
    def _calculate_ear(self, eye_pts):
        # EAR formula :contentReference[oaicite:2]{index=2}
        A = self._euclidean(eye_pts[1], eye_pts[5])
        B = self._euclidean(eye_pts[2], eye_pts[4])
        C = self._euclidean(eye_pts[0], eye_pts[3])

        return (A + B) / (2.0 * C) if C > 0 else 0

    # -----------------------------
    # Iris / pupil calculations
    # -----------------------------
    def _iris_diameter(self, iris_pts):
        # Approximate diameter using bounding circle
        center = np.mean(iris_pts, axis=0)
        distances = [self._euclidean(center, p) for p in iris_pts]
        radius = np.mean(distances)
        return radius * 2, center

    def _pupil_movement(self, iris_center, eye_pts):
        # Eye center = midpoint between corners
        eye_center = (eye_pts[0] + eye_pts[3]) / 2
        delta = iris_center - eye_center
        return delta, eye_center

    def _gaze_direction(self, delta, eye_width, eye_height):
        """
        Normalize delta → approximate gaze vector
        """
        if eye_width == 0 or eye_height == 0:
            return {"yaw": 0, "pitch": 0}

        yaw = delta[0] / eye_width   # horizontal
        pitch = delta[1] / eye_height  # vertical

        return {
            "yaw": float(yaw),
            "pitch": float(pitch)
        }

    # -----------------------------
    # Main processing function
    # -----------------------------
    def process_frame(self, image) -> Dict:

        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            return {"face_detected": False}

        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        # Convert all landmarks
        pts = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks])

        # -----------------------------
        # Extract regions
        # -----------------------------
        left_eye_pts = pts[self.LEFT_EYE]
        right_eye_pts = pts[self.RIGHT_EYE]

        left_iris_pts = pts[self.LEFT_IRIS]
        right_iris_pts = pts[self.RIGHT_IRIS]

        # -----------------------------
        # EAR (blink detection)
        # -----------------------------
        left_ear = self._calculate_ear(left_eye_pts)
        right_ear = self._calculate_ear(right_eye_pts)

        # -----------------------------
        # Iris / pupil diameter
        # -----------------------------
        left_diameter, left_center = self._iris_diameter(left_iris_pts)
        right_diameter, right_center = self._iris_diameter(right_iris_pts)

        # -----------------------------
        # Pupil movement
        # -----------------------------
        left_delta, left_eye_center = self._pupil_movement(left_center, left_eye_pts)
        right_delta, right_eye_center = self._pupil_movement(right_center, right_eye_pts)

        # -----------------------------
        # Eye dimensions (for normalization)
        # -----------------------------
        left_width = self._euclidean(left_eye_pts[0], left_eye_pts[3])
        left_height = self._euclidean(left_eye_pts[1], left_eye_pts[5])

        right_width = self._euclidean(right_eye_pts[0], right_eye_pts[3])
        right_height = self._euclidean(right_eye_pts[1], right_eye_pts[5])

        # -----------------------------
        # Gaze direction
        # -----------------------------
        left_gaze = self._gaze_direction(left_delta, left_width, left_height)
        right_gaze = self._gaze_direction(right_delta, right_width, right_height)

        # -----------------------------
        # Final output
        # -----------------------------
        return {
            "face_detected": True,
            "eyes": {
                "left": {
                    "EAR": float(left_ear),
                    "pupil_diameter": float(left_diameter),
                    "pupil_center": left_center.tolist(),
                    "pupil_delta": left_delta.tolist(),
                    "gaze": left_gaze
                },
                "right": {
                    "EAR": float(right_ear),
                    "pupil_diameter": float(right_diameter),
                    "pupil_center": right_center.tolist(),
                    "pupil_delta": right_delta.tolist(),
                    "gaze": right_gaze
                }
            }
        }