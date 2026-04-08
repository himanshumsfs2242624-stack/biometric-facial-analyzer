# src/core/biometric/mouth.py

import mediapipe as mp
import numpy as np
from typing import Dict, Optional


class MouthTracker:
    def __init__(self, mar_threshold: float = 0.6):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Mouth landmark indices
        # Based on common MediaPipe mappings
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.UPPER_LIP = 13
        self.LOWER_LIP = 14

        # Optional richer lip regions (outer + inner lips)
        self.LIPS = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146
        ]  # :contentReference[oaicite:0]{index=0}

        # State machine
        self.mar_threshold = mar_threshold
        self.state = "closed"
        self.open_start_frame: Optional[int] = None

    # -----------------------------
    # Utility functions
    # -----------------------------
    def _euclidean(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def _calculate_mar(self, upper, lower, left, right):
        """
        MAR = vertical_distance / horizontal_distance
        """
        vertical = self._euclidean(upper, lower)
        horizontal = self._euclidean(left, right)

        return vertical / horizontal if horizontal > 0 else 0

    # -----------------------------
    # Main processing function
    # -----------------------------
    def process_frame(
        self,
        image: np.ndarray,
        frame_id: int,
        timestamp: Optional[float] = None
    ) -> Dict:

        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            return {"face_detected": False}

        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        # Convert to pixel coords
        pts = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks])

        # -----------------------------
        # Extract lip key points
        # -----------------------------
        upper = pts[self.UPPER_LIP]
        lower = pts[self.LOWER_LIP]
        left = pts[self.MOUTH_LEFT]
        right = pts[self.MOUTH_RIGHT]

        # Full lip contour (optional)
        lip_coords = pts[self.LIPS]

        # -----------------------------
        # MAR calculation
        # -----------------------------
        mar = self._calculate_mar(upper, lower, left, right)

        # MAR reflects vertical lip opening vs horizontal width :contentReference[oaicite:1]{index=1}

        # -----------------------------
        # State machine (open/close)
        # -----------------------------
        duration = None

        if mar > self.mar_threshold:
            if self.state == "closed":
                # Transition → OPEN
                self.state = "open"
                self.open_start_frame = frame_id
                open_timestamp = timestamp
            else:
                open_timestamp = None
        else:
            if self.state == "open":
                # Transition → CLOSED
                self.state = "closed"

                if self.open_start_frame is not None:
                    duration = frame_id - self.open_start_frame

                self.open_start_frame = None
            open_timestamp = None

        # -----------------------------
        # Output
        # -----------------------------
        return {
            "face_detected": True,
            "mouth": {
                "MAR": float(mar),
                "state": self.state,
                "open_start_frame": self.open_start_frame,
                "duration_frames": duration,
                "timestamp": timestamp
            },
            "lips": {
                "upper": upper.tolist(),
                "lower": lower.tolist(),
                "left": left.tolist(),
                "right": right.tolist(),
                "contour": lip_coords.tolist()
            }
        }