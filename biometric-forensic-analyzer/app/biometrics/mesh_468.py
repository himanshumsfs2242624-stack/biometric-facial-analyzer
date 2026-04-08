# src/core/biometric/mesh_468.py

import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional


class FaceMeshTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # -----------------------------
    # Landmark region definitions
    # -----------------------------
    # NOTE: These are curated subsets (not exhaustive)
    # MediaPipe gives 468 landmarks total :contentReference[oaicite:0]{index=0}

    NOSTRIL_IDX = [1, 2, 98, 327]  # around nose tip & nostrils
    CHEEK_IDX = [50, 280]          # right cheek, left cheek :contentReference[oaicite:1]{index=1}
    CHIN_IDX = [152]               # chin bottom
    EAR_IDX = [234, 454]           # approximate ear regions (side face)

    def _extract_landmarks(self, landmarks, image_shape) -> List[Dict]:
        h, w, _ = image_shape

        points = []
        for idx, lm in enumerate(landmarks):
            points.append({
                "id": idx,
                "x": lm.x * w,
                "y": lm.y * h,
                "z": lm.z
            })
        return points

    def _get_region(self, landmarks, indices):
        return [landmarks[i] for i in indices if i < len(landmarks)]

    def _compute_delta(self, current_pts, prev_pts):
        if prev_pts is None:
            return None

        deltas = []
        for c, p in zip(current_pts, prev_pts):
            deltas.append({
                "dx": c["x"] - p["x"],
                "dy": c["y"] - p["y"]
            })
        return deltas

    # -----------------------------
    # Main processing function
    # -----------------------------
    def process_frame(
        self,
        image_array: np.ndarray,
        previous_frame_landmarks: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Process a frame and extract:
        - 468 landmarks
        - facial regions
        - movement deltas (chin + cheeks)
        """

        results = self.face_mesh.process(image_array)

        if not results.multi_face_landmarks:
            return {"face_detected": False}

        face_landmarks = results.multi_face_landmarks[0].landmark

        # Extract all 468 landmarks
        landmarks = self._extract_landmarks(face_landmarks, image_array.shape)

        # -----------------------------
        # Region extraction
        # -----------------------------
        nostrils = self._get_region(landmarks, self.NOSTRIL_IDX)
        cheeks = self._get_region(landmarks, self.CHEEK_IDX)
        chin = self._get_region(landmarks, self.CHIN_IDX)
        ears = self._get_region(landmarks, self.EAR_IDX)

        # -----------------------------
        # Previous frame region extraction
        # -----------------------------
        prev_cheeks = None
        prev_chin = None

        if previous_frame_landmarks:
            prev_cheeks = self._get_region(previous_frame_landmarks, self.CHEEK_IDX)
            prev_chin = self._get_region(previous_frame_landmarks, self.CHIN_IDX)

        # -----------------------------
        # Movement deltas
        # -----------------------------
        cheek_movement = self._compute_delta(cheeks, prev_cheeks)
        chin_movement = self._compute_delta(chin, prev_chin)

        # -----------------------------
        # Final output
        # -----------------------------
        return {
            "face_detected": True,
            "landmarks": landmarks,  # all 468
            "regions": {
                "nostrils": nostrils,
                "cheeks": cheeks,
                "chin": chin,
                "ears": ears
            },
            "movement": {
                "cheeks_delta": cheek_movement,
                "chin_delta": chin_movement
            }
        }