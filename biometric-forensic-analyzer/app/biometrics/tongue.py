# src/core/biometric/tongue.py

import os
import numpy as np
from typing import Dict, Optional
from ultralytics import YOLO


class TongueTracker:
    def __init__(self, model_path: str = "models/custom_yolo_tongue.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load YOLOv8 model
        self.model = YOLO(model_path)

    # -----------------------------
    # Utility functions
    # -----------------------------
    def _crop_lower_face(self, image: np.ndarray):
        """
        Crop lower half of the image (mouth/tongue region)
        """
        h, w, _ = image.shape
        cropped = image[h // 2 :, :]  # lower half
        return cropped, h // 2  # offset for y correction

    def _process_detections(self, results, y_offset: int):
        """
        Extract tongue detection info
        """
        detections = []

        if not results or len(results[0].boxes) == 0:
            return None

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

            # Only keep 'tongue' class (assuming class 0 = tongue)
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = xyxy

            # Adjust y back to original image space
            y1 += y_offset
            y2 += y_offset

            # Center point
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": conf,
                "center": [float(cx), float(cy)]
            })

        return detections if detections else None

    # -----------------------------
    # Main processing function
    # -----------------------------
    def process_frame(self, image_array: np.ndarray) -> Dict:
        """
        Detect tongue in frame using YOLOv8

        Returns:
        - bounding box
        - confidence
        - center point
        """

        # Step 1: Crop lower face
        cropped, y_offset = self._crop_lower_face(image_array)

        # Step 2: Run YOLO inference
        results = self.model(cropped, verbose=False)

        # Step 3: Process detections
        detections = self._process_detections(results, y_offset)

        if detections is None:
            return {
                "tongue_detected": False,
                "detections": []
            }

        return {
            "tongue_detected": True,
            "detections": detections
        }