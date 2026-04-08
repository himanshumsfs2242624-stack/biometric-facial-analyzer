# src/core/forensic/ela.py

import numpy as np
from io import BytesIO
from typing import Tuple, Dict
from PIL import Image, ImageChops, ImageEnhance


def perform_ela(image_array: np.ndarray, quality: int = 90) -> Dict:
    """
    Perform Error Level Analysis (ELA) on an image.

    Steps:
    1. Convert numpy image → PIL
    2. Recompress as JPEG (lossy)
    3. Compute absolute difference
    4. Enhance brightness
    5. Return ELA mask + discrepancy score

    Args:
        image_array (np.ndarray): Input image (H, W, C)
        quality (int): JPEG compression quality

    Returns:
        dict:
            - ela_image (np.ndarray)
            - max_discrepancy (float)
    """

    # -----------------------------
    # Step 1: Convert to PIL
    # -----------------------------
    original = Image.fromarray(image_array.astype("uint8")).convert("RGB")

    # -----------------------------
    # Step 2: Save to in-memory JPEG
    # -----------------------------
    buffer = BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    compressed = Image.open(buffer)

    # -----------------------------
    # Step 3: Compute difference
    # -----------------------------
    diff = ImageChops.difference(original, compressed)

    # Convert to numpy for scoring
    diff_np = np.array(diff).astype(np.float32)

    # -----------------------------
    # Step 4: Compute discrepancy score
    # -----------------------------
    max_discrepancy = float(np.max(diff_np))

    # Avoid division by zero
    scale = 255.0 / max_discrepancy if max_discrepancy != 0 else 1.0

    # -----------------------------
    # Step 5: Enhance brightness
    # -----------------------------
    enhancer = ImageEnhance.Brightness(diff)
    ela_image = enhancer.enhance(scale)

    ela_np = np.array(ela_image)

    # -----------------------------
    # Output
    # -----------------------------
    return {
        "ela_image": ela_np,
        "max_discrepancy": max_discrepancy
    }