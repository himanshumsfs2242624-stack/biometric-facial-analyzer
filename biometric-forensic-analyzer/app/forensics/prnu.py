# src/core/forensic/prnu.py

import numpy as np
import pywt
import cv2
from typing import Dict


def _wavelet_denoise(image: np.ndarray, wavelet: str = "db2", level: int = 2):
    """
    Perform multi-level wavelet decomposition and suppress noise
    """

    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)

    # Zero out detail coefficients (denoising step)
    denoised_coeffs = [coeffs[0]]  # approximation coefficients

    for detail_level in coeffs[1:]:
        denoised_details = tuple(
            pywt.threshold(subband, value=np.std(subband), mode="soft")
            for subband in detail_level
        )
        denoised_coeffs.append(denoised_details)

    reconstructed = pywt.waverec2(denoised_coeffs, wavelet=wavelet)

    return reconstructed


def _wiener_filter(image: np.ndarray, kernel_size: int = 5):
    """
    Apply Wiener filtering using OpenCV approximation
    """

    # Estimate local mean
    local_mean = cv2.blur(image, (kernel_size, kernel_size))

    # Estimate local variance
    local_var = cv2.blur(image**2, (kernel_size, kernel_size)) - local_mean**2

    noise_var = np.mean(local_var)

    # Wiener filter formula
    result = local_mean + (np.maximum(local_var - noise_var, 0) / (local_var + 1e-8)) * (image - local_mean)

    return result


def extract_sensor_noise(image_array: np.ndarray) -> Dict:
    """
    Extract PRNU (sensor noise) from an image.

    Steps:
    1. Convert to grayscale
    2. Wavelet-based denoising
    3. Wiener filtering
    4. Subtract denoised image from original

    Returns:
        dict:
            - noise_residue (np.ndarray)
    """

    # -----------------------------
    # Step 1: Convert to grayscale
    # -----------------------------
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array.copy()

    gray = gray.astype(np.float32)

    # -----------------------------
    # Step 2: Wavelet denoising
    # -----------------------------
    wavelet_denoised = _wavelet_denoise(gray)

    # Ensure same shape
    wavelet_denoised = wavelet_denoised[: gray.shape[0], : gray.shape[1]]

    # -----------------------------
    # Step 3: Wiener filtering
    # -----------------------------
    wiener_denoised = _wiener_filter(wavelet_denoised)

    # -----------------------------
    # Step 4: Extract noise residue
    # -----------------------------
    noise_residue = gray - wiener_denoised

    # Normalize for stability
    noise_residue = noise_residue - np.mean(noise_residue)
    std = np.std(noise_residue)
    if std > 0:
        noise_residue = noise_residue / std

    return {
        "noise_residue": noise_residue
    }