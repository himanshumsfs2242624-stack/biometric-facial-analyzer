# src/core/frame_extractor.py

import os
import subprocess
from typing import List


def extract_frames(video_path: str, output_dir: str) -> List[str]:
    """
    Extract all frames from a video as uncompressed 24-bit PNG images.

    Args:
        video_path (str): Path to input video
        output_dir (str): Directory to store extracted frames

    Returns:
        List[str]: Sorted list of frame file paths
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Output pattern (e.g., 000001.png, 000002.png)
    output_pattern = os.path.join(output_dir, "%06d.png")

    # FFmpeg command:
    # -i input video
    # -vsync 0 → avoid duplicate frames
    # -pix_fmt rgb24 → 24-bit RGB (uncompressed PNG)
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vsync", "0",
        "-pix_fmt", "rgb24",
        output_pattern
    ]

    # Run FFmpeg
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr.decode()}")

    # Collect and return sorted frame paths
    frames = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".png")
    ]

    frames.sort()  # ensure chronological order

    return frames