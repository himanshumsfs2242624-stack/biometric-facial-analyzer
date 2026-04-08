# src/core/forensic/metadata.py

import os
from typing import Dict, Any
import exiftool


def _detect_post_processing(metadata: Dict[str, Any]) -> Dict[str, bool]:
    """
    Detect signs of post-processing tools based on metadata tags
    """

    flags = {
        "adobe_premiere": False,
        "adobe_photoshop": False,
        "ffmpeg": False
    }

    # Combine all values into one searchable string
    metadata_str = " ".join([str(v).lower() for v in metadata.values() if v])

    # Detection patterns
    if "adobe premiere" in metadata_str:
        flags["adobe_premiere"] = True

    if "photoshop" in metadata_str or "adobe photoshop" in metadata_str:
        flags["adobe_photoshop"] = True

    if "ffmpeg" in metadata_str or "lavf" in metadata_str:
        flags["ffmpeg"] = True

    return flags


def _group_metadata(metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Group metadata by category (EXIF, XMP, QuickTime, etc.)
    """

    grouped = {}

    for key, value in metadata.items():
        if ":" in key:
            group, tag = key.split(":", 1)
        else:
            group, tag = "UNKNOWN", key

        if group not in grouped:
            grouped[group] = {}

        grouped[group][tag] = value

    return grouped


def extract_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extract all metadata using ExifTool via pyexiftool.

    Returns:
        Structured dictionary with:
        - grouped metadata (EXIF, XMP, etc.)
        - forensic flags (Adobe, FFmpeg, etc.)
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    try:
        # Use ExifToolHelper (recommended wrapper) :contentReference[oaicite:1]{index=1}
        with exiftool.ExifToolHelper() as et:
            metadata_list = et.get_metadata(video_path)

        if not metadata_list:
            return {
                "metadata": {},
                "grouped": {},
                "forensic_flags": {}
            }

        metadata = metadata_list[0]

        # Group metadata (EXIF, XMP, QuickTime, etc.)
        grouped_metadata = _group_metadata(metadata)

        # Detect post-processing tools
        forensic_flags = _detect_post_processing(metadata)

        # Extract key forensic-relevant fields
        key_fields = {
            "file": {
                "file_name": metadata.get("File:FileName"),
                "file_size": metadata.get("File:FileSize"),
                "mime_type": metadata.get("File:MIMEType"),
            },
            "timestamps": {
                "created": metadata.get("EXIF:CreateDate") or metadata.get("QuickTime:CreateDate"),
                "modified": metadata.get("File:FileModifyDate"),
            },
            "gps": {
                "latitude": metadata.get("Composite:GPSLatitude"),
                "longitude": metadata.get("Composite:GPSLongitude"),
            },
            "codec": {
                "format": metadata.get("File:FileType"),
                "video_codec": metadata.get("QuickTime:CompressorName"),
                "encoder": metadata.get("QuickTime:Encoder"),
            }
        }

        return {
            "metadata": metadata,              # raw full metadata
            "grouped": grouped_metadata,      # grouped by type
            "key_fields": key_fields,         # important forensic fields
            "forensic_flags": forensic_flags  # tampering indicators
        }

    except Exception as e:
        raise RuntimeError(f"Metadata extraction failed: {str(e)}")