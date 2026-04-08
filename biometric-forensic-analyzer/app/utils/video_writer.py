# src/core/utils/video_writer.py

import os
import cv2
from typing import Dict, Optional, List


def _get_sorted_frames(frames_dir: str) -> List[str]:
    frames = [
        f for f in os.listdir(frames_dir)
        if f.endswith(".png")
    ]
    frames.sort()
    return [os.path.join(frames_dir, f) for f in frames]


def _draw_mesh(frame, landmarks):
    for lm in landmarks:
        x, y = int(lm["x"]), int(lm["y"])
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def _draw_tongue(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cx, cy = map(int, det["center"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)


def _draw_gaze(frame, eyes_data):
    for side in ["left", "right"]:
        eye = eyes_data.get(side)
        if not eye:
            continue

        center = eye.get("pupil_center")
        delta = eye.get("pupil_delta")

        if center and delta:
            cx, cy = int(center[0]), int(center[1])
            dx, dy = int(delta[0] * 5), int(delta[1] * 5)  # scale for visibility

            cv2.arrowedLine(
                frame,
                (cx, cy),
                (cx + dx, cy + dy),
                (255, 0, 0),
                2
            )


def _draw_text(frame, text: str, position=(10, 30)):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def stitch_video(
    frames_dir: str,
    output_path: str,
    fps: int = 30,
    overlay_data: Optional[Dict] = None
):
    """
    Stitch PNG frames into MP4 video with optional overlays.

    Args:
        frames_dir: directory containing PNG frames
        output_path: output video file path
        fps: frames per second
        overlay_data: biometric JSON (timeline)
    """

    frame_paths = _get_sorted_frames(frames_dir)

    if not frame_paths:
        raise ValueError("No frames found in directory")

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    # Video writer (MP4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    timeline = overlay_data.get("timeline") if overlay_data else None

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)

        # -----------------------------
        # Overlay logic
        # -----------------------------
        if timeline and idx < len(timeline):
            data = timeline[idx]

            # Mesh (468 landmarks)
            if "landmarks" in data:
                _draw_mesh(frame, data["landmarks"])

            # Tongue detections
            if "tongue_detected" in data and data["tongue_detected"]:
                _draw_tongue(frame, data.get("detections", []))

            # Eye gaze
            if "eyes" in data:
                _draw_gaze(frame, data["eyes"])

            # Optional debug text
            _draw_text(frame, f"Frame: {idx}")

        # -----------------------------
        # Write frame
        # -----------------------------
        writer.write(frame)

    writer.release()