"""
frame_utils.py — Helpers for saving frames and drawing overlays.
"""

import cv2
import numpy as np
import os
from datetime import datetime
import config


def save_frame(frame: np.ndarray, track_id: int, class_name: str) -> str:
    """
    Save the full frame to disk.
    Returns the absolute path to the saved file.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{class_name}_id{track_id}_{ts}.jpg"
    path = os.path.join(config.FRAMES_DIR, filename)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def get_color(track_id: int) -> tuple:
    """Deterministic but visually distinct color per track ID."""
    np.random.seed(int(track_id) % 1000)
    return tuple(int(c) for c in np.random.randint(80, 230, 3))


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes + labels on frame.
    (Optional: Ultralytics result.plot() also does this; use this for custom style.)
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        tid  = det["track_id"]
        name = det["class_name"]
        conf = det["confidence"]
        color = get_color(tid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{tid} {name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return frame


def add_hud(frame: np.ndarray, n_tracked: int, db_count: int) -> np.ndarray:
    """Add a semi-transparent HUD overlay with live stats."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    ts = datetime.now().strftime("%H:%M:%S")
    lines = [
        f"Time     : {ts}",
        f"Tracking : {n_tracked} object(s)",
        f"DB saved : {db_count} observations",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 22 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 1)

    return frame
