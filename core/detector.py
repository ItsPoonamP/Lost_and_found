"""
detector.py — YOLOv11s + BoT-SORT tracking.

detect_and_track(frame) returns:
    detections: list of dicts with track_id, class_name, bbox, confidence, crop
    annotated:  frame with boxes and labels drawn by Ultralytics
"""

from ultralytics import YOLO
import numpy as np
import config


class Detector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)
        self.class_ids = list(config.TRACKED_CLASSES.keys())

    def detect_and_track(self, frame: np.ndarray):
        """
        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            detections (list[dict]), annotated_frame (np.ndarray)
        """
        results = self.model.track(
            frame,
            persist=True,              # keeps Kalman state across calls
            tracker=config.TRACKER,    # botsort.yaml
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=self.class_ids,
            verbose=False,
        )

        detections = []
        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return detections, result.plot()

        for box in result.boxes:
            if box.id is None:
                continue

            track_id   = int(box.id.item())
            class_id   = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            # Crop the object region for embedding
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            detections.append({
                "track_id":   track_id,
                "class_id":   class_id,
                "class_name": config.TRACKED_CLASSES.get(class_id, "unknown"),
                "bbox":       [x1, y1, x2, y2],
                "confidence": confidence,
                "crop":       crop,
            })

        return detections, result.plot()
