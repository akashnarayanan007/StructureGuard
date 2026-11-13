"""
core/detector.py — YOLOv8 Detection Wrapper
=============================================
Wraps the Ultralytics YOLOv8 model for construction site
object detection. Returns structured Detection dataclass
objects instead of raw tensors.

The default yolov8n.pt detects COCO classes (80 classes
including 'person'). For PPE-specific detection (helmet,
vest, harness, gloves) a fine-tuned model trained on a
construction safety dataset (e.g. Roboflow) should be
substituted via config.yaml model.weights.

NOTE: With the default COCO model, only 'person' detections
are meaningful. PPE association then relies on spatial
heuristics in the associator module.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    """Single detection result from YOLOv8."""
    class_name: str
    confidence: float
    bbox: List[float]          # [x1, y1, x2, y2] absolute pixel coords
    center: Tuple[float, float]  # (cx, cy) centroid of bounding box


class SafetyDetector:
    """YOLOv8-based detector for persons and PPE on construction sites."""

    def __init__(self, config: dict):
        model_cfg = config["model"]
        weights_path = model_cfg["weights"]

        # Auto-detect GPU availability; config can override to force CPU
        configured_device = model_cfg.get("device", "cpu")
        if configured_device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif configured_device == "cpu":
            self.device = "cpu"
        else:
            # Fallback: use GPU if available, else CPU
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ultralytics auto-downloads yolov8n.pt if the file doesn't exist
        self.model = YOLO(weights_path)

        # Detection thresholds from config — avoids hardcoding
        self.conf_threshold = model_cfg["confidence_threshold"]
        self.iou_threshold = model_cfg["iou_threshold"]

        # Build a mapping from class index to human-readable name.
        # For COCO model this gives 80 classes; for custom PPE model
        # it maps to the user-defined class list.
        self.class_names = self.model.names  # dict {int: str}

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 inference on a single frame.

        Returns a list of Detection dataclass objects, each containing
        the class name, confidence score, bounding box in absolute
        pixel coordinates, and the centroid.
        """
        # Run inference with configured thresholds
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,  # Suppress per-frame console spam
        )

        detections: List[Detection] = []

        # results is a list (one entry per image); we send one frame
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Extract bounding box as [x1, y1, x2, y2] in pixel coords
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = xyxy

                # Centroid for tracking and zone checks
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Class index → name
                cls_idx = int(box.cls[0].cpu().numpy())
                cls_name = self.class_names.get(cls_idx, f"class_{cls_idx}")

                conf = float(box.conf[0].cpu().numpy())

                detections.append(Detection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    center=(cx, cy),
                ))

        return detections
