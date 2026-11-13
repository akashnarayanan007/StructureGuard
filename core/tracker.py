"""
core/tracker.py — DeepSORT Multi-Object Tracker Wrapper
========================================================
Persistent identity tracking across video frames using
DeepSORT (Deep Simple Online and Realtime Tracking).

DeepSORT combines:
  • Kalman Filter — predicts the next bounding box position using
    a constant-velocity motion model, bridging short occlusions.
  • Hungarian Algorithm — solves the optimal assignment between
    predicted tracks and new detections by minimizing a cost
    matrix of Mahalanobis + appearance distances.
  • Deep Appearance Descriptor — a Re-ID CNN embedding that helps
    re-associate a worker after they're occluded or leave the frame
    briefly, reducing identity switches.

We filter to 'person' detections only before feeding DeepSORT,
because PPE items (helmets, vests) should not be tracked as
independent entities — they are attributes of a tracked person.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


@dataclass
class Track:
    """A confirmed person track with persistent identity."""
    track_id: int
    bbox: List[float]              # [x1, y1, x2, y2]
    center: Tuple[float, float]    # (cx, cy)
    is_confirmed: bool


class WorkerTracker:
    """Wraps DeepSORT for persistent worker identity tracking."""

    def __init__(self, config: dict):
        tracking_cfg = config["tracking"]

        # max_age: frames to keep a track alive without re-association
        # n_init: detection hits needed before a track is "confirmed"
        # max_iou_distance: IoU threshold for gating associations
        self.tracker = DeepSort(
            max_age=tracking_cfg["max_age"],
            n_init=tracking_cfg["n_init"],
            max_iou_distance=tracking_cfg["max_iou_distance"],
        )

    def update(self, detections: list, frame: np.ndarray) -> List[Track]:
        """
        Feed new-frame detections to DeepSORT and return confirmed tracks.

        Only 'person' detections are sent to the tracker — PPE items
        are associated to persons separately by the PPEAssociator.

        Args:
            detections: List of Detection dataclass objects from SafetyDetector.
            frame: The current BGR frame (DeepSORT uses it for appearance features).

        Returns:
            List of confirmed Track objects with persistent IDs.
        """
        # Filter to person detections only
        person_dets = [d for d in detections if d.class_name == "person"]

        # DeepSORT expects detections as list of ([x1, y1, w, h], confidence, class)
        raw_detections = []
        for det in person_dets:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            raw_detections.append(
                ([x1, y1, w, h], det.confidence, det.class_name)
            )

        # Update the tracker — internally runs Kalman predict + Hungarian match
        tracked_objects = self.tracker.update_tracks(
            raw_detections, frame=frame
        )

        tracks: List[Track] = []
        for obj in tracked_objects:
            # Only return confirmed tracks (past n_init threshold)
            if not obj.is_confirmed():
                continue

            # DeepSORT returns ltrb (left, top, right, bottom)
            ltrb = obj.to_ltrb()
            x1, y1, x2, y2 = ltrb
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            try:
                tid = int(obj.track_id)
            except (ValueError, TypeError):
                tid = obj.track_id

            tracks.append(Track(
                track_id=tid,
                bbox=[x1, y1, x2, y2],
                center=(cx, cy),
                is_confirmed=True,
            ))

        return tracks
