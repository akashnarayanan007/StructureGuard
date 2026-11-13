"""
core/associator.py — Person-PPE Body-Zone IoU Matching
=======================================================
THE MOST CRITICAL AND NOVEL MODULE in StructureGuard.

Problem:
  YOLOv8 detects persons and PPE items as separate bounding boxes.
  We need to answer: "Is Worker #3 wearing a helmet?" — not just
  "Is there a helmet somewhere in the frame."

Solution:
  Divide each person's bounding box into anatomical body zones
  (head, torso, lateral, full-body) then compute IoU between each
  PPE detection and the appropriate zone. A helmet near the head
  zone (IoU > threshold) means that person is wearing a helmet.

  This spatial association is what turns generic object detection
  into per-worker compliance checking.

Body-zone proportions (empirically calibrated):
  ┌──────────────┐
  │   HEAD       │  top 22% of person height
  │  (helmet)    │
  ├──────────────┤
  │   TORSO      │  22%–70% of person height
  │  (vest)      │
  ├──────────────┤
  │   LATERAL    │  60%–100% (overlaps torso bottom)
  │  (gloves)    │
  └──────────────┘
  Full body: used for harness (spans entire person)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class ComplianceStatus:
    """Per-worker PPE compliance result."""
    helmet_worn: bool = True
    vest_worn: bool = True
    harness_worn: bool = True
    gloves_worn: bool = True
    violations: List[str] = field(default_factory=list)


class PPEAssociator:
    """Associates PPE detections to specific workers via body-zone IoU."""

    # PPE class → body zone mapping and IoU thresholds
    # Lower IoU thresholds because PPE boxes often only partially
    # overlap their anatomical zone (e.g., a helmet sits atop the head)
    PPE_ZONE_MAP = {
        "helmet":      {"zone": "head_zone",    "iou_thresh": 0.15, "attribute": "helmet_worn"},
        "safety_vest": {"zone": "torso_zone",   "iou_thresh": 0.15, "attribute": "vest_worn"},
        "harness":     {"zone": "full_body",    "iou_thresh": 0.10, "attribute": "harness_worn"},
        "gloves":      {"zone": "lateral_zone", "iou_thresh": 0.10, "attribute": "gloves_worn"},
    }

    # Explicit "not worn" classes — if detected near the person,
    # they override any positive PPE detection (model is saying
    # it sees a head without a helmet, not just absence of helmet)
    NEGATIVE_CLASSES = {
        "no_helmet":  "helmet_worn",
        "no_vest":    "vest_worn",
        "no_harness": "harness_worn",
    }

    def get_body_zones(self, person_bbox: List[float]) -> Dict[str, List[float]]:
        """
        Divide a person bounding box into anatomical zones.

        Args:
            person_bbox: [x1, y1, x2, y2] of the detected person.

        Returns:
            dict mapping zone name → [x1, y1, x2, y2] of that zone.
        """
        x1, y1, x2, y2 = person_bbox
        h = y2 - y1

        return {
            # Top 22% — where helmets and hard hats appear
            "head_zone":    [x1, y1,            x2, y1 + h * 0.22],
            # Middle band — where vests and hi-vis jackets appear
            "torso_zone":   [x1, y1 + h * 0.22, x2, y1 + h * 0.70],
            # Lower 40% — wrists/hands for gloves, overlaps torso bottom
            "lateral_zone": [x1, y1 + h * 0.60, x2, y2],
            # Full body — for harnesses that span head-to-toe
            "full_body":    [x1, y1,            x2, y2],
        }

    @staticmethod
    def compute_iou(box_a: List[float], box_b: List[float]) -> float:
        """
        Compute Intersection over Union between two axis-aligned boxes.

        Uses epsilon guard (1e-6) to prevent division by zero when
        two boxes share zero area (e.g., a degenerate detection).

        Args:
            box_a, box_b: [x1, y1, x2, y2] bounding boxes.

        Returns:
            IoU value in [0.0, 1.0].
        """
        # Intersection rectangle
        inter_x1 = max(box_a[0], box_b[0])
        inter_y1 = max(box_a[1], box_b[1])
        inter_x2 = min(box_a[2], box_b[2])
        inter_y2 = min(box_a[3], box_b[3])

        # Clamp to zero if no overlap
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

        # Areas of each box
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        # Union with epsilon guard against division by zero
        union_area = area_a + area_b - inter_area
        return inter_area / (union_area + 1e-6)

    def associate(self, person_bbox: List[float], all_detections: list) -> ComplianceStatus:
        """
        Determine which PPE items a specific worker is wearing.

        For each body zone, find PPE detections that overlap above
        the IoU threshold. Also check for explicit negative-class
        detections (no_helmet, no_vest) that override positives.

        Args:
            person_bbox: [x1, y1, x2, y2] of the tracked worker.
            all_detections: All Detection objects from this frame.

        Returns:
            ComplianceStatus with per-item worn flags and violation list.
        """
        zones = self.get_body_zones(person_bbox)
        status = ComplianceStatus()

        # --- Step 1: Check positive PPE matches via IoU ---
        # Start by assuming nothing is worn; set to True only if matched
        status.helmet_worn = False
        status.vest_worn = False
        status.harness_worn = False
        status.gloves_worn = False

        for det in all_detections:
            if det.class_name in self.PPE_ZONE_MAP:
                mapping = self.PPE_ZONE_MAP[det.class_name]
                zone_bbox = zones[mapping["zone"]]
                iou = self.compute_iou(zone_bbox, det.bbox)

                # If IoU exceeds threshold, this PPE item belongs to this worker
                if iou > mapping["iou_thresh"]:
                    setattr(status, mapping["attribute"], True)

        # --- Step 2: Override with explicit negative-class detections ---
        # If the model explicitly detects "no_helmet" near this person,
        # that takes priority over absence of a positive helmet detection
        for det in all_detections:
            if det.class_name in self.NEGATIVE_CLASSES:
                # Check if the negative detection overlaps the person at all
                iou = self.compute_iou(person_bbox, det.bbox)
                if iou > 0.10:
                    attribute = self.NEGATIVE_CLASSES[det.class_name]
                    setattr(status, attribute, False)

        # --- Step 3: Build the violations list ---
        if not status.helmet_worn:
            status.violations.append("NO_HELMET")
        if not status.vest_worn:
            status.violations.append("NO_VEST")
        if not status.harness_worn:
            status.violations.append("NO_HARNESS")
        if not status.gloves_worn:
            status.violations.append("NO_GLOVES")

        return status
