"""
tests/test_associator.py — Unit Tests for PPE Body-Zone IoU Matching
=====================================================================
Tests the core spatial association logic that maps PPE detections
to specific workers based on anatomical body zones.
"""

import sys
from pathlib import Path

# Ensure project root is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.associator import PPEAssociator, ComplianceStatus
from core.detector import Detection


class TestPPEAssociator:
    """Tests for the PPEAssociator class."""

    def setup_method(self):
        self.associator = PPEAssociator()

    # ── Body zone geometry ────────────────────────────────

    def test_body_zones_proportions(self):
        """Body zones must cover the correct vertical proportions."""
        bbox = [100, 100, 200, 500]  # 100w × 400h person
        zones = self.associator.get_body_zones(bbox)

        h = 400  # y2 - y1

        # Head zone: top 22%
        head = zones["head_zone"]
        assert head[1] == 100                # starts at y1
        assert abs(head[3] - (100 + h * 0.22)) < 1e-6  # ends at y1 + 22%

        # Torso zone: 22% to 70%
        torso = zones["torso_zone"]
        assert abs(torso[1] - (100 + h * 0.22)) < 1e-6
        assert abs(torso[3] - (100 + h * 0.70)) < 1e-6

        # Lateral zone: 60% to bottom
        lateral = zones["lateral_zone"]
        assert abs(lateral[1] - (100 + h * 0.60)) < 1e-6
        assert lateral[3] == 500             # ends at y2

        # Full body: entire bbox
        assert zones["full_body"] == [100, 100, 200, 500]

    # ── IoU computation ───────────────────────────────────

    def test_iou_identical_boxes(self):
        """Identical boxes must have IoU ≈ 1.0."""
        box = [10, 10, 50, 50]
        iou = PPEAssociator.compute_iou(box, box)
        assert abs(iou - 1.0) < 1e-4

    def test_iou_no_overlap(self):
        """Non-overlapping boxes must have IoU ≈ 0.0."""
        box_a = [0, 0, 10, 10]
        box_b = [100, 100, 200, 200]
        iou = PPEAssociator.compute_iou(box_a, box_b)
        assert iou < 1e-4

    def test_iou_partial_overlap(self):
        """Partially overlapping boxes yield 0 < IoU < 1."""
        box_a = [0, 0, 20, 20]   # area 400
        box_b = [10, 10, 30, 30] # area 400, overlap 10×10=100
        iou = PPEAssociator.compute_iou(box_a, box_b)
        # Union = 400 + 400 - 100 = 700, IoU = 100/700 ≈ 0.1429
        assert 0.14 < iou < 0.15

    # ── Full association ──────────────────────────────────

    def test_associate_helmet_in_head_zone(self):
        """A helmet overlapping the head zone → helmet_worn = True."""
        person_bbox = [100, 100, 200, 500]
        zones = self.associator.get_body_zones(person_bbox)
        head = zones["head_zone"]  # [100, 100, 200, 188]

        # Place a helmet detection overlapping the head zone
        helmet_det = Detection(
            class_name="helmet",
            confidence=0.85,
            bbox=[110, 105, 190, 185],
            center=(150, 145),
        )
        status = self.associator.associate(person_bbox, [helmet_det])
        assert status.helmet_worn is True
        assert "NO_HELMET" not in status.violations

    def test_associate_no_ppe_detected(self):
        """No PPE detections near a person → all violations flagged."""
        person_bbox = [100, 100, 200, 500]
        # No PPE detections at all
        status = self.associator.associate(person_bbox, [])
        assert status.helmet_worn is False
        assert status.vest_worn is False
        assert "NO_HELMET" in status.violations
        assert "NO_VEST" in status.violations

    def test_negative_class_overrides_positive(self):
        """Explicit 'no_helmet' near person overrides any positive match."""
        person_bbox = [100, 100, 200, 500]
        no_helmet_det = Detection(
            class_name="no_helmet",
            confidence=0.90,
            bbox=[105, 105, 195, 190],
            center=(150, 147),
        )
        status = self.associator.associate(person_bbox, [no_helmet_det])
        assert status.helmet_worn is False
        assert "NO_HELMET" in status.violations


# Allow running with: python -m pytest tests/test_associator.py
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
