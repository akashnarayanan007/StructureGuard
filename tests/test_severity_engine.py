"""
tests/test_severity_engine.py — Unit Tests for Severity Scoring
==================================================================
Tests the weighted severity scoring formula and level classification.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.severity_engine import SeverityEngine


class TestSeverityEngine:
    """Tests for the SeverityEngine class."""

    def setup_method(self):
        """Create a SeverityEngine with known config values."""
        self.config = {
            "severity": {
                "no_helmet_base": 80,
                "no_vest_base": 50,
                "no_harness_base": 90,
                "no_gloves_base": 30,
                "zone_multiplier": 1.5,
                "confidence_weight": True,
                "thresholds": {
                    "critical": 80,
                    "high": 55,
                    "medium": 35,
                    "low": 0,
                },
            }
        }
        self.engine = SeverityEngine(self.config)

    # ── No violations ─────────────────────────────────────

    def test_no_violations_returns_low(self):
        """Empty violation list → score 0, level COMPLIANT."""
        result = self.engine.score([], [], 0.9)
        assert result.score == 0.0
        assert result.level == "LOW"
        assert result.label == "COMPLIANT"

    # ── Single violations ─────────────────────────────────

    def test_no_helmet_critical(self):
        """NO_HELMET (base 80) at full confidence → CRITICAL."""
        result = self.engine.score(["NO_HELMET"], [], 1.0)
        # base=80, no zone, conf=1.0 → score=80 → CRITICAL
        assert result.score == 80.0
        assert result.level == "CRITICAL"

    def test_no_vest_high(self):
        """NO_VEST (base 50) at full confidence → MEDIUM (50 < 55)."""
        result = self.engine.score(["NO_VEST"], [], 1.0)
        # base=50, no zone, conf=1.0 → score=50 → MEDIUM
        assert result.score == 50.0
        assert result.level == "MEDIUM"

    def test_no_gloves_low(self):
        """NO_GLOVES (base 30) at full confidence → LOW (30 < 35)."""
        result = self.engine.score(["NO_GLOVES"], [], 1.0)
        assert result.score == 30.0
        assert result.level == "LOW"

    # ── Zone multiplier ───────────────────────────────────

    def test_zone_multiplier_applied(self):
        """Being in a danger zone multiplies the base score by 1.5."""
        result_safe = self.engine.score(["NO_VEST"], [], 1.0)
        result_zone = self.engine.score(["NO_VEST"], ["Machinery Zone"], 1.0)

        # 50 * 1.5 = 75 in zone vs 50 outside
        assert result_zone.score == 75.0
        assert result_zone.score == result_safe.score * 1.5

    def test_zone_escalates_severity(self):
        """Zone multiplier can push a MEDIUM violation to HIGH."""
        # NO_VEST = 50 (MEDIUM), in zone → 75 (HIGH)
        result = self.engine.score(["NO_VEST"], ["Machinery Zone"], 1.0)
        assert result.level == "HIGH"

    # ── Confidence weighting ──────────────────────────────

    def test_confidence_scales_score(self):
        """Lower detection confidence reduces the severity score."""
        result_full = self.engine.score(["NO_HELMET"], [], 1.0)
        result_half = self.engine.score(["NO_HELMET"], [], 0.5)

        # 80 * 0.5 = 40 → MEDIUM instead of CRITICAL
        assert result_half.score == 40.0
        assert result_half.score < result_full.score

    # ── Multiple violations ───────────────────────────────

    def test_multiple_violations_sum(self):
        """Multiple violations sum their base scores."""
        result = self.engine.score(
            ["NO_HELMET", "NO_VEST"], [], 1.0
        )
        # 80 + 50 = 130 → CRITICAL
        assert result.score == 130.0
        assert result.level == "CRITICAL"

    def test_all_violations_in_zone(self):
        """Worst case: all violations + zone + full confidence."""
        result = self.engine.score(
            ["NO_HELMET", "NO_VEST", "NO_HARNESS", "NO_GLOVES"],
            ["Machinery Zone"], 1.0
        )
        # (80 + 50 + 90 + 30) * 1.5 * 1.0 = 375
        assert result.score == 375.0
        assert result.level == "CRITICAL"

    # ── Color and label ───────────────────────────────────

    def test_severity_colors_are_bgr(self):
        """All severity colors should be 3-element BGR tuples."""
        for violations in [["NO_HELMET"], ["NO_GLOVES"], []]:
            result = self.engine.score(violations, [], 1.0)
            assert len(result.color) == 3
            assert all(0 <= c <= 255 for c in result.color)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
