"""
core/severity_engine.py — Weighted Severity Scoring
=====================================================
Computes a numerical severity score for each violation event
and maps it to a categorical level (CRITICAL / HIGH / MEDIUM / LOW).

Scoring formula:
  base_score = sum of base weights for each active violation
    (e.g., no_helmet=80, no_harness=90, no_vest=50, no_gloves=30)

  zone_score = base_score × zone_multiplier   if worker is in a danger zone
               base_score                     otherwise

  final_score = zone_score × avg_confidence   if confidence_weight is enabled
                zone_score                    otherwise

Why multiplicative zone penalty?
  A worker without a helmet in an open area is bad (score 80).
  The same worker without a helmet near heavy machinery is worse
  (80 × 1.5 = 120 → CRITICAL). The multiplier captures that
  context amplifies risk, matching real safety assessment logic.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SeverityResult:
    """Result of severity computation for a violation event."""
    score: float
    level: str                      # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    color: Tuple[int, int, int]     # BGR color for display
    label: str                      # Human-readable label with icon


class SeverityEngine:
    """Computes risk severity from violations, zone presence, and confidence."""

    def __init__(self, config: dict):
        sev_cfg = config["severity"]

        # Base scores per violation type — higher = more dangerous
        self.base_scores = {
            "NO_HELMET":  sev_cfg["no_helmet_base"],
            "NO_VEST":    sev_cfg["no_vest_base"],
            "NO_HARNESS": sev_cfg["no_harness_base"],
            "NO_GLOVES":  sev_cfg["no_gloves_base"],
        }

        # Zone presence multiplier — amplifies score inside danger zones
        self.zone_multiplier = sev_cfg["zone_multiplier"]

        # Whether to weight the final score by detection confidence
        self.confidence_weight = sev_cfg["confidence_weight"]

        # Threshold boundaries for categorical levels
        self.thresholds = sev_cfg["thresholds"]

    def score(self, violations: List[str], in_zones: List[str],
              confidence: float = 1.0) -> SeverityResult:
        """
        Compute severity score and classify into a level.

        Args:
            violations: List of violation strings (e.g., ["NO_HELMET", "NO_VEST"]).
            in_zones: List of danger zone names the worker is inside (empty = safe area).
            confidence: Average detection confidence for this worker (0.0–1.0).

        Returns:
            SeverityResult with numeric score, level, display color, and label.
        """
        if not violations:
            # No violations — return a clean LOW result
            return SeverityResult(
                score=0.0, level="LOW",
                color=(0, 220, 220), label="COMPLIANT",
            )

        # Step 1: Sum base scores for each active violation
        base_score = sum(
            self.base_scores.get(v, 0) for v in violations
        )

        # Step 2: Apply zone multiplier if worker is in any danger zone
        if in_zones:
            zone_score = base_score * self.zone_multiplier
        else:
            zone_score = base_score

        # Step 3: Optionally weight by detection confidence
        # Higher confidence → higher score (we're more certain of the violation)
        if self.confidence_weight and confidence > 0:
            final_score = zone_score * confidence
        else:
            final_score = zone_score

        # Step 4: Map numerical score to categorical severity level
        level, color, label = self._classify(final_score)

        return SeverityResult(
            score=round(final_score, 2),
            level=level,
            color=color,
            label=label,
        )

    def _classify(self, score: float) -> Tuple[str, Tuple[int, int, int], str]:
        """
        Map a numerical severity score to a level, color, and label.

        Thresholds are loaded from config — no hardcoded cutoffs.
        Colors are BGR for OpenCV rendering.
        """
        if score >= self.thresholds["critical"]:
            return "CRITICAL", (0, 0, 220), "CRITICAL"
        elif score >= self.thresholds["high"]:
            return "HIGH", (0, 100, 255), "HIGH"
        elif score >= self.thresholds["medium"]:
            return "MEDIUM", (0, 165, 255), "MEDIUM"
        else:
            return "LOW", (0, 220, 220), "LOW"
