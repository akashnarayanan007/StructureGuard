"""
reporting/dashboard.py — Real-Time OpenCV Overlay Dashboard
=============================================================
Renders a side-by-side display: the annotated video frame on
the left, and a dark-themed statistics panel on the right.

The stats panel provides at-a-glance situational awareness:
  • Live worker count and FPS
  • Severity breakdown with color-coded counters
  • Scrolling list of the 5 most recent alerts

All text overlays use a filled background rectangle for
readability against busy construction site footage.
"""

from typing import List, Dict, Tuple
from datetime import datetime

import cv2
import numpy as np


class DashboardRenderer:
    """Renders annotated frames with a real-time statistics panel."""

    # Dark theme background for the stats panel
    PANEL_BG = (30, 26, 26)       # #1a1a1e in BGR
    HEADER_COLOR = (255, 200, 50) # Bright cyan-ish title
    TEXT_COLOR = (220, 220, 220)  # Light gray body text
    DIVIDER_COLOR = (80, 80, 80)  # Subtle divider lines

    # Severity level → display color (BGR)
    SEVERITY_COLORS = {
        "CRITICAL": (0, 0, 220),
        "HIGH":     (0, 100, 255),
        "MEDIUM":   (0, 165, 255),
        "LOW":      (0, 220, 220),
    }

    # Severity level → emoji prefix for display
    SEVERITY_ICONS = {
        "CRITICAL": "[!]",
        "HIGH":     "[H]",
        "MEDIUM":   "[M]",
        "LOW":      "[L]",
    }

    def __init__(self, config: dict):
        dash_cfg = config.get("dashboard", {})
        self.panel_width = dash_cfg.get("panel_width", 300)
        self.show_fps = dash_cfg.get("show_fps", True)
        self.show_violation_count = dash_cfg.get("show_violation_count", True)
        self.show_track_ids = dash_cfg.get("show_track_ids", True)

    def render(self, frame: np.ndarray, tracks: list,
               violations_map: dict, incidents_summary: dict,
               recent_incidents: list, fps: float,
               frame_number: int, zone_manager=None) -> np.ndarray:
        """
        Compose the full dashboard frame.

        Args:
            frame: BGR video frame to annotate.
            tracks: List of Track objects with bounding boxes.
            violations_map: dict mapping track_id → (ComplianceStatus, SeverityResult).
            incidents_summary: dict of severity level → count.
            recent_incidents: List of recent Incident objects for the alert feed.
            fps: Current processing speed.
            frame_number: Sequential frame counter.
            zone_manager: Optional ZoneManager to draw danger zones.

        Returns:
            Combined frame (annotated video + stats panel).
        """
        annotated = frame.copy()

        # --- Draw danger zones if available ---
        if zone_manager is not None:
            annotated = zone_manager.draw_zones(annotated)

        # --- Draw bounding boxes and labels on each tracked worker ---
        for track in tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = [int(c) for c in track.bbox]

            # Look up this worker's violation status
            if track_id in violations_map:
                compliance, severity = violations_map[track_id]
                color = severity.color
                violations_text = " | ".join(compliance.violations) if compliance.violations else "OK"
                label = f"ID:{track_id} | {violations_text} | {severity.level}"
            else:
                # No violation data yet — draw as neutral
                color = (0, 255, 0)   # Green = compliant
                label = f"ID:{track_id} | OK"

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label with background rectangle for readability
            if self.show_track_ids:
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated,
                              (x1, y1 - th - 10), (x1 + tw + 10, y1),
                              color, -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # --- FPS and frame counter overlay (top-left) ---
        if self.show_fps:
            self._draw_text_with_bg(annotated, f"FPS: {fps:.1f}",
                                     (10, 25), 0.6)
            self._draw_text_with_bg(annotated, f"Frame: {frame_number}",
                                     (10, 50), 0.5)

        # --- Build the right-side stats panel ---
        panel = self._build_stats_panel(
            frame_height=annotated.shape[0],
            total_workers=len(tracks),
            incidents_summary=incidents_summary,
            recent_incidents=recent_incidents,
            fps=fps,
            frame_number=frame_number,
        )

        # Combine: video on left, stats panel on right
        combined = np.hstack([annotated, panel])

        return combined

    def _build_stats_panel(self, frame_height: int, total_workers: int,
                           incidents_summary: dict, recent_incidents: list,
                           fps: float, frame_number: int) -> np.ndarray:
        """
        Build the dark-themed statistics panel.

        Layout:
          Header: StructureGuard
          ───────────────────────
          Live Stats
          ───────────────────────
          Violation Breakdown
          ───────────────────────
          Last 5 Alerts
          ───────────────────────
          Timestamp footer
        """
        panel = np.full((frame_height, self.panel_width, 3),
                        self.PANEL_BG, dtype=np.uint8)
        y = 10  # vertical cursor

        # --- Header ---
        y += 30
        cv2.putText(panel, "StructureGuard", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    self.HEADER_COLOR, 2, cv2.LINE_AA)
        y += 10
        cv2.line(panel, (10, y), (self.panel_width - 10, y),
                 self.DIVIDER_COLOR, 1)

        # --- Section 1: Live Stats ---
        y += 25
        cv2.putText(panel, "Live Stats", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    self.HEADER_COLOR, 1, cv2.LINE_AA)
        y += 5
        cv2.line(panel, (10, y), (self.panel_width - 10, y),
                 self.DIVIDER_COLOR, 1)

        total_violations = sum(incidents_summary.values())
        stats = [
            f"Total Workers:      {total_workers}",
            f"Active Violations:  {total_violations}",
            f"Frames Processed:   {frame_number}",
            f"FPS:                {fps:.1f}",
        ]
        for line in stats:
            y += 22
            cv2.putText(panel, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        self.TEXT_COLOR, 1, cv2.LINE_AA)

        # --- Section 2: Violation Breakdown ---
        y += 20
        cv2.putText(panel, "Violation Breakdown", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    self.HEADER_COLOR, 1, cv2.LINE_AA)
        y += 5
        cv2.line(panel, (10, y), (self.panel_width - 10, y),
                 self.DIVIDER_COLOR, 1)

        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            y += 22
            count = incidents_summary.get(level, 0)
            icon = self.SEVERITY_ICONS[level]
            color = self.SEVERITY_COLORS[level]
            text = f"{icon} {level:10s}: {count}"
            cv2.putText(panel, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        color, 1, cv2.LINE_AA)

        # --- Section 3: Last 5 Alerts ---
        y += 20
        cv2.putText(panel, "Recent Alerts", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    self.HEADER_COLOR, 1, cv2.LINE_AA)
        y += 5
        cv2.line(panel, (10, y), (self.panel_width - 10, y),
                 self.DIVIDER_COLOR, 1)

        if recent_incidents:
            for inc in recent_incidents[-5:]:
                y += 20
                # Truncate timestamp to HH:MM:SS
                ts_short = inc.timestamp.split(" ")[-1] if " " in inc.timestamp else inc.timestamp
                viol_text = ",".join(inc.violations)[:18]
                alert_text = f"ID:{inc.track_id} {viol_text} {ts_short}"
                level_color = self.SEVERITY_COLORS.get(
                    inc.severity_level, self.TEXT_COLOR)
                cv2.putText(panel, alert_text, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            level_color, 1, cv2.LINE_AA)
        else:
            y += 20
            cv2.putText(panel, "No incidents yet", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        self.TEXT_COLOR, 1, cv2.LINE_AA)

        # --- Footer: Current timestamp ---
        footer_y = frame_height - 15
        cv2.line(panel, (10, footer_y - 15), (self.panel_width - 10, footer_y - 15),
                 self.DIVIDER_COLOR, 1)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(panel, now, (15, footer_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    self.TEXT_COLOR, 1, cv2.LINE_AA)

        return panel

    @staticmethod
    def _draw_text_with_bg(frame: np.ndarray, text: str,
                            position: Tuple[int, int],
                            scale: float = 0.5) -> None:
        """Draw text with a filled black background rectangle."""
        x, y = position
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(frame, (x - 2, y - th - 4),
                      (x + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 255, 0), 1, cv2.LINE_AA)
