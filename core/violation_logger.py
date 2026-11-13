"""
core/violation_logger.py — In-Memory Incident Log
===================================================
Stores every confirmed violation event with metadata,
saves frame snapshots as forensic evidence, and provides
summary statistics for the dashboard and PDF reports.

All incidents are kept in memory (list of Incident dataclass
objects) for the duration of the session. Snapshots are
persisted to disk as JPEG files named by track ID and timestamp.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


@dataclass
class Incident:
    """A single logged safety violation event."""
    track_id: int
    violations: List[str]
    severity_level: str
    severity_score: float
    in_zones: List[str]
    timestamp: str               # ISO format: YYYY-MM-DD HH:MM:SS
    snapshot_path: str
    frame_number: int


class ViolationLogger:
    """Logs violation incidents and saves frame snapshots."""

    def __init__(self, config: dict):
        reporting_cfg = config.get("reporting", {})
        output_dir = Path(reporting_cfg.get("output_dir", "output"))

        self.snapshots_dir = output_dir / "snapshots"
        # Auto-create output directories if they don't exist
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_on_violation = reporting_cfg.get("snapshot_on_violation", True)

        # In-memory incident store
        self.incidents: List[Incident] = []

    def log(self, track_id: int, violations: List[str],
            severity_level: str, severity_score: float,
            in_zones: List[str], frame: np.ndarray,
            frame_number: int, timestamp: str = "") -> None:
        """
        Record a violation incident and save a frame snapshot.

        Args:
            track_id: Persistent worker identity from DeepSORT.
            violations: List of violation codes (e.g., ["NO_HELMET"]).
            severity_level: Categorical level ("CRITICAL", "HIGH", etc.).
            severity_score: Numerical severity score.
            in_zones: Danger zones the worker is inside.
            frame: The current BGR frame for snapshot capture.
            frame_number: Sequential frame number in the video.
            timestamp: ISO-format timestamp (auto-generated if empty).
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save frame snapshot as forensic evidence
        snapshot_path = ""
        if self.snapshot_on_violation and frame is not None:
            filename = f"track{track_id}_{frame_number}_{datetime.now().strftime('%H%M%S')}.jpg"
            snapshot_path = str(self.snapshots_dir / filename)
            cv2.imwrite(snapshot_path, frame)

        incident = Incident(
            track_id=track_id,
            violations=violations,
            severity_level=severity_level,
            severity_score=severity_score,
            in_zones=in_zones,
            timestamp=timestamp,
            snapshot_path=snapshot_path,
            frame_number=frame_number,
        )

        self.incidents.append(incident)

    def get_all(self) -> List[Incident]:
        """Return all logged incidents."""
        return self.incidents

    def get_summary(self) -> Dict[str, int]:
        """
        Count incidents by severity level.

        Returns:
            dict like {"CRITICAL": 8, "HIGH": 9, "MEDIUM": 4, "LOW": 2}
        """
        summary = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for inc in self.incidents:
            level = inc.severity_level
            if level in summary:
                summary[level] += 1
            else:
                summary[level] = 1
        return summary

    def get_recent(self, n: int = 5) -> List[Incident]:
        """Return the N most recent incidents for dashboard display."""
        return self.incidents[-n:] if self.incidents else []

    def clear(self) -> None:
        """Clear all in-memory incidents."""
        self.incidents.clear()
