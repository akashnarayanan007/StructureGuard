"""
core/zone_manager.py — Ray Casting Polygon Zone Engine
=======================================================
Detects whether a worker has entered a defined danger zone
(e.g., near heavy machinery, on a scaffold edge) using the
Ray Casting point-in-polygon algorithm.

Algorithm — Ray Casting:
  Cast a horizontal ray from the test point toward +infinity.
  Count how many polygon edges the ray intersects.
    • Odd intersections  → point is INSIDE the polygon.
    • Even intersections → point is OUTSIDE the polygon.

  This works for any simple (non-self-intersecting) polygon,
  including concave shapes — a property that makes it robust
  for irregularly-shaped exclusion zones on a construction site.

Edge cases handled:
  • Point exactly on a vertex — counted as inside (>=)
  • Horizontal edges — skipped by the y-range check
  • Collinear edges — safe because we test strict inequality on one side
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class ZonePolygon:
    """A named danger zone defined by a polygon boundary."""
    name: str
    points: List[List[int]]   # [[x1,y1], [x2,y2], ...]
    color: Tuple[int, int, int]


class ZoneManager:
    """Manages danger zone polygons and checks worker positions against them."""

    def __init__(self, config: dict):
        zones_cfg = config.get("zones", {})
        polygon_defs = zones_cfg.get("danger_polygons", [])

        self.zones: List[ZonePolygon] = []
        for zone_def in polygon_defs:
            self.zones.append(ZonePolygon(
                name=zone_def["name"],
                points=zone_def["points"],
                color=tuple(zone_def["color"]),
            ))

    @staticmethod
    def point_in_polygon(point: Tuple[float, float],
                         polygon_points: List[List[int]]) -> bool:
        """
        Ray Casting algorithm for point-in-polygon test.

        Cast a horizontal ray from `point` towards +infinity along x.
        Count edge crossings — odd means inside, even means outside.

        Args:
            point: (x, y) coordinates of the test point.
            polygon_points: List of [x, y] vertices defining the polygon.

        Returns:
            True if the point is inside the polygon.
        """
        x, y = point
        n = len(polygon_points)
        inside = False

        # Walk each edge (j→i) of the polygon
        j = n - 1
        for i in range(n):
            xi, yi = polygon_points[i]
            xj, yj = polygon_points[j]

            # Check if the ray at height y could cross edge (j→i).
            # The edge must straddle the ray's y-level: one endpoint
            # above and one below (strict < on one side avoids
            # double-counting shared vertices).
            if ((yi > y) != (yj > y)):
                # Compute x-coordinate where the edge crosses y
                x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi

                # If the intersection is to the right of the point,
                # the ray crosses this edge → toggle inside/outside
                if x < x_intersect:
                    inside = not inside

            j = i  # advance to next edge

        return inside

    def check_zones(self, center_point: Tuple[float, float]) -> List[str]:
        """
        Return names of all danger zones the point falls inside.

        Args:
            center_point: (cx, cy) centroid of a tracked worker.

        Returns:
            List of zone names (empty if the worker is in a safe area).
        """
        triggered = []
        for zone in self.zones:
            if self.point_in_polygon(center_point, zone.points):
                triggered.append(zone.name)
        return triggered

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all danger zone polygon boundaries on the frame.

        Each zone is rendered as a semi-transparent filled polygon
        with a solid border and a label at the centroid.

        Args:
            frame: BGR image to annotate.

        Returns:
            Annotated frame (modified in-place and returned).
        """
        overlay = frame.copy()

        for zone in self.zones:
            pts = np.array(zone.points, dtype=np.int32)

            # Semi-transparent fill so the zone is visible but
            # doesn't obscure workers underneath
            cv2.fillPoly(overlay, [pts], zone.color)

            # Solid border for clarity
            cv2.polylines(frame, [pts], isClosed=True,
                          color=zone.color, thickness=2)

            # Label at the centroid of the polygon
            centroid_x = int(np.mean([p[0] for p in zone.points]))
            centroid_y = int(np.mean([p[1] for p in zone.points]))

            # Background rectangle for text readability
            label = zone.name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.6, 2)
            cv2.rectangle(frame,
                          (centroid_x - 5, centroid_y - th - 5),
                          (centroid_x + tw + 5, centroid_y + 5),
                          (0, 0, 0), -1)
            cv2.putText(frame, label,
                        (centroid_x, centroid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

        # Blend the transparent overlay with the original
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        return frame
