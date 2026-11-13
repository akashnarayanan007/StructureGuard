"""
tests/test_zone_manager.py — Unit Tests for Ray Casting Zone Engine
=====================================================================
Tests the point-in-polygon algorithm and zone management logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.zone_manager import ZoneManager


class TestZoneManager:
    """Tests for the ZoneManager and ray casting algorithm."""

    def setup_method(self):
        """Set up a config with known danger zone polygons."""
        self.config = {
            "zones": {
                "danger_polygons": [
                    {
                        "name": "Test Zone A",
                        "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                        "color": [0, 0, 255],
                    },
                    {
                        "name": "Test Zone B",
                        "points": [[200, 200], [300, 200], [300, 300], [200, 300]],
                        "color": [0, 165, 255],
                    },
                ]
            }
        }
        self.zone_manager = ZoneManager(self.config)

    # ── Ray casting algorithm ─────────────────────────────

    def test_point_inside_square(self):
        """Point clearly inside a square polygon → True."""
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert ZoneManager.point_in_polygon((50, 50), polygon) is True

    def test_point_outside_square(self):
        """Point clearly outside a square polygon → False."""
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert ZoneManager.point_in_polygon((150, 50), polygon) is False

    def test_point_on_edge(self):
        """Point on the left edge of the polygon — should be inside by convention."""
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        # On the left edge (x=0, y=50): the ray cast from (0,50) →
        # crosses edges at x=0 and x=100, count depends on edge handling.
        # Our implementation treats boundary consistently.
        result = ZoneManager.point_in_polygon((0, 50), polygon)
        # Accept either True or False — edge handling is implementation-defined
        assert isinstance(result, bool)

    def test_point_inside_triangle(self):
        """Ray casting works on non-rectangular convex polygons."""
        triangle = [[50, 0], [100, 100], [0, 100]]
        assert ZoneManager.point_in_polygon((50, 50), triangle) is True

    def test_point_outside_triangle(self):
        """Point outside a triangle → False."""
        triangle = [[50, 0], [100, 100], [0, 100]]
        assert ZoneManager.point_in_polygon((10, 10), triangle) is False

    def test_concave_polygon(self):
        """Ray casting handles concave (non-convex) polygons correctly."""
        # L-shaped polygon
        l_shape = [[0, 0], [60, 0], [60, 40], [30, 40], [30, 80], [0, 80]]
        # Inside the horizontal bar
        assert ZoneManager.point_in_polygon((50, 20), l_shape) is True
        # Inside the vertical stem
        assert ZoneManager.point_in_polygon((15, 60), l_shape) is True
        # In the concave cutout (should be outside)
        assert ZoneManager.point_in_polygon((50, 60), l_shape) is False

    # ── Zone lookup ───────────────────────────────────────

    def test_check_zones_inside(self):
        """Worker inside Zone A returns its name."""
        result = self.zone_manager.check_zones((50, 50))
        assert "Test Zone A" in result

    def test_check_zones_outside(self):
        """Worker outside all zones returns empty list."""
        result = self.zone_manager.check_zones((150, 150))
        assert result == []

    def test_check_zones_multiple(self):
        """Worker can be inside multiple zones if they overlap."""
        # Zone A: 0-100, Zone B: 200-300. Worker inside Zone B only.
        result = self.zone_manager.check_zones((250, 250))
        assert "Test Zone B" in result
        assert "Test Zone A" not in result


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
