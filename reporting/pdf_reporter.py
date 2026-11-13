"""
reporting/pdf_reporter.py — Forensic PDF Incident Report Generator
====================================================================
Generates multi-page PDF reports with cover page, violation tables,
snapshot evidence, and auto-generated recommendations.

Uses fpdf2 (NOT reportlab, NOT fpdf) for PDF generation.

Report structure:
  Page 1    — Cover with title, timestamp, and summary statistics
  Page 2+   — Violation incident table (color-coded rows by severity)
  Next pages — Snapshot evidence (2 snapshots per page with captions)
  Last page  — Auto-generated recommendations based on violation patterns
"""

from pathlib import Path
from datetime import datetime
from typing import List

from fpdf import FPDF


class PDFReporter:
    """Generates forensic PDF incident reports."""

    # Severity level → row background RGB color (light tints)
    ROW_COLORS = {
        "CRITICAL": (255, 200, 200),
        "HIGH":     (255, 225, 200),
        "MEDIUM":   (255, 245, 200),
        "LOW":      (220, 255, 220),
    }

    def __init__(self, config: dict):
        reporting_cfg = config.get("reporting", {})
        output_dir = Path(reporting_cfg.get("output_dir", "output"))

        self.reports_dir = output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, incidents: list, output_path: str = "") -> str:
        """
        Generate a complete forensic PDF report.

        Args:
            incidents: List of Incident objects from ViolationLogger.
            output_path: Optional explicit output path. If empty,
                         auto-generates a timestamped filename.

        Returns:
            Path to the saved PDF file.
        """
        if not output_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                self.reports_dir / f"StructureGuard_Report_{ts}.pdf"
            )

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- PAGE 1: Cover ---
        self._add_cover_page(pdf, incidents)

        # --- PAGE 2+: Violation Table ---
        self._add_violation_table(pdf, incidents)

        # --- SNAPSHOT PAGES: Evidence with captions ---
        self._add_snapshot_pages(pdf, incidents)

        # --- LAST PAGE: Recommendations ---
        self._add_recommendations_page(pdf, incidents)

        # Save the PDF
        pdf.output(output_path)
        return output_path

    def _add_cover_page(self, pdf: FPDF, incidents: list) -> None:
        """Cover page with logo placeholder, title, and summary box."""
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 28)

        # --- Hard-hat icon placeholder (drawn with lines/shapes) ---
        # Draw a simple hard hat shape at the top center
        cx, cy = 105, 45
        # Hat dome (arc approximated by filled rectangle + top rect)
        pdf.set_fill_color(255, 165, 0)
        pdf.rect(cx - 20, cy - 8, 40, 12, "F")  # brim
        pdf.set_fill_color(255, 200, 50)
        pdf.rect(cx - 15, cy - 20, 30, 14, "F")  # dome

        # Title
        pdf.ln(50)
        pdf.set_font("Helvetica", "B", 26)
        pdf.set_text_color(30, 30, 80)
        pdf.cell(0, 12, "StructureGuard", ln=True, align="C")

        pdf.set_font("Helvetica", "", 14)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 8, "Incident Report", ln=True, align="C")

        pdf.ln(5)
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 6, "Construction Site Safety Analysis", ln=True, align="C")

        # Report date & time
        pdf.ln(10)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(50, 50, 50)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 6, f"Report Generated: {now}", ln=True, align="C")

        # --- Summary box ---
        pdf.ln(15)
        summary = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for inc in incidents:
            level = inc.severity_level
            summary[level] = summary.get(level, 0) + 1

        total = len(incidents)

        # Draw summary table
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(240, 240, 250)

        col_w = 38
        x_start = (210 - col_w * 5) / 2
        pdf.set_x(x_start)

        headers = ["Total", "Critical", "High", "Medium", "Low"]
        values = [str(total), str(summary["CRITICAL"]), str(summary["HIGH"]),
                  str(summary["MEDIUM"]), str(summary["LOW"])]
        colors = [
            (200, 200, 220),
            (255, 200, 200),
            (255, 225, 200),
            (255, 245, 200),
            (220, 255, 220),
        ]

        # Header row
        for i, header in enumerate(headers):
            pdf.set_fill_color(*colors[i])
            pdf.cell(col_w, 10, header, border=1, fill=True, align="C")
        pdf.ln()

        # Values row
        pdf.set_x(x_start)
        pdf.set_font("Helvetica", "B", 16)
        for i, val in enumerate(values):
            pdf.set_fill_color(*colors[i])
            pdf.cell(col_w, 12, val, border=1, fill=True, align="C")
        pdf.ln()

    def _add_violation_table(self, pdf: FPDF, incidents: list) -> None:
        """Detailed violation table with color-coded severity rows."""
        if not incidents:
            return

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(30, 30, 80)
        pdf.cell(0, 10, "Violation Log", ln=True, align="C")
        pdf.ln(5)

        # Table header
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(50, 50, 80)
        pdf.set_text_color(255, 255, 255)

        col_widths = [10, 20, 45, 25, 40, 35, 15]
        headers = ["#", "Track", "Violation", "Severity", "Zone", "Timestamp", "Frm"]

        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, border=1, fill=True, align="C")
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(30, 30, 30)

        for idx, inc in enumerate(incidents, 1):
            # Color-code row by severity
            row_color = self.ROW_COLORS.get(inc.severity_level, (255, 255, 255))
            pdf.set_fill_color(*row_color)

            violations_str = ", ".join(inc.violations)[:40]
            zones_str = ", ".join(inc.in_zones)[:35] if inc.in_zones else "Safe area"

            row_data = [
                str(idx),
                str(inc.track_id),
                violations_str,
                inc.severity_level,
                zones_str,
                inc.timestamp,
                str(inc.frame_number),
            ]

            for i, cell_text in enumerate(row_data):
                pdf.cell(col_widths[i], 7, cell_text, border=1,
                         fill=True, align="C")
            pdf.ln()

            # Page break if near bottom
            if pdf.get_y() > 270:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_fill_color(50, 50, 80)
                pdf.set_text_color(255, 255, 255)
                for i, header in enumerate(headers):
                    pdf.cell(col_widths[i], 8, header, border=1,
                             fill=True, align="C")
                pdf.ln()
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(30, 30, 30)

    def _add_snapshot_pages(self, pdf: FPDF, incidents: list) -> None:
        """Embed up to 2 snapshot images per page with captions."""
        # Filter to incidents that have a valid snapshot file
        snapshot_incidents = [
            inc for inc in incidents
            if inc.snapshot_path and Path(inc.snapshot_path).exists()
        ]

        if not snapshot_incidents:
            return

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(30, 30, 80)
        pdf.cell(0, 10, "Snapshot Evidence", ln=True, align="C")
        pdf.ln(5)

        count_on_page = 0

        for inc in snapshot_incidents:
            if count_on_page >= 2:
                pdf.add_page()
                count_on_page = 0

            # Caption above the image
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(50, 50, 50)
            violations_text = ", ".join(inc.violations)
            caption = (f"Track ID: {inc.track_id} | "
                       f"{violations_text} | "
                       f"{inc.severity_level} | "
                       f"{inc.timestamp}")
            pdf.cell(0, 7, caption, ln=True)

            # Embed the snapshot image
            try:
                # Fit to page width with border
                img_width = 170
                pdf.set_draw_color(100, 100, 100)
                pdf.rect(pdf.get_x() + 9, pdf.get_y(), img_width + 2, 90)
                pdf.image(inc.snapshot_path, x=20, w=img_width, h=88)
                pdf.ln(5)
            except Exception:
                # If image can't be loaded, show placeholder text
                pdf.set_font("Helvetica", "I", 9)
                pdf.cell(0, 10, f"[Snapshot unavailable: {inc.snapshot_path}]",
                         ln=True)
                pdf.ln(5)

            count_on_page += 1

    def _add_recommendations_page(self, pdf: FPDF, incidents: list) -> None:
        """Auto-generate recommendations based on violation patterns."""
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(30, 30, 80)
        pdf.cell(0, 10, "Recommendations", ln=True, align="C")
        pdf.ln(10)

        # Count violations by type and severity
        summary = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        violation_counts = {}
        for inc in incidents:
            summary[inc.severity_level] = summary.get(inc.severity_level, 0) + 1
            for v in inc.violations:
                violation_counts[v] = violation_counts.get(v, 0) + 1

        # Build recommendations based on observed patterns
        recommendations = []

        if summary.get("CRITICAL", 0) > 5:
            recommendations.append(
                "URGENT: Immediate site supervisor alert required. "
                "Critical violation threshold exceeded. Consider halting "
                "work in affected zones until compliance is restored."
            )

        if summary.get("CRITICAL", 0) > 0:
            recommendations.append(
                "Deploy additional safety officers to zones with "
                "critical violations during peak work hours."
            )

        if violation_counts.get("NO_HELMET", 0) > 3:
            recommendations.append(
                "Conduct mandatory helmet compliance briefing for all "
                "site workers. Enforce helmet-on policy at site entry gates."
            )

        if violation_counts.get("NO_VEST", 0) > 3:
            recommendations.append(
                "Distribute high-visibility vests at entry checkpoints. "
                "Consider color-coded vests by crew/trade for easier tracking."
            )

        if violation_counts.get("NO_HARNESS", 0) > 2:
            recommendations.append(
                "Review fall protection protocols. Ensure harness "
                "anchor points are accessible and properly rated at all "
                "elevated work areas."
            )

        if violation_counts.get("NO_GLOVES", 0) > 3:
            recommendations.append(
                "Provide task-appropriate gloves at material handling "
                "stations. Brief crews on hand injury statistics."
            )

        zone_incidents = [inc for inc in incidents if inc.in_zones]
        if len(zone_incidents) > 5:
            recommendations.append(
                "Increase physical barriers around marked danger zones. "
                "Consider audible alarms triggered by zone entry for "
                "high-risk machinery areas."
            )

        if not recommendations:
            recommendations.append(
                "No critical patterns detected. Continue routine safety "
                "monitoring and maintain current PPE compliance protocols."
            )

        # Always add general best practices
        recommendations.append(
            "Schedule periodic StructureGuard report reviews with the "
            "site safety committee. Use trend data to identify recurring "
            "non-compliance windows (shift changes, break periods)."
        )

        # Render recommendations
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(50, 50, 50)

        for i, rec in enumerate(recommendations, 1):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(8, 7, f"{i}.", ln=False)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(170, 6, rec)
            pdf.ln(4)

        # Footer disclaimer
        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.multi_cell(0, 5,
            "This report was auto-generated by StructureGuard. "
            "Recommendations are based on statistical patterns detected "
            "during the monitoring session and should be reviewed by a "
            "qualified safety officer before implementation."
        )
