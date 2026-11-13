"""
main.py — StructureGuard Entry Point
======================================
CLI-based launcher for the real-time construction site safety
violation detection pipeline.

Pipeline per frame:
  1. Read frame from video source
  2. Enhance via CLAHE on luminance channel
  3. Detect persons and PPE via YOLOv8
  4. Track persons persistently via DeepSORT
  5. Associate PPE to workers via body-zone IoU
  6. Check danger zone entry via ray casting
  7. Score severity and debounce (15-frame per-track)
  8. Log confirmed violations with snapshots
  9. Render dashboard overlay
 10. Periodically generate PDF reports

Usage:
  python main.py --source sample_videos/site.mp4 --save-video
  python main.py --source 0                   # webcam
  python main.py --source site.mp4 --no-display --report-now
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import yaml
import numpy as np

from core.preprocessor import FramePreprocessor
from core.detector import SafetyDetector
from core.tracker import WorkerTracker
from core.associator import PPEAssociator
from core.zone_manager import ZoneManager
from core.severity_engine import SeverityEngine
from core.violation_logger import ViolationLogger
from reporting.dashboard import DashboardRenderer
from reporting.pdf_reporter import PDFReporter


def load_config(config_path: str) -> dict:
    """Load and return the YAML configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def ensure_output_dirs(config: dict) -> None:
    """Create all output directories if they don't exist."""
    output_dir = Path(config.get("reporting", {}).get("output_dir", "output"))
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (output_dir / "annotated_videos").mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="StructureGuard — Real-Time Construction Safety Violation Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --source sample_videos/site.mp4 --save-video
  python main.py --source 0                  (webcam)
  python main.py --source site.mp4 --no-display
  python main.py --source site.mp4 --report-now
        """,
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video file path or 0 for webcam (default: 0)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headless — no OpenCV window",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Save annotated output video to output/annotated_videos/",
    )
    parser.add_argument(
        "--report-now", action="store_true",
        help="Generate PDF report immediately on exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load configuration ────────────────────────────────
    config = load_config(args.config)
    ensure_output_dirs(config)

    # ── Initialize all pipeline modules ───────────────────
    print("[INFO] Initializing StructureGuard modules...")
    preprocessor = FramePreprocessor(config)
    detector     = SafetyDetector(config)
    tracker      = WorkerTracker(config)
    associator   = PPEAssociator()
    zone_manager = ZoneManager(config)
    severity_eng = SeverityEngine(config)
    logger       = ViolationLogger(config)
    dashboard    = DashboardRenderer(config)
    reporter     = PDFReporter(config)

    # ── Open video source ─────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        print("  If using a file, check the path exists.")
        print("  If using webcam, try --source 0")
        sys.exit(1)

    # Video properties for the writer
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video source opened: {orig_w}x{orig_h} @ {orig_fps:.1f} FPS")

    # ── Video writer (optional) ───────────────────────────
    video_writer = None
    if args.save_video:
        output_dir = Path(config["reporting"]["output_dir"])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_out_path = str(
            output_dir / "annotated_videos" / f"annotated_{ts}.mp4"
        )
        # Dashboard adds a panel on the right, so output width increases
        panel_w = config.get("dashboard", {}).get("panel_width", 300)
        target_w, target_h = config["preprocessing"]["target_size"]
        out_w = target_w + panel_w
        out_h = target_h

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            video_out_path, fourcc, orig_fps, (out_w, out_h)
        )
        print(f"[INFO] Saving annotated video to: {video_out_path}")

    # ── Per-track debounce counters ───────────────────────
    # Key: track_id → value: consecutive frames with violations
    # A violation only triggers a log entry after debounce_frames
    # consecutive detections (avoids false positives from flicker)
    debounce_counters: dict = defaultdict(int)
    debounce_threshold = config["tracking"]["debounce_frames"]

    # Track which track_ids have already been logged this debounce cycle
    logged_tracks: set = set()

    # ── Reporting timer ───────────────────────────────────
    report_interval = config["reporting"]["report_interval_seconds"]
    last_report_time = time.time()

    # ── Main loop ─────────────────────────────────────────
    frame_number = 0
    total_workers_seen: set = set()
    fps = 0.0

    print("[INFO] Starting StructureGuard pipeline — press 'q' to quit")
    print("=" * 55)

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream")
                break

            frame_number += 1

            # Step 1: Enhance frame (CLAHE on luminance)
            enhanced = preprocessor.enhance(frame)

            # Step 2: Detect objects (persons + PPE)
            detections = detector.detect(enhanced)

            # Step 3: Track persons across frames (DeepSORT)
            tracks = tracker.update(detections, enhanced)

            # Step 4-7: Per-track compliance check
            violations_map = {}  # track_id → (ComplianceStatus, SeverityResult)

            for track in tracks:
                total_workers_seen.add(track.track_id)

                # Associate PPE items to this worker via body-zone IoU
                compliance = associator.associate(track.bbox, detections)

                # Check if worker is inside any danger zone
                in_zones = zone_manager.check_zones(track.center)

                # Compute average confidence for detections near this worker
                avg_conf = _avg_confidence_near(track.bbox, detections)

                # Score the severity of any violations
                sev_result = severity_eng.score(
                    compliance.violations, in_zones, avg_conf
                )

                violations_map[track.track_id] = (compliance, sev_result)

                # Temporal debounce: only log after N consecutive frames
                if compliance.violations:
                    debounce_counters[track.track_id] += 1

                    if (debounce_counters[track.track_id] >= debounce_threshold
                            and track.track_id not in logged_tracks):
                        # Violation persisted past debounce — log it
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        logger.log(
                            track_id=track.track_id,
                            violations=compliance.violations,
                            severity_level=sev_result.level,
                            severity_score=sev_result.score,
                            in_zones=in_zones,
                            frame=enhanced,
                            frame_number=frame_number,
                            timestamp=timestamp,
                        )
                        logged_tracks.add(track.track_id)

                        # Print alert to console
                        viol_str = ", ".join(compliance.violations)
                        zone_str = ", ".join(in_zones) if in_zones else "Safe area"
                        print(
                            f"  [ALERT] Track {str(track.track_id):>3s} | "
                            f"{viol_str:30s} | {sev_result.level:8s} | "
                            f"Zone: {zone_str}"
                        )
                else:
                    # No violations this frame — reset debounce for this track
                    debounce_counters[track.track_id] = 0
                    logged_tracks.discard(track.track_id)

            # Step 8: Compute FPS
            elapsed = time.time() - loop_start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            # Step 9: Render dashboard
            summary = logger.get_summary()
            recent = logger.get_recent(5)
            output_frame = dashboard.render(
                frame=enhanced,
                tracks=tracks,
                violations_map=violations_map,
                incidents_summary=summary,
                recent_incidents=recent,
                fps=fps,
                frame_number=frame_number,
                zone_manager=zone_manager,
            )

            # Step 10: Display / write
            if not args.no_display:
                cv2.imshow("StructureGuard", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] User pressed 'q' — exiting")
                    break

            if video_writer is not None:
                video_writer.write(output_frame)

            # Step 11: Periodic report generation
            if time.time() - last_report_time >= report_interval:
                incidents = logger.get_all()
                if incidents:
                    report_path = reporter.generate(incidents)
                    print(f"  [REPORT] Periodic report saved: {report_path}")
                last_report_time = time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # ── Cleanup ───────────────────────────────────────
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        # ── Final report ──────────────────────────────────
        all_incidents = logger.get_all()
        report_path = ""
        if args.report_now or all_incidents:
            report_path = reporter.generate(all_incidents)

        # ── Console summary ───────────────────────────────
        summary = logger.get_summary()
        _print_summary(
            frame_number=frame_number,
            total_workers=len(total_workers_seen),
            total_incidents=len(all_incidents),
            summary=summary,
            report_path=report_path,
        )


def _avg_confidence_near(person_bbox: list, detections: list) -> float:
    """
    Compute the average detection confidence for boxes overlapping
    the person. Falls back to 1.0 if no relevant detections.
    """
    from core.associator import PPEAssociator
    assoc = PPEAssociator()

    confs = []
    for det in detections:
        iou = assoc.compute_iou(person_bbox, det.bbox)
        if iou > 0.05:
            confs.append(det.confidence)

    return sum(confs) / len(confs) if confs else 1.0


def _print_summary(frame_number: int, total_workers: int,
                   total_incidents: int, summary: dict,
                   report_path: str) -> None:
    """Print the formatted session summary to the console."""
    print()
    print("╔" + "═" * 42 + "╗")
    print("║   StructureGuard -- Session Summary      ║")
    print("╚" + "═" * 42 + "╝")
    print(f"  Total Frames Processed : {frame_number}")
    print(f"  Total Workers Tracked  : {total_workers}")
    print(f"  Total Incidents Logged : {total_incidents}")
    print("  " + "─" * 40)
    print(f"  CRITICAL : {summary.get('CRITICAL', 0)}")
    print(f"  HIGH     : {summary.get('HIGH', 0)}")
    print(f"  MEDIUM   : {summary.get('MEDIUM', 0)}")
    print(f"  LOW      : {summary.get('LOW', 0)}")
    print("  " + "─" * 40)
    if report_path:
        print(f"  Report saved to: {report_path}")
    else:
        print("  No report generated (no incidents)")
    print("═" * 44)


if __name__ == "__main__":
    main()
