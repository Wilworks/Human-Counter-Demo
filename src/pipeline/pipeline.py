"""
pipeline.py - Master Pipeline orchestrator for 3D Physics Meter & Angle Human Counter.
"""

import os
import sys
import time
import cv2
import yaml
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich import box

# Ensure project root in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.detection.detector import HumanDetector
from src.tracking.tracker import HumanTracker
from src.counting.counter import HumanCounterPhysics
from src.utils.visualization import (
    draw_physics_virtual_gate,
    draw_physics_pointers_and_telemetry,
    draw_hud_physics,
    draw_new_count_flash_physics,
)
from src.utils.logger import CountLogger

console = Console()


class HumanCountingPipeline:
    """End-to-end 3D Physics processing pipeline with fullscreen display and laser pointer rays."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize pipeline with YAML settings."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        det_cfg = self.config.get("detection", {})
        trk_cfg = self.config.get("tracking", {})
        phys_cfg = self.config.get("physics_gate", {})
        out_cfg = self.config.get("output", {})
        disp_cfg = self.config.get("display", {})

        self.fullscreen = disp_cfg.get("fullscreen", True)

        # Component instances
        self.detector = HumanDetector(
            weights_path=det_cfg.get("model_weights", "yolov8n.pt"),
            target_class_id=det_cfg.get("target_class_id", 0),
            conf_thresh=det_cfg.get("conf_thresh", 0.50),
            iou_thresh=det_cfg.get("iou_thresh", 0.45),
            imgsz=det_cfg.get("imgsz", 640),
            device=det_cfg.get("device", "0"),
        )

        self.tracker = HumanTracker(
            algorithm=trk_cfg.get("algorithm", "bytetrack"),
            track_activation_threshold=trk_cfg.get("track_high_thresh", 0.50),
            lost_track_buffer=trk_cfg.get("track_buffer", 60),
            minimum_matching_threshold=trk_cfg.get("match_thresh", 0.80),
            frame_rate=self.config.get("source", {}).get("fps", 30),
        )

        self.counter = HumanCounterPhysics(
            mode=phys_cfg.get("mode", "frontal_depth"),
            direction=phys_cfg.get("direction", "towards_camera"),
            target_depth_meters=phys_cfg.get("target_depth_meters", 3.5),
            focal_length_px=phys_cfg.get("focal_length_px", 800.0),
            human_height_meters=phys_cfg.get("human_height_meters", 1.75),
            wall_x_ratio=phys_cfg.get("wall_x_ratio", 0.50),
            angular_threshold_deg=phys_cfg.get("angular_threshold_deg", 3.0),
            min_track_age=phys_cfg.get("min_track_age", 5),
            spatial_merge_distance=phys_cfg.get("spatial_merge_distance", 90.0),
        )

        self.display = out_cfg.get("display", True)
        self.save_video = out_cfg.get("save_video", False)
        self.save_path = out_cfg.get("save_path", "output/")

        self.logger = CountLogger(self.save_path) if out_cfg.get("log_counts", True) else None

    def run(self, source: str = None) -> dict:
        """Run real-time inference loop on camera feed or video file."""
        if source is None:
            source = self.config.get("source", {}).get("device_index", 0)

        if isinstance(source, str) and source.isdigit():
            source = int(source)

        console.print(f"[bold cyan]Opening Video Source:[/bold cyan] {source}")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            console.print(f"[bold red]ERROR: Cannot access camera or source {source}[/bold red]")
            return {"error": f"Failed to open source {source}"}

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0

        console.print(f"[dim]Stream Specs: {frame_width}x{frame_height} @ {fps_source:.1f} FPS[/dim]")
        console.print(f"[bold green]3D Physics Gate:[/bold green] Mode={self.counter.mode.upper()} | Gate Z={self.counter.target_depth_meters}m | Dir={self.counter.direction.upper()}\n")

        writer = None
        if self.save_video:
            os.makedirs(self.save_path, exist_ok=True)
            out_file = os.path.join(self.save_path, "human_physics_demo_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_file, fourcc, fps_source, (frame_width, frame_height))

        window_name = "Wilfred's Lab - 3D Human Counter (Physics Mode)"
        if self.display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if self.fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        frame_count = 0
        start_time = time.time()
        fps_calc = 0.0

        console.print(Panel(
            "[bold bright_white]3D PHYSICS HUMAN COUNTER PIPELINE RUNNING[/bold bright_white]\n\n"
            "  [bold green]Interactive Lab Controls:[/bold green]\n"
            "   - Press [bold white]'q'[/bold white] to stop pipeline & print session summary card\n"
            "   - Press [bold white]'r'[/bold white] to reset virtual line counter & ID history\n"
            "   - Press [bold white]'m'[/bold white] to toggle Gate Mode (frontal_depth <-> sideways_wall)\n"
            "   - Press [bold white]'t'[/bold white] to toggle Tracker Algorithm (bytetrack <-> botsort)\n"
            "   - Press [bold white]'f'[/bold white] to toggle Fullscreen mode",
            border_style="bright_magenta",
            title="[bold bright_magenta] WILFRED'S LAB STREAM ACTIVE [/bold bright_magenta]",
            box=box.DOUBLE_EDGE,
        ))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                t0 = time.time()

                # Stage 1 & 2: Detection and Tracking
                if self.tracker.algorithm == "botsort":
                    tracked = self.tracker.update(None, detector=self.detector, frame=frame)
                else:
                    detections = self.detector.detect_supervision(frame)
                    tracked = self.tracker.update(detections)

                # Stage 3: Physics Gate Engine with Visual Re-ID
                count_res = self.counter.update(tracked, frame, frame_width, frame_height)

                # Stage 4: Logging
                if self.logger and count_res["new_counts"]:
                    for ent_id in count_res["new_counts"]:
                        self.logger.log_crossing(frame_count, ent_id, count_res["total_count"])

                # Stage 5: Overlay Visualization with Pointer Rays
                if self.display or writer:
                    frame = draw_physics_virtual_gate(
                        frame,
                        mode=self.counter.mode,
                        target_depth_meters=self.counter.target_depth_meters,
                        wall_x_ratio=self.counter.wall_x_ratio,
                        focal_length_px=self.counter.focal_length_px,
                        human_height_meters=self.counter.human_height_meters,
                    )

                    frame = draw_physics_pointers_and_telemetry(
                        frame,
                        telemetry=count_res["telemetry"],
                        counted_entities=self.counter.counted_entities,
                        mode=self.counter.mode,
                        wall_x_ratio=self.counter.wall_x_ratio,
                    )

                    frame = draw_hud_physics(
                        frame,
                        count_res["total_count"],
                        count_res["count_in"],
                        count_res["count_out"],
                        count_res["active_tracks"],
                        fps_calc,
                        self.counter.mode,
                        self.counter.target_depth_meters,
                        self.tracker.algorithm,
                    )

                    if count_res["new_counts"]:
                        frame = draw_new_count_flash_physics(frame, count_res["new_counts"])

                    if writer:
                        writer.write(frame)

                    if self.display:
                        cv2.imshow(window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            console.print("[bold yellow]Session stopped by user.[/bold yellow]")
                            break
                        elif key == ord("r"):
                            self.counter.reset()
                            self.tracker.reset()
                        elif key == ord("m"):
                            self.counter.mode = "sideways_wall" if self.counter.mode == "frontal_depth" else "frontal_depth"
                            console.print(f"[bold cyan]Toggled Gate Mode to:[/bold cyan] [bold yellow]{self.counter.mode.upper()}[/bold yellow]")
                        elif key == ord("t"):
                            new_algo = "botsort" if self.tracker.algorithm == "bytetrack" else "bytetrack"
                            self.tracker = HumanTracker(
                                algorithm=new_algo,
                                track_activation_threshold=self.config.get("tracking", {}).get("track_high_thresh", 0.50),
                                lost_track_buffer=self.config.get("tracking", {}).get("track_buffer", 60),
                                minimum_matching_threshold=self.config.get("tracking", {}).get("match_thresh", 0.80),
                                frame_rate=self.config.get("source", {}).get("fps", 30),
                            )
                            console.print(f"[bold cyan]Toggled Tracker to:[/bold cyan] [bold yellow]{new_algo.upper()}[/bold yellow]")
                        elif key == ord("f"):
                            self.fullscreen = not self.fullscreen
                            val = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, val)

                fps_calc = 1.0 / max(time.time() - t0, 1e-6)

        except KeyboardInterrupt:
            console.print("[bold yellow]Keyboard interrupt received.[/bold yellow]")
        finally:
            cap.release()
            if writer:
                writer.release()
            if self.display:
                cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        avg_fps = frame_count / max(elapsed, 1e-6)

        if self.logger:
            self.logger.log_session_summary(
                self.counter.total_count, frame_count, avg_fps, str(source)
            )

        # Final Telemetry Summary Card
        console.print("\n")
        console.print(Panel(
            f"[bold bright_white]Wilfred's Lab - Physics Gate Results[/bold bright_white]\n\n"
            f"  [bold cyan]Total Humans Counted:[/bold cyan]  [bold bright_green]{self.counter.total_count}[/bold bright_green] (IN: {self.counter.count_in} | OUT: {self.counter.count_out})\n"
            f"  [bold cyan]Total Frames Processed:[/bold cyan] {frame_count}\n"
            f"  [bold cyan]Elapsed Session Time:[/bold cyan]   {elapsed:.2f}s\n"
            f"  [bold cyan]Average Frame Rate:[/bold cyan]     {avg_fps:.1f} FPS\n"
            f"  [bold cyan]Unique Entities Counted:[/bold cyan]{len(self.counter.counted_entities)}",
            border_style="bright_green",
            title="[bold bright_green] EXECUTION SUMMARY [/bold bright_green]",
            box=box.DOUBLE_EDGE,
        ))

        return {
            "total_count": self.counter.total_count,
            "count_in": self.counter.count_in,
            "count_out": self.counter.count_out,
            "frames": frame_count,
            "fps": avg_fps,
        }
