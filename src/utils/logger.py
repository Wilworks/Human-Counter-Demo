"""
logger.py - CSV/JSON Logging telemetry for crossing events and session summaries.
"""

import os
import csv
from datetime import datetime
from rich.console import Console

console = Console()


class CountLogger:
    """Logs individual line crossings and session summaries to disk."""

    def __init__(self, output_dir: str = "output/"):
        """Initialize logger directory and files."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.csv_path = os.path.join(output_dir, "human_crossing_log.csv")
        
        # Write CSV header if file doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "frame_number", "track_id", "total_count"])

    def log_crossing(self, frame_num: int, track_id: int, total_count: int) -> None:
        """Record crossing event to CSV log."""
        timestamp = datetime.now().isoformat()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, frame_num, track_id, total_count])

    def log_session_summary(self, total_count: int, total_frames: int, avg_fps: float, source: str) -> None:
        """Write session summary to log file."""
        summary_path = os.path.join(self.output_dir, "session_summary.txt")
        with open(summary_path, "w") as f:
            f.write("=== HUMAN COUNTER FEASIBILITY DEMO SUMMARY ===\n")
            f.write(f"Timestamp:        {datetime.now().isoformat()}\n")
            f.write(f"Video Source:     {source}\n")
            f.write(f"Total Count:      {total_count}\n")
            f.write(f"Frames Processed: {total_frames}\n")
            f.write(f"Average FPS:      {avg_fps:.2f}\n")
        console.print(f"[dim]Session summary saved to: {summary_path}[/dim]")
