"""
tracker.py - ByteTrack Multi-Object Tracker integration for human tracking.
"""

import warnings
import numpy as np
import supervision as sv
from rich.console import Console

console = Console()
warnings.filterwarnings("ignore", category=FutureWarning)


class HumanTracker:
    """ByteTrack wrapper for multi-person tracking and persistent ID assignment."""

    def __init__(
        self,
        track_activation_threshold: float = 0.50,
        lost_track_buffer: int = 60,
        minimum_matching_threshold: float = 0.80,
        frame_rate: int = 30,
    ):
        """Initialize ByteTrack tracker instance."""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        console.print("[bold black on bright_cyan] READY [/bold black on bright_cyan] "
                      f"ByteTrack tracker active (Buffer={lost_track_buffer} frames)")

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with frame detections and return tracked bounding boxes with IDs."""
        return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracker.reset()
