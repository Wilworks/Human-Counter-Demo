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
    """Multi-Object Tracker wrapper supporting ByteTrack and BoT-SORT (DeepSORT)."""

    def __init__(
        self,
        algorithm: str = "bytetrack",
        track_activation_threshold: float = 0.50,
        lost_track_buffer: int = 60,
        minimum_matching_threshold: float = 0.80,
        frame_rate: int = 30,
    ):
        """Initialize tracker instance based on requested algorithm."""
        self.algorithm = str(algorithm).lower()
        
        if self.algorithm == "botsort":
            self.tracker = None
            console.print("[bold black on bright_yellow] READY [/bold black on bright_yellow] "
                          "BoT-SORT (DeepSORT - Kalman Filter + Appearance Re-ID) active.")
        else:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=frame_rate,
            )
            console.print("[bold black on bright_cyan] READY [/bold black on bright_cyan] "
                          f"ByteTrack tracker active (Buffer={lost_track_buffer} frames)")

    def update(self, detections: sv.Detections = None, detector=None, frame=None) -> sv.Detections:
        """Update tracker with frame detections or direct model tracking."""
        if self.algorithm == "botsort":
            if detector is None or frame is None:
                raise ValueError("BoT-SORT requires detector and frame parameters")
            return detector.track_supervision(frame, tracker_type="botsort.yaml")
        else:
            if detections is None:
                raise ValueError("ByteTrack requires supervision detections input")
            return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Reset tracker state."""
        if self.tracker is not None:
            self.tracker.reset()
