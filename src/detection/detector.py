"""
detector.py - Zero-shot YOLOv8 Human Detector module.

Uses pre-trained YOLOv8 weights (COCO dataset, Class 0 = Person)
to perform real-time detection on webcam frames.
"""

import numpy as np
import torch
from ultralytics import YOLO
from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()


class HumanDetector:
    """YOLOv8 Wrapper tailored for human detection (COCO Class 0)."""

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        target_class_id: int = 0,
        conf_thresh: float = 0.50,
        iou_thresh: float = 0.45,
        imgsz: int = 640,
        device: str = "0",
    ):
        """Initialize the Human Detector with specified confidence and hardware device."""
        self.weights_path = weights_path
        self.target_class_id = target_class_id
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        
        # Fall back to CPU if CUDA isn't available on device string "0"
        if device == "0" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        console.print(Panel(
            f"[bold bright_white]Human Detector Zoo Architecture[/bold bright_white]\n"
            f"  [cyan]Weights Source:[/cyan]   [bold yellow]{weights_path}[/bold yellow] (COCO Pretrained)\n"
            f"  [cyan]Target Class:[/cyan]     [bold white]ID {target_class_id} (person)[/bold white]\n"
            f"  [cyan]Inference Dev:[/cyan]    [bold green]{self.device.upper()}[/bold green] | Conf: {conf_thresh} | IoU: {iou_thresh}\n"
            f"  [cyan]Image Size:[/cyan]       {imgsz}x{imgsz}",
            border_style="bright_cyan",
            title="[bold bright_cyan]DETECTOR INIT[/bold bright_cyan]",
            box=box.DOUBLE_EDGE,
        ))

        self.model = YOLO(weights_path)
        console.print("[bold black on bright_green] READY [/bold black on bright_green] "
                      "Zero-shot YOLOv8 detector loaded successfully.")

    def detect_supervision(self, frame: np.ndarray):
        """Run detection on frame and return supervision-compatible Detections filtered for human class."""
        import supervision as sv

        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            classes=[self.target_class_id],
            verbose=False,
        )[0]

        return sv.Detections.from_ultralytics(results)

    def track_supervision(self, frame: np.ndarray, tracker_type: str = "botsort.yaml"):
        """Run YOLOv8 object detection + native tracker (e.g. BoT-SORT / DeepSORT with Re-ID embeddings)."""
        import supervision as sv

        results = self.model.track(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            classes=[self.target_class_id],
            tracker=tracker_type,
            persist=True,
            verbose=False,
        )[0]

        return sv.Detections.from_ultralytics(results)
