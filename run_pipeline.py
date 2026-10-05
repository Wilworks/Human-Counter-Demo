#!/usr/bin/env python3
"""
run_pipeline.py - Master bootstrap script for Human Counter Feasibility Demo.

Handles environment sanity checks, hardware cards, rich telemetry,
and launches real-time laptop webcam human detection and line crossing counter.
"""

import os
import sys
import argparse

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def display_hero_banner():
    """Render glowing hero banner with double-edge border."""
    console.print(Panel(
        "[bold bright_white]HUMAN COUNTER FEASIBILITY DEMO[/bold bright_white]\n"
        "[bold bright_cyan]Zero-Shot Human Detection + ByteTrack + 3-Layer Anti-Double-Count Gate[/bold bright_cyan]\n"
        "[dim]Proof-of-Concept Pipeline for Lab Verification & Real-Time Telemetry[/dim]",
        border_style="bright_cyan",
        title="[bold bright_cyan]HUMAN COUNTER DEMO[/bold bright_cyan]",
        box=box.DOUBLE_EDGE,
    ))


def check_environment() -> bool:
    """Verify hardware and python package dependencies."""
    checks = []

    # Hardware Telemetry Card
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU Only (CUDA Unavailable)"
        checks.append(("PyTorch Framework", torch.__version__, True, True))
        checks.append(("CUDA Acceleration", "Enabled" if cuda_ok else "Disabled (CPU Mode)", cuda_ok, False))
        checks.append(("Primary Compute Device", gpu_name, True, True))
    except ImportError:
        checks.append(("PyTorch Framework", "NOT INSTALLED", False, True))

    # Computer Vision Packages
    try:
        import ultralytics
        checks.append(("Ultralytics YOLOv8", ultralytics.__version__, True, True))
    except ImportError:
        checks.append(("Ultralytics YOLOv8", "NOT INSTALLED", False, True))

    try:
        import supervision
        checks.append(("Supervision Tracking Kit", supervision.__version__, True, True))
    except ImportError:
        checks.append(("Supervision Tracking Kit", "NOT INSTALLED", False, True))

    try:
        import cv2
        checks.append(("OpenCV Library", cv2.__version__, True, True))
    except ImportError:
        checks.append(("OpenCV Library", "NOT INSTALLED", False, True))

    table = Table(title="[bold cyan]Runtime Environment & Telemetry Matrix[/bold cyan]", box=box.ROUNDED)
    table.add_column("Component", style="bright_white")
    table.add_column("Detected Version / Hardware", style="dim")
    table.add_column("Status Pill", justify="center")

    all_pass = True
    for name, version, ok, required in checks:
        if required and not ok:
            all_pass = False

        if ok:
            pill = "[bold black on bright_green] PASS [/bold black on bright_green]"
        elif not required:
            pill = "[bold black on bright_yellow] INFO [/bold black on bright_yellow]"
        else:
            pill = "[bold black on bright_red] FAIL [/bold black on bright_red]"

        table.add_row(name, version, pill)

    console.print(table)
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Human Counter Feasibility Demo - Lab Real-Time Pipeline")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to config file")
    parser.add_argument("--source", type=str, default=None, help="Video source (0 for webcam, or video path)")
    parser.add_argument("--check", action="store_true", help="Run environment check only")
    args = parser.parse_args()

    display_hero_banner()

    console.print("\n[bold white]Running Diagnostic Environment Telemetry...[/bold white]")
    env_ok = check_environment()

    if args.check:
        if env_ok:
            console.print("\n[bold black on bright_green] ⚡ READY [/bold black on bright_green] Environment checks passed.")
        else:
            console.print("\n[bold black on bright_red] FAIL [/bold black on bright_red] Missing dependencies. Run: pip install -r requirements.txt")
        return

    if not env_ok:
        console.print("\n[bold black on bright_red] CRITICAL [/bold black on bright_red] Environment verification failed!")
        console.print("Please install requirements using: [yellow]pip install -r requirements.txt[/yellow]")
        sys.exit(1)

    # Launch Pipeline
    from src.pipeline.pipeline import HumanCountingPipeline
    pipeline = HumanCountingPipeline(args.config)
    pipeline.run(args.source)


if __name__ == "__main__":
    main()
