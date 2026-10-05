"""
visualization.py - Physics-Based 3D Visualizer & Laser Ray Telemetry.
"""

from typing import Set, List, Dict, Any
import cv2
import numpy as np
import supervision as sv


def draw_physics_virtual_gate(
    frame: np.ndarray,
    mode: str = "frontal_depth",
    target_depth_meters: float = 3.5,
    wall_x_ratio: float = 0.50,
    focal_length_px: float = 800.0,
    human_height_meters: float = 1.75,
    color: tuple = (0, 255, 255),
) -> np.ndarray:
    """Render clean, minimal gate line across screen."""
    h, w = frame.shape[:2]

    if mode == "sideways_wall":
        x_wall = int(wall_x_ratio * w)
        # Vertical line
        cv2.line(frame, (x_wall, 0), (x_wall, h), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GATE (0.0°)", (max(10, x_wall - 50), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # Horizontal gate line at depth threshold
        y_gate = int(h * 0.65)
        cv2.line(frame, (0, y_gate), (w, y_gate), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"GATE ({target_depth_meters:.1f}m)", (15, y_gate - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_physics_pointers_and_telemetry(
    frame: np.ndarray,
    telemetry: Dict[int, Dict[str, Any]],
    counted_entities: Set[int],
    mode: str = "frontal_depth",
    wall_x_ratio: float = 0.50,
) -> np.ndarray:
    """Draw clean, sleek bounding boxes and subtle text labels."""
    for tid, data in telemetry.items():
        entity_id = data["entity_id"]
        depth_m = data["depth_m"]
        angle_deg = data["angle_deg"]
        bbox = data["bbox"].astype(int)

        is_counted = entity_id in counted_entities
        color = (0, 255, 0) if is_counted else (255, 200, 0) # Green if counted, Gold if tracking

        # 1. Sleek 2px Bounding Box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # 2. Minimalist Text Label above box (no block background obscuring face)
        status_tag = " [COUNTED]" if is_counted else ""
        if mode == "frontal_depth":
            badge_str = f"#{entity_id} | {depth_m}m{status_tag}"
        else:
            badge_str = f"#{entity_id} | {angle_deg}°{status_tag}"

        cv2.putText(
            frame,
            badge_str,
            (bbox[0], max(15, bbox[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_hud_physics(
    frame: np.ndarray,
    total_count: int,
    count_in: int,
    count_out: int,
    active_tracks: int,
    fps: float,
    mode: str,
    target_depth: float,
    tracker_name: str = "bytetrack",
) -> np.ndarray:
    """Draw sleek semi-transparent HUD overlay."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (380, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
    cv2.rectangle(frame, (15, 15), (380, 100), (80, 80, 80), 1)

    cv2.putText(frame, f"COUNT: {total_count}  (IN: {count_in} | OUT: {count_out})", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"MODE: {mode.upper()} | GATE: {target_depth:.1f}m", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ACTIVE: {active_tracks} | FPS: {fps:.1f} | TRACKER: {tracker_name.upper()}", (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def draw_new_count_flash_physics(frame: np.ndarray, new_entities: List[int]) -> np.ndarray:
    """Flash border accent on frame when a new entity crossing occurs."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 6)
    msg = f"COUNT +1: Entity #{','.join(map(str, new_entities))}"
    cv2.putText(frame, msg, (int(w / 2) - 150, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    return frame
