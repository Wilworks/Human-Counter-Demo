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
    """Render 3D physical gate boundary (Ground Depth Plane or Vertical Wall Plane)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if mode == "sideways_wall":
        x_wall = int(wall_x_ratio * w)

        # Draw Vertical Wall Plane (0° Angle Line)
        for y_step in range(0, h, 40):
            cv2.line(overlay, (x_wall - 25, y_step), (x_wall + 25, y_step + 20), color, 1)

        pts = np.array([[x_wall - 30, 0], [x_wall + 30, 0], [x_wall + 30, h], [x_wall - 30, h]], np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Center Wall Axis (0.0° Plane)
        cv2.line(frame, (x_wall, 0), (x_wall, h), (255, 255, 255), 3)
        cv2.line(frame, (x_wall - 15, 0), (x_wall - 15, h), color, 1)
        cv2.line(frame, (x_wall + 15, 0), (x_wall + 15, h), color, 1)

        cv2.putText(frame, f"VERTICAL WALL GATE (0.0 deg Plane)", (max(10, x_wall - 140), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    else:
        # Frontal Depth Gate Plane: Calculate Y position corresponding to target_depth_meters
        # h_bbox = (f * H) / Z
        gate_bbox_h = (focal_length_px * human_height_meters) / target_depth_meters
        # Map depth to Y coordinate on screen
        y_gate = int(h * 0.65) # Visual depth line

        # Ground Plane Grid
        for x_step in range(0, w, 60):
            cv2.line(overlay, (x_step, y_gate), (int(w / 2 + (x_step - w / 2) * 1.5), h), color, 1)

        pts = np.array([[0, y_gate - 15], [w, y_gate - 15], [w, y_gate + 15], [0, y_gate + 15]], np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

        # Baseline
        cv2.line(frame, (0, y_gate), (w, y_gate), (255, 255, 255), 2)
        cv2.line(frame, (0, y_gate - 15), (w, y_gate - 15), color, 1)
        cv2.line(frame, (0, y_gate + 15), (w, y_gate + 15), color, 1)

        # Standing Pillars
        cv2.line(frame, (40, y_gate), (40, y_gate - 180), color, 3)
        cv2.line(frame, (w - 40, y_gate), (w - 40, y_gate - 180), color, 3)
        cv2.line(frame, (40, y_gate - 180), (w - 40, y_gate - 180), color, 2)

        cv2.putText(frame, f"3D DEPTH GATE (Target Z = {target_depth_meters:.1f}m)", (20, max(30, y_gate - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return frame


def draw_physics_pointers_and_telemetry(
    frame: np.ndarray,
    telemetry: Dict[int, Dict[str, Any]],
    counted_entities: Set[int],
    mode: str = "frontal_depth",
    wall_x_ratio: float = 0.50,
) -> np.ndarray:
    """Draw persistent laser pointer rays and live distance/angle telemetry badges."""
    h, w = frame.shape[:2]

    # Camera Origin Anchor Point
    if mode == "sideways_wall":
        cam_origin = (int(wall_x_ratio * w), h)
    else:
        cam_origin = (int(w / 2), h)

    for tid, data in telemetry.items():
        entity_id = data["entity_id"]
        depth_m = data["depth_m"]
        angle_deg = data["angle_deg"]
        feet_pt = data["feet_pt"]
        bbox = data["bbox"].astype(int)

        is_counted = entity_id in counted_entities
        color = (0, 255, 0) if is_counted else (255, 165, 0) # Green if passed, Amber if tracking

        # 1. Bounding Box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # 2. Feet Anchor Point
        cv2.circle(frame, feet_pt, 6, (0, 0, 255), -1)
        cv2.circle(frame, feet_pt, 12, (0, 255, 255), 2)

        # 3. Persistent Laser Pointer Ray (Line tagged from camera anchor to person's feet)
        cv2.line(frame, cam_origin, feet_pt, color, 2, cv2.LINE_AA)
        
        # Ray midpoint distance marker
        mid_pt = (int((cam_origin[0] + feet_pt[0]) / 2), int((cam_origin[1] + feet_pt[1]) / 2))
        metric_txt = f"{depth_m}m" if mode == "frontal_depth" else f"{angle_deg}°"
        cv2.putText(frame, metric_txt, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Live Telemetry Badge Pill above Bounding Box
        status_tag = " [PASSED]" if is_counted else ""
        if mode == "frontal_depth":
            badge_str = f"Entity #{entity_id} | d = {depth_m}m{status_tag}"
        else:
            badge_str = f"Entity #{entity_id} | angle = {angle_deg}°{status_tag}"

        (w_txt, h_txt), _ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (bbox[0], bbox[1] - h_txt - 10), (bbox[0] + w_txt + 10, bbox[1]), color, -1)
        cv2.putText(
            frame,
            badge_str,
            (bbox[0] + 5, bbox[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
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
    """Draw top HUD telemetry card with physics metrics."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (450, 135), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (15, 15), (450, 135), (0, 255, 255), 1)

    mode_label = f"FRONTAL (Gate Z={target_depth:.1f}m)" if mode == "frontal_depth" else "SIDEWAYS (Gate Angle=0.0 deg)"

    cv2.putText(frame, f"WILFRED'S LAB - 3D PHYSICS GATE", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"TOTAL PASSED: {total_count}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"IN: {count_in}  |  OUT: {count_out}  |  MODE: {mode.upper()}", (25, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ACTIVE: {active_tracks}  |  FPS: {fps:.1f}  |  TRK: {tracker_name.upper()}", (25, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def draw_new_count_flash_physics(frame: np.ndarray, new_entities: List[int]) -> np.ndarray:
    """Flash border accent on frame when a new entity crossing occurs."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)
    msg = f"PASSED GATE: Entity #{','.join(map(str, new_entities))}"
    cv2.putText(frame, msg, (int(w / 2) - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 3, cv2.LINE_AA)
    return frame
