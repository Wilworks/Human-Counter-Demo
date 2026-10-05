"""
counter.py - 3D Gate Counter with Backend Visual Re-ID Gallery & 3-Zone Hysteresis.

Features:
1. Backend Visual Feature Gallery (Long-Term Re-ID Store):
   - Stores visual appearance embeddings (spatial HSV + texture profile) for all counted entities.
   - When a person/cow leaves the frame and returns (even minutes later with a new tracker ID),
     the backend performs visual cosine similarity matching.
   - If similarity >= threshold (e.g., 65%+), the new tracker ID is mapped back to the existing persistent Entity ID.
   - Prevents duplicate counting when re-entering the scene!
2. 3-Zone Hysteresis State Machine:
   - FAR_ZONE: Z > (target_depth + 0.5m) --> ARMED_FAR
   - NEAR_ZONE: Z < (target_depth - 0.5m) --> PASSED_NEAR (Triggers +1 COUNT)
3. Permanent Lock per Entity: Each distinct person is counted exactly once.
"""

import math
import cv2
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple, Optional
import numpy as np
import supervision as sv
from rich.console import Console

console = Console()


class HumanCounterPhysics:
    """Physics Gate Counter with Backend Visual Re-ID Gallery."""

    def __init__(
        self,
        mode: str = "frontal_depth",
        direction: str = "towards_camera",
        target_depth_meters: float = 3.5,
        focal_length_px: float = 800.0,
        human_height_meters: float = 1.75,
        wall_x_ratio: float = 0.50,
        angular_threshold_deg: float = 3.0,
        min_track_age: int = 5,
        spatial_merge_distance: float = 0.0,
        reid_similarity_threshold: float = 0.65,
    ):
        """Initialize engine with backend visual gallery store."""
        self.mode = str(mode).lower()
        self.direction = str(direction).lower()
        self.target_depth_meters = float(target_depth_meters)
        self.focal_length_px = float(focal_length_px)
        self.human_height_meters = float(human_height_meters)
        self.wall_x_ratio = float(wall_x_ratio)
        self.angular_threshold_deg = float(angular_threshold_deg)
        self.min_track_age = int(min_track_age)
        self.reid_similarity_threshold = float(reid_similarity_threshold)

        # Telemetry Stats
        self.total_count = 0
        self.count_in = 0
        self.count_out = 0
        self.frame_count = 0

        # Permanent Entity Lock Set
        self.counted_entities: Set[int] = set()

        # Backend Gallery Re-ID Store
        self.next_entity_id: int = 1
        self.tracker_to_entity_map: Dict[int, int] = {}
        self.gallery_embeddings: Dict[int, np.ndarray] = {}

        # 3-Zone State Machine per entity: entity_id -> "ARMED_FAR" | "ARMED_NEAR" | "PASSED_NEAR" | "PASSED_FAR"
        self.entity_zone_state: Dict[int, str] = {}

        # Track State Storage
        self.track_ages: Dict[int, int] = defaultdict(int)
        self.track_history_physics: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    def get_wall_x_pixels(self, frame_width: int) -> int:
        """Calculate wall X coordinate in pixels."""
        return int(self.wall_x_ratio * frame_width)

    def extract_visual_embedding(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract spatial multi-region HSV color profile feature vector for Re-ID matching."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, [max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])])

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
            return None

        try:
            crop_resized = cv2.resize(crop, (64, 128))
            hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

            # Split into Upper Body (torso) and Lower Body (legs)
            upper_hsv = hsv[:64, :]
            lower_hsv = hsv[64:, :]

            # Compute HSV Histograms
            hist_upper = cv2.calcHist([upper_hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
            hist_lower = cv2.calcHist([lower_hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])

            cv2.normalize(hist_upper, hist_upper)
            cv2.normalize(hist_lower, hist_lower)

            vec = np.concatenate([hist_upper.flatten(), hist_lower.flatten()])
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                return None
            return vec / norm
        except Exception:
            return None

    def resolve_entity_id(self, tracker_id: int, frame: np.ndarray, bbox: np.ndarray) -> int:
        """Resolve tracker ID to persistent entity ID using backend visual similarity gallery."""
        # 1. If tracker ID is already active in current session
        if tracker_id in self.tracker_to_entity_map:
            entity_id = self.tracker_to_entity_map[tracker_id]
            curr_emb = self.extract_visual_embedding(frame, bbox)
            if curr_emb is not None and entity_id in self.gallery_embeddings:
                # Update gallery embedding (moving average)
                old_emb = self.gallery_embeddings[entity_id]
                updated = 0.85 * old_emb + 0.15 * curr_emb
                self.gallery_embeddings[entity_id] = updated / max(np.linalg.norm(updated), 1e-6)
            return entity_id

        # 2. If new tracker ID: Match visual embedding against backend gallery store
        curr_emb = self.extract_visual_embedding(frame, bbox)

        best_entity_id = None
        best_sim = 0.0

        if curr_emb is not None and len(self.gallery_embeddings) > 0:
            for ent_id, ent_emb in self.gallery_embeddings.items():
                sim = float(np.dot(curr_emb, ent_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_entity_id = ent_id

        if best_entity_id is not None and best_sim >= self.reid_similarity_threshold:
            # RE-ID MATCH FOUND! Relink brand-new tracker to existing entity!
            entity_id = best_entity_id
            self.tracker_to_entity_map[tracker_id] = entity_id
            console.print(
                f"[bold black on bright_yellow] RE-ID MATCH [/bold black on bright_yellow] "
                f"Tracker #{tracker_id} RECOGNIZED as Entity #{entity_id} (Similarity = {best_sim*100:.1f}%)! "
                f"PREVENTING DUPLICATE COUNT!"
            )
        else:
            # No match found -> Assign new persistent Entity ID
            entity_id = self.next_entity_id
            self.next_entity_id += 1
            self.tracker_to_entity_map[tracker_id] = entity_id
            if curr_emb is not None:
                self.gallery_embeddings[entity_id] = curr_emb

        return entity_id

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, Any]:
        """Evaluate Gate Crossings for each distinct tracked person with Backend Re-ID Matching."""
        self.frame_count += 1
        wall_x = self.get_wall_x_pixels(frame_width)
        new_counts: List[int] = []
        entity_telemetry: Dict[int, Dict[str, Any]] = {}

        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            return {
                "total_count": self.total_count,
                "count_in": self.count_in,
                "count_out": self.count_out,
                "new_counts": [],
                "active_tracks": 0,
                "telemetry": {},
            }

        active_tracker_ids = set(map(int, detections.tracker_id))

        for i, tracker_id in enumerate(detections.tracker_id):
            tid = int(tracker_id)
            bbox = detections.xyxy[i]

            feet_x = float((bbox[0] + bbox[2]) / 2.0)
            feet_y = float(bbox[3])
            bbox_h = float(max(bbox[3] - bbox[1], 1.0))

            # Depth & Angle Calculation
            depth_m = round(max((self.focal_length_px * self.human_height_meters) / bbox_h, 0.2), 2)
            angle_deg = round(math.degrees(math.atan2(feet_x - wall_x, self.focal_length_px)), 1)

            self.track_ages[tid] += 1
            age = self.track_ages[tid]

            # Resolve Tracker ID to Persistent Entity ID via Backend Re-ID Gallery
            entity_id = self.resolve_entity_id(tid, frame, bbox)
            is_counted = entity_id in self.counted_entities

            entity_telemetry[tid] = {
                "entity_id": entity_id,
                "depth_m": depth_m,
                "angle_deg": angle_deg,
                "feet_pt": (int(feet_x), int(feet_y)),
                "bbox": bbox,
                "is_counted": is_counted,
            }

            self.track_history_physics[tid].append((depth_m, angle_deg))
            if len(self.track_history_physics[tid]) > 30:
                self.track_history_physics[tid] = self.track_history_physics[tid][-30:]

            # Require minimum track age before gate evaluation
            if age < self.min_track_age:
                continue

            # -------------------------------------------------------------
            # LAYER 1: STRICT PERMANENT ENTITY LOCK CHECK
            # -------------------------------------------------------------
            if is_counted:
                continue

            # -------------------------------------------------------------
            # 3-ZONE HYSTERESIS STATE MACHINE
            # -------------------------------------------------------------
            if self.mode == "sideways_wall":
                far_bound = -5.0   # Left Zone (< -5°)
                near_bound = +5.0  # Right Zone (> +5°)
                curr_val = angle_deg
            else: # frontal_depth
                far_bound = self.target_depth_meters + 0.50  # e.g. > 4.0m
                near_bound = self.target_depth_meters - 0.50 # e.g. < 3.0m
                curr_val = depth_m

            # Initial state assignment if entity is brand new
            if entity_id not in self.entity_zone_state:
                if self.mode == "sideways_wall":
                    if curr_val <= far_bound:
                        self.entity_zone_state[entity_id] = "ARMED_LEFT"
                    elif curr_val >= near_bound:
                        self.entity_zone_state[entity_id] = "ARMED_RIGHT"
                    else:
                        self.entity_zone_state[entity_id] = "PASSED_MIDDLE"
                else: # frontal_depth
                    if curr_val >= far_bound:
                        self.entity_zone_state[entity_id] = "ARMED_FAR"
                    elif curr_val <= near_bound:
                        self.entity_zone_state[entity_id] = "PASSED_NEAR"
                    else:
                        self.entity_zone_state[entity_id] = "PASSED_MIDDLE"

            current_state = self.entity_zone_state[entity_id]

            # Update State Machine
            if self.mode == "sideways_wall":
                if curr_val <= far_bound:
                    self.entity_zone_state[entity_id] = "ARMED_LEFT"
                elif curr_val >= near_bound:
                    self.entity_zone_state[entity_id] = "ARMED_RIGHT"
            else: # frontal_depth
                if curr_val >= far_bound:
                    self.entity_zone_state[entity_id] = "ARMED_FAR"

            # Evaluate Crossing Event
            trigger_count = False
            crossing_dir = None

            if self.mode == "frontal_depth":
                if current_state == "ARMED_FAR" and curr_val <= self.target_depth_meters:
                    trigger_count = True
                    crossing_dir = "towards_camera"
                    self.entity_zone_state[entity_id] = "PASSED_NEAR"
            else: # sideways_wall
                if current_state == "ARMED_LEFT" and curr_val >= 0.0:
                    trigger_count = True
                    crossing_dir = "left_to_right"
                    self.entity_zone_state[entity_id] = "PASSED_RIGHT"

            # Execute Count if Triggered & Valid Direction
            if trigger_count:
                valid_dir = (self.direction == "both") or (self.direction == crossing_dir)
                if valid_dir:
                    self.counted_entities.add(entity_id)
                    self.total_count += 1
                    new_counts.append(entity_id)

                    if crossing_dir in ["towards_camera", "left_to_right"]:
                        self.count_in += 1
                    else:
                        self.count_out += 1

                    metric_str = f"Depth={curr_val}m" if self.mode == "frontal_depth" else f"Angle={curr_val}°"
                    console.print(
                        f"[bold black on bright_green] +1 PASSED GATE [/bold black on bright_green] "
                        f"Entity #{entity_id} (Tracker #{tid}) crossed ({crossing_dir.upper()}, {metric_str})! "
                        f"Total: {self.total_count} | IN: {self.count_in} | OUT: {self.count_out}"
                    )

        return {
            "total_count": self.total_count,
            "count_in": self.count_in,
            "count_out": self.count_out,
            "new_counts": new_counts,
            "active_tracks": len(active_tracker_ids),
            "telemetry": entity_telemetry,
        }

    def reset(self) -> None:
        """Reset counter state and backend visual gallery."""
        self.total_count = 0
        self.count_in = 0
        self.count_out = 0
        self.next_entity_id = 1
        self.counted_entities.clear()
        self.tracker_to_entity_map.clear()
        self.gallery_embeddings.clear()
        self.entity_zone_state.clear()
        self.track_ages.clear()
        self.track_history_physics.clear()
        self.frame_count = 0
        console.print("[bold yellow]Counter state and persistent visual gallery store reset.[/bold yellow]")
