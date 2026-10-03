"""
counter.py - Physics-Based 3D Gate Engine with 3-Zone Hysteresis & Visual Re-ID.

Features:
1. Permanent Entity Count Lock: Once an entity_id is in self.counted_entities, it CANNOT trigger +1 again under any circumstance!
2. 3-Zone Hysteresis State Machine:
   - FAR_ZONE: Z > (target_depth + 0.5m) --> State = ARMED_FAR (Ready to count towards camera)
   - NEAR_ZONE: Z < (target_depth - 0.5m) --> State = PASSED_NEAR
   - Tracks born inside/near camera start as PASSED_NEAR (cannot trigger false count on spawn).
3. Pose-Invariant Multi-Template Visual Re-ID Engine.
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
    """Physics-Based 3D Gate Engine with 3-Zone Hysteresis & Visual Re-ID."""

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
        spatial_merge_distance: float = 90.0,
        reid_similarity_threshold: float = 0.65,
    ):
        """Initialize engine with 3-Zone Hysteresis & Permanent Entity Lock."""
        self.mode = str(mode).lower()
        self.direction = str(direction).lower()
        self.target_depth_meters = float(target_depth_meters)
        self.focal_length_px = float(focal_length_px)
        self.human_height_meters = float(human_height_meters)
        self.wall_x_ratio = float(wall_x_ratio)
        self.angular_threshold_deg = float(angular_threshold_deg)
        self.min_track_age = int(min_track_age)
        self.spatial_merge_distance = float(spatial_merge_distance)
        self.reid_similarity_threshold = float(reid_similarity_threshold)

        # Telemetry Stats
        self.total_count = 0
        self.count_in = 0
        self.count_out = 0
        self.frame_count = 0

        # Entity Linkage Memory
        self.tracker_to_entity: Dict[int, int] = {}
        self.entity_next_id = 1
        self.counted_entities: Set[int] = set()

        # 3-Zone State Machine per entity: entity_id -> "FAR" | "NEAR" | "ARMED_FAR" | "ARMED_NEAR" | "PASSED_NEAR" | "PASSED_FAR"
        self.entity_zone_state: Dict[int, str] = {}

        # Pose-Invariant Gallery: entity_id -> List[np.ndarray]
        self.entity_gallery: Dict[int, List[np.ndarray]] = defaultdict(list)

        # Track State Storage
        self.track_ages: Dict[int, int] = defaultdict(int)
        self.track_history_physics: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.track_accumulated_features: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.recent_lost_entities: Dict[int, Tuple[float, float, int]] = {}

    def get_wall_x_pixels(self, frame_width: int) -> int:
        """Calculate wall X coordinate in pixels."""
        return int(self.wall_x_ratio * frame_width)

    def extract_visual_signature(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract normalized HSV color histogram feature vector from bbox crop."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, int(bbox[0])), max(0, int(bbox[1])), min(w, int(bbox[2])), min(h, int(bbox[3]))

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        crop = frame[y1:y2, x1:x2]
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv_crop], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        vec = hist.flatten()

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        return vec

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(vec1, vec2))

    def _resolve_entity(self, tracker_id: int, feet_x: float, feet_y: float, feature_vec: Optional[np.ndarray]) -> int:
        """Map tracker_id to persistent entity_id using Spatial Memory + Multi-Frame Re-ID."""
        # 1. Direct Tracker Memory
        if tracker_id in self.tracker_to_entity:
            entity_id = self.tracker_to_entity[tracker_id]
            self.recent_lost_entities[entity_id] = (feet_x, feet_y, self.frame_count)
            if feature_vec is not None and len(self.entity_gallery[entity_id]) < 5:
                if not any(self._cosine_similarity(feature_vec, gv) > 0.90 for gv in self.entity_gallery[entity_id]):
                    self.entity_gallery[entity_id].append(feature_vec)
            return entity_id

        # Accumulate feature vector
        if feature_vec is not None:
            self.track_accumulated_features[tracker_id].append(feature_vec)
            if len(self.track_accumulated_features[tracker_id]) > 10:
                self.track_accumulated_features[tracker_id] = self.track_accumulated_features[tracker_id][-10:]

        # 2. Short-Term Spatial Proximity Re-Link (for brief dropouts)
        best_spatial_entity = None
        best_dist = float("inf")

        for ent_id, (lx, ly, lframe) in list(self.recent_lost_entities.items()):
            if self.frame_count - lframe <= 60:
                dist = math.hypot(feet_x - lx, feet_y - ly)
                if dist <= self.spatial_merge_distance and dist < best_dist:
                    best_dist = dist
                    best_spatial_entity = ent_id

        if best_spatial_entity is not None:
            self.tracker_to_entity[tracker_id] = best_spatial_entity
            self.recent_lost_entities[best_spatial_entity] = (feet_x, feet_y, self.frame_count)
            console.print(f"[bold yellow]Spatial Re-link:[bold yellow] Tracker #{tracker_id} -> Entity #{best_spatial_entity} (Dist={best_dist:.1f}px)")
            return best_spatial_entity

        # 3. Multi-Frame Visual Cosine Similarity Re-ID Match
        accumulated = self.track_accumulated_features[tracker_id]
        if len(accumulated) > 0 and len(self.entity_gallery) > 0:
            best_reid_entity = None
            best_sim = -1.0

            for ent_id, pose_list in self.entity_gallery.items():
                for gallery_vec in pose_list:
                    for trk_vec in accumulated:
                        sim = self._cosine_similarity(trk_vec, gallery_vec)
                        if sim >= self.reid_similarity_threshold and sim > best_sim:
                            best_sim = sim
                            best_reid_entity = ent_id

            if best_reid_entity is not None:
                self.tracker_to_entity[tracker_id] = best_reid_entity
                self.recent_lost_entities[best_reid_entity] = (feet_x, feet_y, self.frame_count)
                console.print(
                    f"[bold black on bright_green] RE-ID MATCH [/bold black on bright_green] "
                    f"Tracker #{tracker_id} RECOGNIZED as Entity #{best_reid_entity} (Similarity = {best_sim * 100:.1f}%)! "
                    f"PREVENTING DUPLICATE COUNT!"
                )
                return best_reid_entity

        # 4. Brand New Entity Creation
        new_entity = self.entity_next_id
        self.entity_next_id += 1
        self.tracker_to_entity[tracker_id] = new_entity
        self.recent_lost_entities[new_entity] = (feet_x, feet_y, self.frame_count)
        if feature_vec is not None:
            self.entity_gallery[new_entity].append(feature_vec)

        return new_entity

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, Any]:
        """Evaluate 3-Zone Hysteresis Gate Crossings with strict Entity Lock."""
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

            depth_m = round(max((self.focal_length_px * self.human_height_meters) / bbox_h, 0.2), 2)
            angle_deg = round(math.degrees(math.atan2(feet_x - wall_x, self.focal_length_px)), 1)

            self.track_ages[tid] += 1
            age = self.track_ages[tid]

            feature_vec = self.extract_visual_signature(frame, bbox)
            entity_id = self._resolve_entity(tid, feet_x, feet_y, feature_vec)

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
            # If entity_id is ALREADY in self.counted_entities, it CANNOT count again!
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
                        self.entity_zone_state[entity_id] = "PASSED_MIDDLE" # Spawned inside gate -> locked
                else: # frontal_depth
                    if curr_val >= far_bound:
                        self.entity_zone_state[entity_id] = "ARMED_FAR"
                    elif curr_val <= near_bound:
                        self.entity_zone_state[entity_id] = "PASSED_NEAR" # Spawned near camera -> locked
                    else:
                        self.entity_zone_state[entity_id] = "PASSED_MIDDLE" # Spawned on gate line -> locked

            current_state = self.entity_zone_state[entity_id]

            # Update State Machine
            if self.mode == "sideways_wall":
                if curr_val <= far_bound:
                    self.entity_zone_state[entity_id] = "ARMED_LEFT"
                elif curr_val >= near_bound:
                    self.entity_zone_state[entity_id] = "ARMED_RIGHT"
            else: # frontal_depth
                if curr_val >= far_bound: # Depth > 4.0m
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
        """Reset counter state."""
        self.total_count = 0
        self.count_in = 0
        self.count_out = 0
        self.counted_entities.clear()
        self.tracker_to_entity.clear()
        self.entity_gallery.clear()
        self.entity_zone_state.clear()
        self.track_accumulated_features.clear()
        self.entity_next_id = 1
        self.track_ages.clear()
        self.track_history_physics.clear()
        self.recent_lost_entities.clear()
        self.frame_count = 0
        console.print("[bold yellow]Physics counter engine & permanent entity lock memory reset.[/bold yellow]")
