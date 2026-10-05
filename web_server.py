"""
web_server.py - Flask Web Server & Telemetry Backend for Wilfred's Lab.

Provides:
- MJPEG Live Video Feed (/video_feed)
- JSON Telemetry API (/api/status)
- Interactive Config Update API (/api/update_config)
- Memory Reset API (/api/reset)
- Industrial Minimalist Web Dashboard (/)
"""

import os
import sys
import time
import cv2
import yaml
from flask import Flask, render_template_string, Response, jsonify, request, send_from_directory, send_file
from rich.console import Console

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

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
app = Flask(__name__, static_folder="static", static_url_path="/static")

# Pipeline Global State
pipeline_state = {
    "config_path": "config/settings.yaml",
    "source": 0,
    "cap": None,
    "detector": None,
    "tracker": None,
    "counter": None,
    "logger": None,
    "last_result": {
        "total_count": 0,
        "count_in": 0,
        "count_out": 0,
        "active_tracks": 0,
        "fps": 0.0,
        "width": 1280,
        "height": 720,
        "mode": "frontal_depth",
        "telemetry": {},
    },
}


def init_pipeline():
    """Initialize counter components from config."""
    config_path = pipeline_state["config_path"]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    det_cfg = config.get("detection", {})
    trk_cfg = config.get("tracking", {})
    phys_cfg = config.get("physics_gate", {})
    out_cfg = config.get("output", {})

    pipeline_state["detector"] = HumanDetector(
        weights_path=det_cfg.get("model_weights", "yolov8n.pt"),
        target_class_id=det_cfg.get("target_class_id", 0),
        conf_thresh=det_cfg.get("conf_thresh", 0.50),
        iou_thresh=det_cfg.get("iou_thresh", 0.45),
        imgsz=det_cfg.get("imgsz", 640),
        device=det_cfg.get("device", "0"),
    )

    pipeline_state["tracker"] = HumanTracker(
        algorithm=trk_cfg.get("algorithm", "bytetrack"),
        track_activation_threshold=trk_cfg.get("track_high_thresh", 0.50),
        lost_track_buffer=trk_cfg.get("track_buffer", 60),
        minimum_matching_threshold=trk_cfg.get("match_thresh", 0.80),
        frame_rate=config.get("source", {}).get("fps", 30),
    )

    pipeline_state["counter"] = HumanCounterPhysics(
        mode=phys_cfg.get("mode", "frontal_depth"),
        direction=phys_cfg.get("direction", "towards_camera"),
        target_depth_meters=phys_cfg.get("target_depth_meters", 3.5),
        focal_length_px=phys_cfg.get("focal_length_px", 800.0),
        human_height_meters=phys_cfg.get("human_height_meters", 1.75),
        wall_x_ratio=phys_cfg.get("wall_x_ratio", 0.50),
        angular_threshold_deg=phys_cfg.get("angular_threshold_deg", 3.0),
        min_track_age=phys_cfg.get("min_track_age", 5),
        spatial_merge_distance=phys_cfg.get("spatial_merge_distance", 90.0),
        reid_similarity_threshold=phys_cfg.get("reid_similarity_threshold", 0.65),
    )

    save_path = out_cfg.get("save_path", "output/")
    pipeline_state["logger"] = CountLogger(save_path)


def generate_frames():
    """MJPEG Stream generator."""
    if pipeline_state["cap"] is None:
        source = pipeline_state["source"]
        pipeline_state["cap"] = cv2.VideoCapture(source)

    cap = pipeline_state["cap"]
    detector = pipeline_state["detector"]
    tracker = pipeline_state["tracker"]
    counter = pipeline_state["counter"]
    logger = pipeline_state["logger"]

    frame_count = 0
    fps_calc = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue

        frame_count += 1
        t0 = time.time()
        frame_height, frame_width = frame.shape[:2]

        # Stage 1 & 2: Detection & Tracking
        if tracker.algorithm == "botsort":
            tracked = tracker.update(None, detector=detector, frame=frame)
        else:
            detections = detector.detect_supervision(frame)
            tracked = tracker.update(detections)

        # Stage 3: Physics Gate Engine with Multi-Frame Re-ID & Directional State Lock
        count_res = counter.update(tracked, frame, frame_width, frame_height)

        if logger and count_res["new_counts"]:
            for ent_id in count_res["new_counts"]:
                logger.log_crossing(frame_count, ent_id, count_res["total_count"])

        # Stage 4: Overlay Visualization
        frame = draw_physics_virtual_gate(
            frame,
            mode=counter.mode,
            target_depth_meters=counter.target_depth_meters,
            wall_x_ratio=counter.wall_x_ratio,
            focal_length_px=counter.focal_length_px,
            human_height_meters=counter.human_height_meters,
        )

        frame = draw_physics_pointers_and_telemetry(
            frame,
            telemetry=count_res["telemetry"],
            counted_entities=counter.counted_entities,
            mode=counter.mode,
            wall_x_ratio=counter.wall_x_ratio,
        )

        frame = draw_hud_physics(
            frame,
            count_res["total_count"],
            count_res["count_in"],
            count_res["count_out"],
            count_res["active_tracks"],
            fps_calc,
            counter.mode,
            counter.target_depth_meters,
            tracker.algorithm,
        )

        if count_res["new_counts"]:
            frame = draw_new_count_flash_physics(frame, count_res["new_counts"])

        fps_calc = 1.0 / max(time.time() - t0, 1e-6)

        # Update global state for API
        pipeline_state["last_result"] = {
            "total_count": count_res["total_count"],
            "count_in": count_res["count_in"],
            "count_out": count_res["count_out"],
            "active_tracks": count_res["active_tracks"],
            "fps": fps_calc,
            "width": frame_width,
            "height": frame_height,
            "mode": counter.mode,
            "telemetry": count_res["telemetry"],
        }

        # Encode JPEG
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = jpeg.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return send_file("static/index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(pipeline_state["last_result"])


@app.route("/api/update_config", methods=["POST"])
def api_update_config():
    data = request.json or {}
    counter = pipeline_state["counter"]
    if not counter:
        return jsonify({"status": "error"}), 400

    if "mode" in data:
        counter.mode = str(data["mode"]).lower()
    if "target_depth_meters" in data:
        counter.target_depth_meters = float(data["target_depth_meters"])
    if "reid_similarity_threshold" in data:
        counter.reid_similarity_threshold = float(data["reid_similarity_threshold"])

    return jsonify({"status": "ok", "mode": counter.mode, "target_depth": counter.target_depth_meters})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    if pipeline_state["counter"]:
        pipeline_state["counter"].reset()
    if pipeline_state["tracker"]:
        pipeline_state["tracker"].reset()
    return jsonify({"status": "reset_complete"})


@app.route("/output/<path:filename>")
def download_output(filename):
    return send_from_directory("output", filename)


if __name__ == "__main__":
    init_pipeline()
    console.print("\n[bold black on bright_green] WILFRED LABS WEB SERVER RUNNING [/bold black on bright_green]")
    console.print("[bold cyan]Access Dashboard at:[/bold cyan] [bold yellow]http://localhost:5000[/bold yellow]\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
