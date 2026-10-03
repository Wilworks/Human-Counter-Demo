# Real-Time Human Counting Feasibility Demo via YOLOv8, ByteTrack & 3-Layer Anti-Double-Count Logic

**Author:** Wilfred Ayine Asumboya  
**Target Application:** Lab Feasibility Demo for Cattle-Counter Architecture Validation  

This project provides a zero-shot, real-time proof-of-concept pipeline using standard laptop webcam input (`source = 0`) to detect humans (`COCO Class 0`), track them via **ByteTrack**, and count them as they cross a virtual directional gate line with **3-Layer Anti-Double-Count Protection**.

---

## 🚀 Quick Start Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Diagnostic Environment Check
```bash
python run_pipeline.py --check
```

### 3. Launch Live Laptop Webcam Human Counter
```bash
python run_pipeline.py --source 0
```

---

## ⚙ System Controls During Stream
- **`q`**: Stop live camera stream and display execution summary card.
- **`r`**: Reset counter memory and clear locked IDs in real time.

---

## 🛠 Project Schema & Architecture
- `config/settings.yaml`: Configurable thresholds, line Y position, and video source settings.
- `src/detection/detector.py`: Zero-shot YOLOv8 person detector wrapper (`yolov8n.pt`).
- `src/tracking/tracker.py`: ByteTrack persistent multi-person tracker.
- `src/counting/counter.py`: 3-Layer anti-double-count directional gate engine.
- `src/utils/visualization.py`: Neon virtual line, HUD telemetry card, and new count flash overlay.
- `src/utils/logger.py`: Event timestamped CSV logging (`output/human_crossing_log.csv`).
