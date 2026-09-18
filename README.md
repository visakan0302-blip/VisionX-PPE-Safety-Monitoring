# VisionX — Construction PPE Detection & Worker Safety Monitoring

![VisionX Banner](assets/videoconstruc2.gif)

> *"4,764 workers died on the job in 2020 (3.4 per 100,000 full-time equivalent workers). Workers in transportation and material moving occupations and construction and extraction occupations accounted for nearly half of all fatal occupational injuries (47.4 percent), representing 1,282 and 976 workplace deaths, respectively."*  
> — **Occupational Safety and Health Administration (OSHA, US Department of Labor)**

---

## 1. Problem Statement

Standard computer vision approaches in industrial safety perform isolated, frame-by-frame object detection (e.g., detecting a hardhat or a vest as independent bounding boxes). However, raw frame-level detection alone suffers from fundamental limitations in real-world construction environments:
1. **Lack of Worker Association**: It identifies PPE items without tracking which worker wears what.
2. **False Alarm Fatigue**: Transient occlusions or single-frame detection flickers cause flickering violations.
3. **Absence of Contextual Risk**: A worker missing a hardhat while standing directly next to heavy moving machinery faces a vastly higher imminent hazard than one in an empty staging area.
4. **No Audit Trail**: Traditional detectors do not automatically generate forensically valid incident snapshots or persistent compliance event records.

---

## 2. Proposed Solution: Worker-Centric Dynamic Safety Risk Engine

**VisionX** transitions construction monitoring from static object detection to an intelligent, worker-centric dynamic safety risk intelligence pipeline:

```
Camera / Video Stream
       ↓
YOLO PPE Detection (models/best.pt: Person, PPE, Machinery, Vehicle)
       ↓
ByteTrack Worker Tracking (persistent worker IDs)
       ↓
Worker ↔ PPE Association (expanded region, anatomical positioning, normalized distance)
       ↓
Hazard Proximity Analysis (Machinery & Vehicle geometric proximity radius)
       ↓
Temporal Violation Validation (history queue, 5-frame confirmation, noise tolerance)
       ↓
Dynamic Worker Safety Risk Engine (0–100 heuristic explainable score & reasons)
       ↓
Upgraded OpenCV Dashboard & Visual Alerts (color-coded bounding boxes, risk badges, top risk panel)
       ↓
Automatic Safety Evidence Generation (annotated snapshot + CSV audit trail with cooldown)
```

---

## 3. Team System-Level Contributions

> **Important Model Attribution:**  
> The weights in `models/best.pt` represent a pretrained/existing YOLOv8 model trained on the Roboflow/Kaggle Construction Site Safety dataset. We do **not** claim global novelty on the underlying YOLO model or base tracking algorithms.

### Our Innovation & Original Contributions:
Our contribution is the **end-to-end system-level engineering and safety intelligence architecture**:
1. **Worker-Centric PPE Association**: Anatomically constrained bounding-box association combining expanded head/body regions, vertical position gating, and normalized center-distance scoring.
2. **Temporal Violation Validation**: Multi-frame sliding window history buffer ($N=10$, threshold=5) with noise tolerance to eliminate false positives from temporary detection dropouts.
3. **Context-Aware Hazard Proximity Analysis**: Scale-adaptive Euclidean bounding-box distance calculation between workers and detected site hazards (`machinery`, `vehicle`).
4. **Dynamic Explainable Risk Engine**: Real-time 0–100 heuristic risk scoring with transparent causal reasoning:
   - Missing Hardhat: `+35`
   - Missing Safety Vest: `+25`
   - Missing required Mask: `+15`
   - Machinery Near: `+25`
   - Vehicle Near: `+20`
   - Persistent Violation: `+10`
   - Risk Levels: **LOW** (0–30), **MEDIUM** (31–60), **HIGH** (61–100).
5. **Automated Evidence Generation & Audit Trail**: Event-driven snapshot capture (`violation_snapshots/worker_<ID>_<timestamp>.jpg`) with 10-second per-worker cooldown and escalation override, coupled with structured tabular logging (`violation_logs/safety_events.csv`).
6. **Live Telemetry OpenCV Dashboard**: Real-time Heads-Up Display (HUD) presenting site-level compliance metrics, priority high-risk worker cards, and state-coded visual alerts.

---

## 4. Technology Stack

- **Core Programming**: Python 3.10+
- **Computer Vision**: OpenCV (`cv2`)
- **Deep Learning Framework**: Ultralytics YOLOv8 & PyTorch
- **Worker Tracking**: ByteTrack (`bytetrack.yaml`)
- **Hardware Acceleration**: Automatic CUDA / GPU acceleration (`device=0`) with seamless CPU fallback
- **Data & Logging**: NumPy, Python CSV, File System Snapshots

---

## 5. Model Classes & Detection Capabilities

The underlying detector (`models/best.pt`) detects 10 distinct classes:
```python
{
    0: 'Hardhat',
    1: 'Mask',
    2: 'NO-Hardhat',
    3: 'NO-Mask',
    4: 'NO-Safety Vest',
    5: 'Person',
    6: 'Safety Cone',
    7: 'Safety Vest',
    8: 'machinery',
    9: 'vehicle'
}
```

---

## 6. Directory Structure

```
C:\VisionX\PPE-Base\
│
├── ppe_compliance.py              # Main OpenCV live monitor entry point
├── upload_app.py                  # Browser-based Video Upload & Analysis application (FastAPI)
├── video_processor.py             # Reusable core detection, tracking & risk pipeline
├── safety_engine.py               # Modular Safety Risk Engine & Evidence Recorder
├── ppe_compliance_v4_working.py   # Immutable baseline recovery backup
│
├── models\
│   ├── best.pt                    # Pretrained YOLOv8 PPE detection weights
│   └── yolov8n.pt                 # Base YOLOv8 nano model
│
├── source_files\                  # Test videos and images
│   ├── hardhat.mp4                # Video test stream (640x360, 30 FPS)
│   ├── construction-safety.jpg    # Multi-hazard test image (Machinery + Vehicle)
│   └── ...
│
├── processed\                     # Generated annotated videos from web analysis
├── violation_snapshots\           # Automatically captured evidence snapshots
├── violation_logs\                # Persistent safety audit trail (safety_events.csv)
└── assets\                        # Documentation media and visualizations
```

---

## 7. How to Run

VisionX offers two distinct demo modes sharing the exact same underlying detection and risk engine:

### Mode A: OpenCV Live Monitor (Webcam & Video)
Run directly in the console:
```bash
cd C:\VisionX\PPE-Base
python ppe_compliance.py
```
- **Video Source**: At the top of [ppe_compliance.py](ppe_compliance.py), set `VIDEO_SOURCE = "source_files\\hardhat.mp4"` or `VIDEO_SOURCE = 0` for live webcam.
- **Controls**: Press **Q** in the video window to stop.

### Mode B: Browser-Based Video Upload & Analysis (Web App)
Start the local FastAPI server:
```bash
cd C:\VisionX\PPE-Base
python upload_app.py
```
- **URL**: Open your browser and navigate to:
  `http://127.0.0.1:8000`
- **Features**:
  - Drag & drop any construction video (`.mp4`, `.avi`, `.mov`, `.mkv`).
  - Click **ANALYZE VIDEO** or use the quick sample video link.
  - View real-time annotated video playback, site summary metrics, highest-risk worker breakdown, and violation snapshot evidence gallery.
  - Download annotated video and persistent CSV safety events audit log.

---

## 8. Audit Trail & CSV Evidence Log

When a worker triggers a **CONFIRMED VIOLATION**, a structured audit row is recorded in `violation_logs/safety_events.csv`:

| Timestamp | Worker_ID | Hardhat_Status | Vest_Status | Mask_Status | Machinery_Near | Vehicle_Near | Persistent_Violation | Risk_Score | Risk_Level | Reasons | Snapshot_Path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-18 11:12:24 | Worker #15 | MISSING | MISSING | MISSING | FAR | FAR | YES | 70/100 | HIGH | Missing Hardhat \| Missing Safety Vest \| Persistent Violation | violation_snapshots\worker_15_20260918_111224.jpg |

A corresponding full-resolution annotated snapshot is saved to `violation_snapshots/` preserving worker status tags, bounding boxes, and risk levels for safety officers and regulatory compliance.
