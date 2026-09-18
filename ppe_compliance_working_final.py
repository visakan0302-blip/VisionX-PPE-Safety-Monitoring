"""
=============================================================================
VISIONX - OD-04 CONSTRUCTION PPE COMPLIANCE & WORKER SAFETY MONITORING
VERSION 4.1 - WORKER-CENTRIC DYNAMIC SAFETY RISK ENGINE
=============================================================================
Original System Innovations:
1. Worker-centric PPE compliance & risk tracking
2. Persistent worker-level violation history with noise tolerance
3. Temporal violation confirmation (preventing single-frame false alarms)
4. Context-aware hazard proximity scoring (machinery & vehicle geometry)
5. Automatic safety evidence generation (annotated snapshots & CSV audit trail)
6. Explainable heuristic risk score (0-100) and human-readable causal reasons

Recovery Backup:
ppe_compliance_v4_working.py remains untouched as the recovery baseline.
=============================================================================
"""

import os
import sys
import time
import math
import cv2
import torch
from ultralytics import YOLO

# Import the modular Safety Risk Engine
from safety_engine import (
    SafetyRiskEngine,
    VIOLATION_CONFIRMATION_FRAMES,
    HISTORY_LENGTH,
    SNAPSHOT_COOLDOWN_SECONDS,
    HAZARD_PROXIMITY_FACTOR,
)

# =============================================================================
# CONFIGURATION / DEMO MODE
# =============================================================================

# Set VIDEO_SOURCE to 0 for live webcam, or path to test video file
VIDEO_SOURCE = 0
# VIDEO_SOURCE = 0  # Uncomment to use default webcam

MODEL_PATH = "models\\best.pt"
CONFIDENCE = 0.30

# GPU / CUDA acceleration: automatically utilize CUDA device 0 if supported by torch,
# otherwise fallback gracefully to CPU without crashing
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Required PPE items for full compliance
REQUIRED_PPE = [
    "Hardhat",
    "Safety Vest"
]

# Detection classes for PPE
PPE_CLASSES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Safety Vest"
]

# Detection classes for contextual site hazards
HAZARD_CLASSES = [
    "machinery",
    "vehicle"
]

# =============================================================================
# LOAD MODEL & INITIALIZE ENGINE
# =============================================================================

print("=" * 60)
print("VisionX - Construction PPE Compliance & Worker Safety Monitoring")
print(f"Loading YOLO model from: {MODEL_PATH}")
print(f"Compute device: {DEVICE} (CUDA available: {torch.cuda.is_available()})")

model = YOLO(MODEL_PATH)
names = model.names

print("Model loaded successfully.")
print("Model classes:", names)

# Initialize Safety Risk Engine
safety_engine = SafetyRiskEngine(
    history_length=HISTORY_LENGTH,
    required_violation_frames=VIOLATION_CONFIRMATION_FRAMES,
    cooldown_seconds=SNAPSHOT_COOLDOWN_SECONDS,
    proximity_factor=HAZARD_PROXIMITY_FACTOR,
    require_mask=False,
)

print(f"Safety Risk Engine initialized (Confirmation window: {VIOLATION_CONFIRMATION_FRAMES} frames).")
print("=" * 60)


# =============================================================================
# OPEN VIDEO OR WEBCAM STREAM
# =============================================================================

print(f"Opening video source: {VIDEO_SOURCE} ...")
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"ERROR: Could not open video source: {VIDEO_SOURCE}")
    if VIDEO_SOURCE == 0:
        print("Webcam device 0 not accessible. Ensure camera permissions or try video path.")
    sys.exit(1)

print()
print("VisionX PPE Compliance & Dynamic Risk System started.")
print("Tracking: ByteTrack (persist=True)")
print("Keyboard control: Press 'Q' to quit.")
print()


# =============================================================================
# GEOMETRIC ASSOCIATION HELPERS (Preserved from Working V4)
# =============================================================================

def center(box):
    """Return center (cx, cy) of bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def expanded_region(person_box, frame_width, frame_height):
    """
    Expand worker region.
    The top is expanded because a helmet may be above the Person bounding box.
    """
    x1, y1, x2, y2 = person_box
    width = x2 - x1
    height = y2 - y1

    return (
        max(0.0, x1 - width * 0.20),
        max(0.0, y1 - height * 0.40),
        min(float(frame_width), x2 + width * 0.20),
        min(float(frame_height), y2 + height * 0.15)
    )


def inside_region(box, region):
    """Check whether box center is inside region."""
    cx, cy = center(box)
    rx1, ry1, rx2, ry2 = region
    return (rx1 <= cx <= rx2 and ry1 <= cy <= ry2)


def body_position_ok(ppe_name, ppe_box, person_box):
    """Check whether PPE is in an anatomically reasonable body position."""
    _, py = center(ppe_box)
    x1, y1, x2, y2 = person_box
    height = max(y2 - y1, 1.0)
    relative_y = (py - y1) / height

    # Helmet must be near upper body / head
    if ppe_name in ["Hardhat", "NO-Hardhat"]:
        return relative_y <= 0.50

    # Mask must be near head / mid-face
    if ppe_name in ["Mask", "NO-Mask"]:
        return relative_y <= 0.65

    # Vest must be on torso / upper body
    if ppe_name in ["Safety Vest", "NO-Safety Vest"]:
        return 0.10 <= relative_y <= 1.00

    return True


def association_score(ppe_box, person_box):
    """
    Calculate normalized distance between PPE center and worker center.
    Lower score = better association.
    """
    ppe_c = center(ppe_box)
    person_c = center(person_box)
    distance_value = point_distance(ppe_c, person_c)
    person_height = max(person_box[3] - person_box[1], 1.0)
    return distance_value / person_height


# =============================================================================
# MAIN VIDEO PROCESSING LOOP
# =============================================================================

# Maximum frames to process (None for continuous/full video, or set via VISIONX_MAX_FRAMES)
MAX_FRAMES = int(os.environ.get("VISIONX_MAX_FRAMES", "0")) or None

frame_number = 0
last_evidence_banner = ""
last_evidence_time = 0.0

while True:
    if MAX_FRAMES and frame_number >= MAX_FRAMES:
        print(f"Reached configured limit of {MAX_FRAMES} frames. Exiting.")
        break

    ret, frame = cap.read()
    if not ret:
        print("End of video stream reached or frame read error.")
        break

    frame_number += 1
    frame_height, frame_width = frame.shape[:2]

    # Periodically prune inactive worker IDs to maintain efficiency
    if frame_number % 30 == 0:
        safety_engine.prune_stale_workers(frame_number)

    # =========================================================================
    # YOLO DETECTION & BYTETRACK TRACKING
    # =========================================================================
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        device=DEVICE,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    # Storage for categorized detections
    persons = []
    ppe_detections = []
    machinery_boxes = []
    vehicle_boxes = []

    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            class_name = names[class_id]
            box = tuple(boxes.xyxy[i].tolist())

            # Worker detection
            if class_name == "Person":
                track_id = int(boxes.id[i]) if boxes.id is not None else None
                persons.append({
                    "id": track_id,
                    "box": box,
                    "confidence": confidence
                })

            # PPE detection
            elif class_name in PPE_CLASSES:
                ppe_detections.append({
                    "name": class_name,
                    "box": box,
                    "confidence": confidence
                })

            # Hazard detections
            elif class_name == "machinery":
                machinery_boxes.append(box)

            elif class_name == "vehicle":
                vehicle_boxes.append(box)

    # Draw hazard bounding boxes for clear visual context
    for mbox in machinery_boxes:
        mx1, my1, mx2, my2 = map(int, mbox)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 140, 255), 2)
        cv2.putText(
            frame, "MACHINERY [HAZARD]", (mx1, max(my1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA
        )

    for vbox in vehicle_boxes:
        vx1, vy1, vx2, vy2 = map(int, vbox)
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 180), 2)
        cv2.putText(
            frame, "VEHICLE [HAZARD]", (vx1, max(vy1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 180), 1, cv2.LINE_AA
        )

    # =========================================================================
    # WORKER ASSESSMENT & COMPLIANCE PIPELINE
    # =========================================================================
    worker_assessments = []
    compliant_count = 0
    confirmed_violation_count = 0
    active_violation_count = 0
    check_count = 0

    for worker_index, person in enumerate(persons):
        person_box = person["box"]
        worker_id = person["id"]

        # Fallback ID if tracker has not yet assigned one
        if worker_id is None:
            worker_id = worker_index + 1

        # 1. Expanded worker association region
        association_area = expanded_region(person_box, frame_width, frame_height)

        # 2. Find PPE detections belonging to this worker
        associated_ppe = []
        for ppe in ppe_detections:
            ppe_box = ppe["box"]

            if not inside_region(ppe_box, association_area):
                continue

            if not body_position_ok(ppe["name"], ppe_box, person_box):
                continue

            score = association_score(ppe_box, person_box)
            associated_ppe.append({
                "name": ppe["name"],
                "box": ppe_box,
                "confidence": ppe["confidence"],
                "score": score
            })

        # 3. Retain best detection for each PPE category
        best_ppe = {}
        for ppe in associated_ppe:
            name = ppe["name"]
            if name not in best_ppe or ppe["confidence"] > best_ppe[name]["confidence"]:
                best_ppe[name] = ppe

        detected_names = set(best_ppe.keys())

        # 4. Determine raw PPE statuses
        # Hardhat
        if "NO-Hardhat" in detected_names:
            hardhat_status = "MISSING"
        elif "Hardhat" in detected_names:
            hardhat_status = "YES"
        else:
            hardhat_status = "UNKNOWN"

        # Safety Vest
        if "NO-Safety Vest" in detected_names:
            vest_status = "MISSING"
        elif "Safety Vest" in detected_names:
            vest_status = "YES"
        else:
            vest_status = "UNKNOWN"

        # Mask
        if "NO-Mask" in detected_names:
            mask_status = "MISSING"
        elif "Mask" in detected_names:
            mask_status = "YES"
        else:
            mask_status = "UNKNOWN"

        # 5. Evaluate through the Safety Risk Engine
        assessment = safety_engine.evaluate_worker(
            worker_id=worker_id,
            hardhat_status=hardhat_status,
            vest_status=vest_status,
            mask_status=mask_status,
            worker_box=person_box,
            machinery_boxes=machinery_boxes,
            vehicle_boxes=vehicle_boxes,
            frame_idx=frame_number
        )

        worker_assessments.append((person_box, assessment))

        # Count state statistics
        if assessment.compliance_state == "CONFIRMED VIOLATION":
            confirmed_violation_count += 1
        elif assessment.compliance_state == "ACTIVE VIOLATION":
            active_violation_count += 1
        elif assessment.compliance_state == "CHECK":
            check_count += 1
        else:
            compliant_count += 1

    # =========================================================================
    # DRAW WORKER HUD & VISUAL ALERTS
    # =========================================================================
    for person_box, assessment in worker_assessments:
        x1, y1, x2, y2 = map(int, person_box)
        state = assessment.compliance_state

        # Color coding by safety state:
        # Green: Compliant
        # Yellow: Check / Uncertain
        # Orange: Active Violation (temporal unconfirmed)
        # Red: Confirmed Violation / High Risk
        if state == "CONFIRMED VIOLATION":
            box_color = (30, 30, 230)      # Vivid Red
            line_thickness = 3
        elif state == "ACTIVE VIOLATION":
            box_color = (0, 140, 255)      # Amber / Orange
            line_thickness = 2
        elif state == "CHECK":
            box_color = (0, 215, 255)      # Yellow
            line_thickness = 2
        else:
            box_color = (40, 200, 40)      # Clean Green
            line_thickness = 2

        # Draw worker bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, line_thickness)

        # Worker information card above bounding box
        label_y = max(y1 - 85, 25)

        # Semi-transparent background badge for high legibility
        card_w = max(210, x2 - x1 + 20)
        card_h = 80
        badge_x1 = max(0, x1)
        badge_y1 = max(0, label_y - 18)
        badge_x2 = min(frame_width, badge_x1 + card_w)
        badge_y2 = min(frame_height, badge_y1 + card_h)

        sub_overlay = frame[badge_y1:badge_y2, badge_x1:badge_x2].copy()
        cv2.rectangle(sub_overlay, (0, 0), (badge_x2 - badge_x1, badge_y2 - badge_y1), (20, 20, 20), -1)
        cv2.addWeighted(sub_overlay, 0.70, frame[badge_y1:badge_y2, badge_x1:badge_x2], 0.30, 0, frame[badge_y1:badge_y2, badge_x1:badge_x2])
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, 1)

        # Line 1: Worker ID & Risk Level Badge
        risk_tag = f"RISK: {assessment.risk_score}/100 [{assessment.risk_level}]"
        cv2.putText(
            frame, f"Worker #{assessment.worker_id} | {risk_tag}",
            (badge_x1 + 6, badge_y1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Line 2: PPE Statuses
        h_col = (40, 220, 40) if assessment.hardhat_status == "YES" else ((30, 30, 230) if assessment.hardhat_status == "MISSING" else (0, 215, 255))
        v_col = (40, 220, 40) if assessment.vest_status == "YES" else ((30, 30, 230) if assessment.vest_status == "MISSING" else (0, 215, 255))
        m_col = (40, 220, 40) if assessment.mask_status == "YES" else ((30, 30, 230) if assessment.mask_status == "MISSING" else (0, 215, 255))

        cv2.putText(
            frame, f"Hardhat:{assessment.hardhat_status} | Vest:{assessment.vest_status} | Mask:{assessment.mask_status}",
            (badge_x1 + 6, badge_y1 + 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA
        )

        # Line 3: Hazards Proximity
        mach_str = "NEAR" if assessment.machinery_near else "FAR"
        veh_str = "NEAR" if assessment.vehicle_near else "FAR"
        haz_col = (30, 30, 230) if (assessment.machinery_near or assessment.vehicle_near) else (200, 200, 200)
        cv2.putText(
            frame, f"Machinery:{mach_str} | Vehicle:{veh_str}",
            (badge_x1 + 6, badge_y1 + 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, haz_col, 1, cv2.LINE_AA
        )

        # Line 4: State & Causal Reasons
        reasons_text = " | ".join(assessment.risk_reasons) if assessment.risk_reasons else "All Safety Gear In Place"
        if len(reasons_text) > 34:
            reasons_text = reasons_text[:31] + "..."
        cv2.putText(
            frame, f"{state}: {reasons_text}",
            (badge_x1 + 6, badge_y1 + 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA
        )

        # High Risk Banner directly below worker box if high risk
        if assessment.risk_level == "HIGH":
            alert_text = f"HIGH RISK: {assessment.risk_reasons[0] if assessment.risk_reasons else 'Violation'}"
            cv2.putText(
                frame, alert_text,
                (x1, min(y2 + 20, frame_height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (30, 30, 230), 2, cv2.LINE_AA
            )

    # =========================================================================
    # EVIDENCE GENERATION & LOGGING (TRIGGERED ON CONFIRMED VIOLATION)
    # =========================================================================
    for _, assessment in worker_assessments:
        if assessment.compliance_state == "CONFIRMED VIOLATION":
            # Pass annotated frame so the recorded snapshot shows full context
            saved_path = safety_engine.record_evidence_if_needed(assessment, frame)
            if saved_path:
                last_evidence_banner = f"EVIDENCE SAVED: {os.path.basename(saved_path)}"
                last_evidence_time = time.time()
                print(f"[ALERT] Frame {frame_number}: Confirmed Violation for Worker #{assessment.worker_id} (Score: {assessment.risk_score}/100) -> Saved {saved_path}")

    # =========================================================================
    # UPGRADED OPENCV DASHBOARD & TELEMETRY PANEL
    # =========================================================================
    total_workers = len(persons)
    compliance_rate = (compliant_count / total_workers * 100.0) if total_workers > 0 else 0.0

    # Dashboard container dimensions
    dash_x1, dash_y1 = 10, 10
    dash_w = min(470, frame_width - 20)
    dash_h = 160
    dash_x2 = dash_x1 + dash_w
    dash_y2 = dash_y1 + dash_h

    # Semi-transparent dark background
    dash_overlay = frame[dash_y1:dash_y2, dash_x1:dash_x2].copy()
    cv2.rectangle(dash_overlay, (0, 0), (dash_w, dash_h), (12, 12, 18), -1)
    cv2.addWeighted(dash_overlay, 0.85, frame[dash_y1:dash_y2, dash_x1:dash_x2], 0.15, 0, frame[dash_y1:dash_y2, dash_x1:dash_x2])
    cv2.rectangle(frame, (dash_x1, dash_y1), (dash_x2, dash_y2), (60, 60, 70), 1)

    # Header
    cv2.putText(
        frame, "VISIONX - PPE MONITOR",
        (dash_x1 + 14, dash_y1 + 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )

    # Device & Tracker Status Pill
    dev_str = f"DEV:{DEVICE}"
    cv2.putText(
        frame, f"TRACKING: ON (ByteTrack) | {dev_str}",
        (dash_x1 + 230, dash_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 180), 1, cv2.LINE_AA
    )

    # Divider line
    cv2.line(frame, (dash_x1 + 10, dash_y1 + 35), (dash_x2 - 10, dash_y1 + 35), (70, 70, 80), 1)

    # Site Overview Metrics (Row 1)
    cv2.putText(
        frame, f"WORKERS: {total_workers}",
        (dash_x1 + 14, dash_y1 + 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA
    )

    cv2.putText(
        frame, f"COMPLIANT: {compliant_count}",
        (dash_x1 + 130, dash_y1 + 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (40, 220, 40), 1, cv2.LINE_AA
    )

    cv2.putText(
        frame, f"VIOLATIONS: {confirmed_violation_count + active_violation_count}",
        (dash_x1 + 255, dash_y1 + 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (30, 30, 230), 1, cv2.LINE_AA
    )

    cv2.putText(
        frame, f"CHECK: {check_count}",
        (dash_x1 + 385, dash_y1 + 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1, cv2.LINE_AA
    )

    # Site Overview Metrics (Row 2)
    cv2.putText(
        frame, f"Compliance Rate: {compliance_rate:.1f}%",
        (dash_x1 + 14, dash_y1 + 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA
    )

    cv2.putText(
        frame, f"Frame: {frame_number}",
        (dash_x1 + 255, dash_y1 + 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA
    )

    # Top Risk Worker Context Card
    if worker_assessments:
        # Sort by risk score descending
        sorted_workers = sorted(worker_assessments, key=lambda wa: wa[1].risk_score, reverse=True)
        top_box, top_ass = sorted_workers[0]

        top_color = (30, 30, 230) if top_ass.risk_level == "HIGH" else ((0, 140, 255) if top_ass.risk_level == "MEDIUM" else (40, 220, 40))
        top_reasons = " | ".join(top_ass.risk_reasons) if top_ass.risk_reasons else "Compliant"
        if len(top_reasons) > 38:
            top_reasons = top_reasons[:35] + "..."

        cv2.putText(
            frame, f"TOP RISK: Worker #{top_ass.worker_id} -> {top_ass.risk_score}/100 [{top_ass.risk_level}] ({top_ass.compliance_state})",
            (dash_x1 + 14, dash_y1 + 104),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, top_color, 1, cv2.LINE_AA
        )

        cv2.putText(
            frame, f"Context: H:{top_ass.hardhat_status} V:{top_ass.vest_status} | Mach:{'NEAR' if top_ass.machinery_near else 'FAR'} Veh:{'NEAR' if top_ass.vehicle_near else 'FAR'}",
            (dash_x1 + 14, dash_y1 + 122),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA
        )

        cv2.putText(
            frame, f"Reasons: {top_reasons}",
            (dash_x1 + 14, dash_y1 + 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, top_color, 1, cv2.LINE_AA
        )
    else:
        cv2.putText(
            frame, "Scanning site for workers...",
            (dash_x1 + 14, dash_y1 + 115),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA
        )

    # Transient Evidence Notice Banner
    if last_evidence_banner and (time.time() - last_evidence_time < 4.0):
        banner_w = min(420, frame_width - 20)
        cv2.rectangle(frame, (10, frame_height - 35), (10 + banner_w, frame_height - 8), (0, 100, 0), -1)
        cv2.putText(
            frame, f"[SAVED] {last_evidence_banner}",
            (16, frame_height - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
        )

    # =========================================================================
    # DISPLAY
    # =========================================================================
    cv2.imshow("VisionX - Construction PPE Compliance V4.1", frame)

    # Keyboard input: Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("User requested exit (Q pressed).")
        break


# =============================================================================
# CLEANUP
# =============================================================================
cap.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("VisionX PPE Compliance System stopped.")
print(f"Total frames processed: {frame_number}")
print(f"Safety events logged to: {safety_engine.csv_file}")
print(f"Violation snapshots stored in: {safety_engine.snapshots_dir}")
print("=" * 60)