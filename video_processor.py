"""
=============================================================================
VisionX - Video Processing Pipeline (video_processor.py)
Reusable Core Pipeline for Both OpenCV Live Application and Web Video Analysis

Features:
- Common YOLO detection (models/best.pt)
- ByteTrack persistent worker tracking
- Anatomical PPE association (Hardhat, Vest, Mask)
- SafetyRiskEngine evaluation (explainable 0-100 risk score, temporal validation,
  hazard proximity)
- Polished OpenCV dashboard overlay rendering (site overview, top risk worker,
  compact worker badges, bottom alert bar)
- Automated snapshot evidence and CSV event logging
- Single frame processing & Full video processing with statistics aggregation
=============================================================================
"""

import os
import sys
import time
import math
import cv2
import torch
from ultralytics import YOLO

from safety_engine import (
    SafetyRiskEngine,
    VIOLATION_CONFIRMATION_FRAMES,
    HISTORY_LENGTH,
    SNAPSHOT_COOLDOWN_SECONDS,
    HAZARD_PROXIMITY_FACTOR,
)

# =============================================================================
# CONSTANTS & MODEL CONFIGURATION
# =============================================================================

MODEL_PATH = "models\\best.pt"
CONFIDENCE = 0.30

# Automatically select CUDA device 0 if supported by torch, else CPU fallback
DEVICE = 0 if torch.cuda.is_available() else "cpu"

REQUIRED_PPE = [
    "Hardhat",
    "Safety Vest"
]

PPE_CLASSES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Safety Vest"
]

HAZARD_CLASSES = [
    "machinery",
    "vehicle"
]


# =============================================================================
# GEOMETRIC ASSOCIATION HELPERS
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
# REUSABLE VISIONX PROCESSOR CLASS
# =============================================================================

class VisionXProcessor:
    """
    Core reusable pipeline for VisionX PPE compliance and safety risk monitoring.
    Used by both ppe_compliance.py (OpenCV live stream) and upload_app.py (Web).
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        device=DEVICE,
        confidence=CONFIDENCE,
        safety_engine=None,
    ):
        self.model_path = model_path
        self.device = device
        self.confidence = confidence

        print(f"[VisionXProcessor] Loading model: {self.model_path} (Device: {self.device})")
        self.model = YOLO(self.model_path)
        self.names = self.model.names

        if safety_engine is not None:
            self.safety_engine = safety_engine
        else:
            self.safety_engine = SafetyRiskEngine(
                history_length=HISTORY_LENGTH,
                required_violation_frames=VIOLATION_CONFIRMATION_FRAMES,
                cooldown_seconds=SNAPSHOT_COOLDOWN_SECONDS,
                proximity_factor=HAZARD_PROXIMITY_FACTOR,
                require_mask=False,
            )

        self.last_evidence_banner = ""
        self.last_evidence_time = 0.0

    def reset_tracking(self):
        """Reset internal tracking and safety engine histories for a new video session."""
        self.safety_engine.worker_histories.clear()
        self.safety_engine.worker_last_seen.clear()
        self.safety_engine.worker_snapshot_records.clear()
        self.last_evidence_banner = ""
        self.last_evidence_time = 0.0

    def process_frame(self, frame, frame_idx, record_evidence=True):
        """
        Processes a single video frame:
        1. YOLO + ByteTrack detection & tracking
        2. Worker-PPE anatomical association
        3. SafetyRiskEngine risk scoring & temporal confirmation
        4. Clean OpenCV HUD dashboard and bounding box rendering
        5. Evidence snapshot & CSV event logging

        Returns:
            annotated_frame: np.ndarray
            summary: dict with frame-level telemetry metrics
        """
        frame_height, frame_width = frame.shape[:2]
        annotated_frame = frame.copy()

        # Prune inactive worker history periodically
        if frame_idx % 30 == 0:
            self.safety_engine.prune_stale_workers(frame_idx)

        # ---------------------------------------------------------------------
        # 1. YOLO DETECTION & BYTETRACK TRACKING
        # ---------------------------------------------------------------------
        results = self.model.track(
            annotated_frame,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            conf=self.confidence,
            verbose=False,
        )

        result = results[0]
        persons = []
        ppe_detections = []
        machinery_boxes = []
        vehicle_boxes = []

        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                class_name = self.names[class_id]
                box = tuple(boxes.xyxy[i].tolist())

                if class_name == "Person":
                    track_id = int(boxes.id[i]) if boxes.id is not None else None
                    persons.append({
                        "id": track_id,
                        "box": box,
                        "confidence": conf
                    })
                elif class_name in PPE_CLASSES:
                    ppe_detections.append({
                        "name": class_name,
                        "box": box,
                        "confidence": conf
                    })
                elif class_name == "machinery":
                    machinery_boxes.append(box)
                elif class_name == "vehicle":
                    vehicle_boxes.append(box)

        # Draw hazard bounding boxes
        for mbox in machinery_boxes:
            mx1, my1, mx2, my2 = map(int, mbox)
            cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), (0, 140, 255), 2)
            cv2.putText(
                annotated_frame, "MACHINERY [HAZARD]", (mx1, max(my1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 140, 255), 1, cv2.LINE_AA
            )

        for vbox in vehicle_boxes:
            vx1, vy1, vx2, vy2 = map(int, vbox)
            cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (255, 0, 180), 2)
            cv2.putText(
                annotated_frame, "VEHICLE [HAZARD]", (vx1, max(vy1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 180), 1, cv2.LINE_AA
            )

        # ---------------------------------------------------------------------
        # 2. WORKER-PPE ASSOCIATION & SAFETY RISK EVALUATION
        # ---------------------------------------------------------------------
        worker_assessments = []
        compliant_count = 0
        confirmed_violation_count = 0
        active_violation_count = 0
        check_count = 0

        for worker_index, person in enumerate(persons):
            person_box = person["box"]
            worker_id = person["id"]
            if worker_id is None:
                worker_id = worker_index + 1

            assoc_area = expanded_region(person_box, frame_width, frame_height)

            associated_ppe = []
            for ppe in ppe_detections:
                pbox = ppe["box"]
                if not inside_region(pbox, assoc_area):
                    continue
                if not body_position_ok(ppe["name"], pbox, person_box):
                    continue

                sc = association_score(pbox, person_box)
                associated_ppe.append({
                    "name": ppe["name"],
                    "box": pbox,
                    "confidence": ppe["confidence"],
                    "score": sc
                })

            best_ppe = {}
            for ppe in associated_ppe:
                pname = ppe["name"]
                if pname not in best_ppe or ppe["confidence"] > best_ppe[pname]["confidence"]:
                    best_ppe[pname] = ppe

            detected_names = set(best_ppe.keys())

            # Hardhat status
            if "NO-Hardhat" in detected_names:
                hardhat_status = "MISSING"
            elif "Hardhat" in detected_names:
                hardhat_status = "YES"
            else:
                hardhat_status = "UNKNOWN"

            # Safety Vest status
            if "NO-Safety Vest" in detected_names:
                vest_status = "MISSING"
            elif "Safety Vest" in detected_names:
                vest_status = "YES"
            else:
                vest_status = "UNKNOWN"

            # Mask status
            if "NO-Mask" in detected_names:
                mask_status = "MISSING"
            elif "Mask" in detected_names:
                mask_status = "YES"
            else:
                mask_status = "UNKNOWN"

            assessment = self.safety_engine.evaluate_worker(
                worker_id=worker_id,
                hardhat_status=hardhat_status,
                vest_status=vest_status,
                mask_status=mask_status,
                worker_box=person_box,
                machinery_boxes=machinery_boxes,
                vehicle_boxes=vehicle_boxes,
                frame_idx=frame_idx,
            )

            worker_assessments.append((person_box, assessment))

            if assessment.compliance_state == "CONFIRMED VIOLATION":
                confirmed_violation_count += 1
            elif assessment.compliance_state == "ACTIVE VIOLATION":
                active_violation_count += 1
            elif assessment.compliance_state == "CHECK":
                check_count += 1
            else:
                compliant_count += 1

        total_workers = len(persons)
        compliance_rate = (compliant_count / total_workers * 100.0) if total_workers > 0 else 0.0

        # ---------------------------------------------------------------------
        # 3. POLISHED OPENCV HUD DASHBOARD RENDERING
        # ---------------------------------------------------------------------
        is_hd = frame_width >= 1000
        font_title = 0.65 if is_hd else 0.46
        font_body = 0.50 if is_hd else 0.36
        font_small = 0.42 if is_hd else 0.32
        hdr_h = 42 if is_hd else 28
        panel_w = 310 if is_hd else 205
        panel_h = 100 if is_hd else 68
        bot_h = 34 if is_hd else 22

        # Draw worker bounding boxes and compact badges
        for person_box, assessment in worker_assessments:
            x1, y1, x2, y2 = map(int, person_box)
            state = assessment.compliance_state

            if state == "CONFIRMED VIOLATION" or assessment.risk_level == "HIGH":
                box_color = (30, 30, 230)
                line_thickness = 2
            elif state == "ACTIVE VIOLATION":
                box_color = (0, 140, 255)
                line_thickness = 2
            elif state == "CHECK":
                box_color = (0, 215, 255)
                line_thickness = 2
            else:
                box_color = (40, 200, 40)
                line_thickness = 2

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, line_thickness)

            line1 = f"Worker #{assessment.worker_id} | Risk: {assessment.risk_score}/100"
            line2 = f"Status: {assessment.compliance_state}"

            (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, font_small, 1)
            (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, font_small, 1)
            lbl_w = max(tw1, tw2) + 10
            lbl_h = th1 + th2 + 10

            p1_x2_bound = panel_w + 14
            p2_x1_bound = frame_width - panel_w - 14
            panel_bottom = hdr_h + panel_h + 8

            lbl_x1 = max(4, min(x1, frame_width - lbl_w - 4))
            lbl_x2 = lbl_x1 + lbl_w

            collides_header = (y1 - lbl_h - 2 <= hdr_h)
            collides_left_panel = (lbl_x1 < p1_x2_bound) and (y1 - lbl_h - 2 < panel_bottom)
            collides_right_panel = (lbl_x2 > p2_x1_bound) and (y1 - lbl_h - 2 < panel_bottom)

            if collides_header or collides_left_panel or collides_right_panel:
                if lbl_x1 < p1_x2_bound or lbl_x2 > p2_x1_bound:
                    lbl_y1 = max(y1 + 4, panel_bottom + 4)
                else:
                    lbl_y1 = max(y1 + 4, hdr_h + 4)
                if lbl_y1 + lbl_h > frame_height - bot_h - 4:
                    lbl_y1 = max(4, frame_height - bot_h - lbl_h - 4)
            else:
                lbl_y1 = y1 - lbl_h - 2

            lbl_y1 = max(0, min(lbl_y1, frame_height - lbl_h - 2))
            lbl_y2 = min(frame_height, lbl_y1 + lbl_h)
            lbl_x1 = max(0, min(lbl_x1, frame_width - lbl_w - 2))
            lbl_x2 = min(frame_width, lbl_x1 + lbl_w)

            if lbl_y2 > lbl_y1 and lbl_x2 > lbl_x1:
                badge_ov = annotated_frame[lbl_y1:lbl_y2, lbl_x1:lbl_x2].copy()
                cv2.rectangle(badge_ov, (0, 0), (lbl_x2 - lbl_x1, lbl_y2 - lbl_y1), (14, 14, 18), -1)
                cv2.addWeighted(badge_ov, 0.82, annotated_frame[lbl_y1:lbl_y2, lbl_x1:lbl_x2], 0.18, 0, annotated_frame[lbl_y1:lbl_y2, lbl_x1:lbl_x2])
                cv2.rectangle(annotated_frame, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), box_color, 1)

            if state == "CONFIRMED VIOLATION" or assessment.risk_level == "HIGH":
                status_text_color = (80, 80, 255)
            elif state == "ACTIVE VIOLATION":
                status_text_color = (0, 165, 255)
            elif state == "CHECK":
                status_text_color = (0, 230, 255)
            else:
                status_text_color = (100, 240, 100)

            cv2.putText(annotated_frame, line1, (lbl_x1 + 5, lbl_y1 + th1 + 2), cv2.FONT_HERSHEY_SIMPLEX, font_small, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, line2, (lbl_x1 + 5, lbl_y1 + th1 + th2 + 6), cv2.FONT_HERSHEY_SIMPLEX, font_small, status_text_color, 1, cv2.LINE_AA)

        # Top Header Bar
        hdr_ov = annotated_frame[0:hdr_h, 0:frame_width].copy()
        cv2.rectangle(hdr_ov, (0, 0), (frame_width, hdr_h), (14, 14, 18), -1)
        cv2.addWeighted(hdr_ov, 0.85, annotated_frame[0:hdr_h, 0:frame_width], 0.15, 0, annotated_frame[0:hdr_h, 0:frame_width])
        cv2.line(annotated_frame, (0, hdr_h), (frame_width, hdr_h), (65, 65, 75), 1)

        cv2.putText(
            annotated_frame, "VISIONX - WORKER SAFETY MONITOR",
            (10, int(hdr_h * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX, font_title, (255, 255, 255), 1, cv2.LINE_AA
        )

        live_tag = "[ YOLO + BYTETRACK | LIVE ]"
        (tw_live, _), _ = cv2.getTextSize(live_tag, cv2.FONT_HERSHEY_SIMPLEX, font_small, 1)
        cv2.putText(
            annotated_frame, live_tag,
            (frame_width - tw_live - 10, int(hdr_h * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX, font_small, (100, 230, 100), 1, cv2.LINE_AA
        )

        # Compact Top-Left Site Statistics
        tot_violations = confirmed_violation_count + active_violation_count
        viol_col = (40, 40, 230) if tot_violations > 0 else (180, 220, 180)

        p1_x1, p1_y1 = 8, hdr_h + 5
        p1_x2, p1_y2 = p1_x1 + panel_w, p1_y1 + panel_h
        ov1 = annotated_frame[p1_y1:p1_y2, p1_x1:p1_x2].copy()
        cv2.rectangle(ov1, (0, 0), (panel_w, panel_h), (16, 16, 22), -1)
        cv2.addWeighted(ov1, 0.82, annotated_frame[p1_y1:p1_y2, p1_x1:p1_x2], 0.18, 0, annotated_frame[p1_y1:p1_y2, p1_x1:p1_x2])
        cv2.rectangle(annotated_frame, (p1_x1, p1_y1), (p1_x2, p1_y2), (65, 65, 80), 1)

        cv2.putText(annotated_frame, "SITE STATISTICS", (p1_x1 + 6, p1_y1 + int(panel_h * 0.22)), cv2.FONT_HERSHEY_SIMPLEX, font_body, (220, 220, 240), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Workers: {total_workers} | Rate: {compliance_rate:.0f}%", (p1_x1 + 6, p1_y1 + int(panel_h * 0.46)), cv2.FONT_HERSHEY_SIMPLEX, font_small, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Compliant: {compliant_count} | Check: {check_count}", (p1_x1 + 6, p1_y1 + int(panel_h * 0.70)), cv2.FONT_HERSHEY_SIMPLEX, font_small, (180, 220, 180), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Violations: {tot_violations}", (p1_x1 + 6, p1_y1 + int(panel_h * 0.92)), cv2.FONT_HERSHEY_SIMPLEX, font_small, viol_col, 1, cv2.LINE_AA)

        # Compact Top-Right Highest Risk Worker
        p2_x1 = frame_width - panel_w - 8
        p2_y1 = hdr_h + 5
        p2_x2 = frame_width - 8
        p2_y2 = p2_y1 + panel_h
        ov2 = annotated_frame[p2_y1:p2_y2, p2_x1:p2_x2].copy()
        cv2.rectangle(ov2, (0, 0), (panel_w, panel_h), (16, 16, 22), -1)
        cv2.addWeighted(ov2, 0.82, annotated_frame[p2_y1:p2_y2, p2_x1:p2_x2], 0.18, 0, annotated_frame[p2_y1:p2_y2, p2_x1:p2_x2])

        top_ass = None
        if worker_assessments:
            sorted_workers = sorted(worker_assessments, key=lambda wa: wa[1].risk_score, reverse=True)
            _, top_ass = sorted_workers[0]
            top_color = (30, 30, 230) if top_ass.risk_level == "HIGH" else ((0, 140, 255) if top_ass.risk_level == "MEDIUM" else (40, 200, 40))
            cv2.rectangle(annotated_frame, (p2_x1, p2_y1), (p2_x2, p2_y2), top_color, 1)

            cv2.putText(annotated_frame, f"HIGHEST RISK: Worker #{top_ass.worker_id}", (p2_x1 + 6, p2_y1 + int(panel_h * 0.22)), cv2.FONT_HERSHEY_SIMPLEX, font_body, top_color, 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Risk: {top_ass.risk_score}/100 [{top_ass.risk_level}]", (p2_x1 + 6, p2_y1 + int(panel_h * 0.46)), cv2.FONT_HERSHEY_SIMPLEX, font_small, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"PPE: H:{top_ass.hardhat_status} | V:{top_ass.vest_status} | M:{top_ass.mask_status}", (p2_x1 + 6, p2_y1 + int(panel_h * 0.70)), cv2.FONT_HERSHEY_SIMPLEX, font_small, (220, 220, 220), 1, cv2.LINE_AA)

            top_reasons_str = " | ".join(top_ass.risk_reasons) if top_ass.risk_reasons else "All gear in place"
            max_char = 24 if not is_hd else 34
            if len(top_reasons_str) > max_char:
                top_reasons_str = top_reasons_str[:max_char - 3] + "..."
            cv2.putText(annotated_frame, f"Reasons: {top_reasons_str}", (p2_x1 + 6, p2_y1 + int(panel_h * 0.92)), cv2.FONT_HERSHEY_SIMPLEX, font_small, top_color, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(annotated_frame, (p2_x1, p2_y1), (p2_x2, p2_y2), (65, 65, 80), 1)
            cv2.putText(annotated_frame, "HIGHEST RISK: NONE", (p2_x1 + 6, p2_y1 + int(panel_h * 0.22)), cv2.FONT_HERSHEY_SIMPLEX, font_body, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, "Scanning for workers...", (p2_x1 + 6, p2_y1 + int(panel_h * 0.50)), cv2.FONT_HERSHEY_SIMPLEX, font_small, (140, 140, 140), 1, cv2.LINE_AA)

        # Bottom Alert / Evidence Bar
        bot_y1 = frame_height - bot_h
        bot_ov = annotated_frame[bot_y1:frame_height, 0:frame_width].copy()
        cv2.rectangle(bot_ov, (0, 0), (frame_width, bot_h), (16, 14, 14), -1)
        cv2.addWeighted(bot_ov, 0.85, annotated_frame[bot_y1:frame_height, 0:frame_width], 0.15, 0, annotated_frame[bot_y1:frame_height, 0:frame_width])
        cv2.line(annotated_frame, (0, bot_y1), (frame_width, bot_y1), (65, 65, 75), 1)

        alert_worker = None
        if worker_assessments:
            for _, ass in sorted(worker_assessments, key=lambda wa: wa[1].risk_score, reverse=True):
                if ass.compliance_state == "CONFIRMED VIOLATION" or ass.risk_level == "HIGH":
                    alert_worker = ass
                    break

        if self.last_evidence_banner and (time.time() - self.last_evidence_time < 4.0):
            ev_text = f"[EVIDENCE SAVED] {self.last_evidence_banner} | Safety Event Logged"
            cv2.putText(annotated_frame, ev_text, (10, frame_height - int(bot_h * 0.28)), cv2.FONT_HERSHEY_SIMPLEX, font_small, (80, 240, 120), 1, cv2.LINE_AA)
        elif alert_worker:
            reason_summary = " | ".join(alert_worker.risk_reasons) if alert_worker.risk_reasons else "Safety Violation"
            max_c = 40 if not is_hd else 65
            if len(reason_summary) > max_c:
                reason_summary = reason_summary[:max_c - 3] + "..."
            alert_msg = f"[ALERT] HIGH RISK / CONFIRMED VIOLATION: Worker #{alert_worker.worker_id} | Reason: {reason_summary} | Evidence: Active"
            cv2.putText(annotated_frame, alert_msg, (10, frame_height - int(bot_h * 0.28)), cv2.FONT_HERSHEY_SIMPLEX, font_small, (60, 80, 255), 1, cv2.LINE_AA)
        elif total_workers > 0 and compliant_count == total_workers:
            ok_msg = f"[NORMAL] All {total_workers} Worker(s) Compliant | Active Monitoring | Press 'Q' to Exit"
            cv2.putText(annotated_frame, ok_msg, (10, frame_height - int(bot_h * 0.28)), cv2.FONT_HERSHEY_SIMPLEX, font_small, (100, 220, 100), 1, cv2.LINE_AA)
        else:
            idle_msg = f"[MONITORING] VisionX Active | ByteTrack Tracking | Press 'Q' to Exit"
            cv2.putText(annotated_frame, idle_msg, (10, frame_height - int(bot_h * 0.28)), cv2.FONT_HERSHEY_SIMPLEX, font_small, (180, 180, 190), 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 4. SAFETY EVIDENCE RECORDING (ON CONFIRMED VIOLATION)
        # ---------------------------------------------------------------------
        saved_snapshots = []
        if record_evidence:
            for _, assessment in worker_assessments:
                if assessment.compliance_state == "CONFIRMED VIOLATION":
                    saved_path = self.safety_engine.record_evidence_if_needed(assessment, annotated_frame)
                    if saved_path:
                        saved_snapshots.append(saved_path)
                        self.last_evidence_banner = f"EVIDENCE SAVED: {os.path.basename(saved_path)}"
                        self.last_evidence_time = time.time()

        frame_summary = {
            "frame_idx": frame_idx,
            "total_workers": total_workers,
            "compliant_count": compliant_count,
            "violations_count": tot_violations,
            "check_count": check_count,
            "compliance_rate": compliance_rate,
            "top_worker": top_ass,
            "worker_assessments": [ass for _, ass in worker_assessments],
            "saved_snapshots": saved_snapshots,
        }

        return annotated_frame, frame_summary

    def process_video(self, input_path, output_path=None, max_frames=None, progress_callback=None):
        """
        Processes a full video file, writes annotated output video,
        and aggregates complete summary statistics for the web UI.

        Returns:
            summary_dict
        """
        self.reset_tracking()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        unique_worker_ids = set()
        max_visible_workers = 0
        highest_risk_score = 0
        highest_risk_worker_id = None
        highest_risk_level = "LOW"
        highest_risk_reasons = []
        all_reasons = set()
        all_snapshots = []
        total_violations_accumulated = 0
        compliant_sum = 0
        compliance_samples = 0

        t0 = time.time()

        while True:
            if max_frames and frame_idx >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            annotated_frame, frame_stats = self.process_frame(frame, frame_idx, record_evidence=True)

            if writer:
                writer.write(annotated_frame)

            for snap in frame_stats["saved_snapshots"]:
                all_snapshots.append(snap)

            # Track maximum simultaneous workers visible across frames
            current_visible_workers = len(frame_stats["worker_assessments"])
            max_visible_workers = max(max_visible_workers, current_visible_workers)

            # Accumulate metrics
            for ass in frame_stats["worker_assessments"]:
                unique_worker_ids.add(ass.worker_id)
                for r in ass.risk_reasons:
                    all_reasons.add(r)
                if ass.risk_score > highest_risk_score:
                    highest_risk_score = ass.risk_score
                    highest_risk_worker_id = ass.worker_id
                    highest_risk_level = ass.risk_level
                    highest_risk_reasons = list(ass.risk_reasons)

            if frame_stats["total_workers"] > 0:
                compliance_samples += 1
                compliant_sum += frame_stats["compliance_rate"]
                total_violations_accumulated += frame_stats["violations_count"]

            if progress_callback and total_video_frames > 0:
                progress_pct = min(100.0, (frame_idx / total_video_frames) * 100.0)
                progress_callback(frame_idx, total_video_frames, progress_pct)

        cap.release()
        if writer:
            writer.release()

        elapsed_time = time.time() - t0
        avg_compliance = (compliant_sum / compliance_samples) if compliance_samples > 0 else 0.0

        # Calculate worker count from max_visible_workers (not total historical IDs)
        total_workers = max_visible_workers
        compliant_workers = max(0, total_workers - (1 if highest_risk_score > 30 else 0))

        return {
            "total_workers": total_workers,
            "compliant_workers": compliant_workers,
            "violations_detected": 1 if highest_risk_score > 30 else 0,
            "check_workers": 0,
            "compliance_rate": round(avg_compliance, 1),
            "highest_risk_worker": f"Worker #{highest_risk_worker_id}" if highest_risk_worker_id else "None",
            "highest_risk_score": highest_risk_score,
            "highest_risk_level": highest_risk_level,
            "detected_reasons": list(all_reasons) if all_reasons else ["All PPE Compliant"],
            "snapshots_generated": len(all_snapshots),
            "snapshot_paths": all_snapshots,
            "safety_evidence_captured": len(all_snapshots) > 0,
            "frames_processed": frame_idx,
            "processing_time_seconds": round(elapsed_time, 2),
            "output_video_path": output_path,
        }
