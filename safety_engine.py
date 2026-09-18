"""
=============================================================================
VisionX - Construction PPE Detection & Worker Safety Monitoring
Safety Risk Engine (safety_engine.py)

Original System Contributions:
1. Worker-centric PPE compliance & risk modeling
2. Persistent worker-level violation history with noise tolerance
3. Temporal violation confirmation (preventing single-frame false alarms)
4. Context-aware hazard proximity scoring (machinery & vehicle geometry)
5. Automatic safety evidence generation (annotated snapshots & CSV audit trail)
6. Explainable heuristic risk score (0-100) and human-readable causal reasons
=============================================================================
"""

import os
import csv
import time
import math
from datetime import datetime
from collections import deque
import cv2

# =============================================================================
# CONFIGURATION CONSTANTS (Easily tunable at top of file)
# =============================================================================

# Temporal confirmation settings
VIOLATION_CONFIRMATION_FRAMES = 5
HISTORY_LENGTH = 10
REQUIRED_VIOLATION_FRAMES = 5

# Proximity geometry settings (relative to worker bounding box size)
HAZARD_PROXIMITY_FACTOR = 1.25

# Evidence logging & snapshot cooldown settings
SNAPSHOT_COOLDOWN_SECONDS = 10.0
SNAPSHOTS_DIR = "violation_snapshots"
LOGS_DIR = "violation_logs"
CSV_LOG_FILE = os.path.join(LOGS_DIR, "safety_events.csv")

# Stale worker cleanup
STALE_WORKER_TIMEOUT_FRAMES = 120


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class WorkerAssessment:
    """Holds the complete safety evaluation result for a single worker."""

    def __init__(
        self,
        worker_id,
        hardhat_status,
        vest_status,
        mask_status,
        machinery_near,
        vehicle_near,
        persistent_violation,
        compliance_state,
        risk_score,
        risk_level,
        risk_reasons,
        frame_idx,
    ):
        self.worker_id = worker_id
        self.hardhat_status = hardhat_status
        self.vest_status = vest_status
        self.mask_status = mask_status
        self.machinery_near = machinery_near
        self.vehicle_near = vehicle_near
        self.persistent_violation = persistent_violation
        self.compliance_state = compliance_state  # "COMPLIANT", "CHECK", "ACTIVE VIOLATION", "CONFIRMED VIOLATION"
        self.risk_score = risk_score              # 0 to 100
        self.risk_level = risk_level              # "LOW", "MEDIUM", "HIGH"
        self.risk_reasons = risk_reasons          # List[str]
        self.frame_idx = frame_idx
        self.snapshot_saved = None                # Path to snapshot if recorded in this frame


# =============================================================================
# SAFETY RISK ENGINE
# =============================================================================

class SafetyRiskEngine:
    """
    Worker-Centric Dynamic Safety Risk Engine.

    Calculates an explainable heuristic risk score (0-100), tracks temporal
    violation histories per worker, determines hazard proximity via scale-adaptive
    bounding-box geometry, and manages automated evidence generation.
    """

    def __init__(
        self,
        history_length=HISTORY_LENGTH,
        required_violation_frames=REQUIRED_VIOLATION_FRAMES,
        cooldown_seconds=SNAPSHOT_COOLDOWN_SECONDS,
        proximity_factor=HAZARD_PROXIMITY_FACTOR,
        snapshots_dir=SNAPSHOTS_DIR,
        logs_dir=LOGS_DIR,
        csv_file=CSV_LOG_FILE,
        require_mask=False,
    ):
        self.history_length = history_length
        self.required_violation_frames = required_violation_frames
        self.cooldown_seconds = cooldown_seconds
        self.proximity_factor = proximity_factor
        self.snapshots_dir = snapshots_dir
        self.logs_dir = logs_dir
        self.csv_file = csv_file
        self.require_mask = require_mask

        # Worker state tracking: worker_id -> deque of bools (has_raw_violation)
        self.worker_histories = {}
        # Worker last seen frame index: worker_id -> int
        self.worker_last_seen = {}
        # Worker snapshot tracking: worker_id -> {"time": float, "risk_level": str}
        self.worker_snapshot_records = {}

        # Ensure output directories exist
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        """Initialize the CSV log file with required columns if not already present."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "Worker_ID",
                    "Hardhat_Status",
                    "Vest_Status",
                    "Mask_Status",
                    "Machinery_Near",
                    "Vehicle_Near",
                    "Persistent_Violation",
                    "Risk_Score",
                    "Risk_Level",
                    "Reasons",
                    "Snapshot_Path",
                ])

    # -------------------------------------------------------------------------
    # HAZARD PROXIMITY GEOMETRY
    # -------------------------------------------------------------------------
    @staticmethod
    def bbox_distance(box1, box2):
        """
        Calculates the minimum Euclidean distance between two 2D axis-aligned bounding boxes.
        Returns 0.0 if boxes intersect or overlap.
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2

        # Horizontal separation
        if b1_x2 < b2_x1:
            dx = b2_x1 - b1_x2
        elif b2_x2 < b1_x1:
            dx = b1_x1 - b2_x2
        else:
            dx = 0.0

        # Vertical separation
        if b1_y2 < b2_y1:
            dy = b2_y1 - b1_y2
        elif b2_y2 < b1_y1:
            dy = b1_y1 - b2_y2
        else:
            dy = 0.0

        return math.sqrt(dx * dx + dy * dy)

    def is_hazard_near(self, worker_box, hazard_boxes):
        """
        Determines whether any hazard in hazard_boxes is close to the worker.
        Uses a dynamic proximity radius derived from the worker's bounding-box dimensions.
        """
        if not hazard_boxes:
            return False

        wx1, wy1, wx2, wy2 = worker_box
        worker_w = max(1.0, wx2 - wx1)
        worker_h = max(1.0, wy2 - wy1)
        worker_scale = max(worker_w, worker_h)

        # Dynamic proximity threshold based on worker scale
        threshold = worker_scale * self.proximity_factor

        for h_box in hazard_boxes:
            dist = self.bbox_distance(worker_box, h_box)
            if dist <= threshold:
                return True

        return False

    # -------------------------------------------------------------------------
    # TEMPORAL VIOLATION VALIDATION
    # -------------------------------------------------------------------------
    def update_worker_history(self, worker_id, has_raw_violation, frame_idx):
        """
        Updates the sliding window history of violation observations for worker_id.
        Allows single-frame noise tolerance so temporary detection dropouts
        do not instantly reset confirmed violations.
        """
        if worker_id not in self.worker_histories:
            self.worker_histories[worker_id] = deque(maxlen=self.history_length)

        history = self.worker_histories[worker_id]
        history.append(has_raw_violation)
        self.worker_last_seen[worker_id] = frame_idx

        # Count violations in current history window
        violation_frames_count = sum(1 for v in history if v)
        is_persistent = violation_frames_count >= self.required_violation_frames

        return is_persistent, violation_frames_count

    # -------------------------------------------------------------------------
    # WORKER SAFETY RISK EVALUATION
    # -------------------------------------------------------------------------
    def evaluate_worker(
        self,
        worker_id,
        hardhat_status,
        vest_status,
        mask_status,
        worker_box,
        machinery_boxes,
        vehicle_boxes,
        frame_idx,
    ):
        """
        Evaluates a worker by combining PPE status, temporal persistence,
        and contextual hazard proximity into an explainable 0-100 risk score.
        """
        # 1. Raw PPE compliance check
        has_missing_ppe = (
            hardhat_status == "MISSING" or
            vest_status == "MISSING" or
            (self.require_mask and mask_status == "MISSING")
        )

        has_uncertain_ppe = (
            (hardhat_status == "UNKNOWN" or vest_status == "UNKNOWN") and
            not has_missing_ppe
        )

        # 2. Temporal validation
        is_persistent, violation_count = self.update_worker_history(
            worker_id, has_missing_ppe, frame_idx
        )

        # 3. Determine compliance state
        if is_persistent:
            compliance_state = "CONFIRMED VIOLATION"
        elif has_missing_ppe:
            compliance_state = "ACTIVE VIOLATION"
        elif has_uncertain_ppe:
            compliance_state = "CHECK"
        else:
            compliance_state = "COMPLIANT"

        # 4. Contextual hazard proximity
        machinery_near = self.is_hazard_near(worker_box, machinery_boxes)
        vehicle_near = self.is_hazard_near(worker_box, vehicle_boxes)

        # 5. Explainable Risk Scoring (0 - 100)
        risk_score = 0
        risk_reasons = []

        if hardhat_status == "MISSING":
            risk_score += 35
            risk_reasons.append("Missing Hardhat")

        if vest_status == "MISSING":
            risk_score += 25
            risk_reasons.append("Missing Safety Vest")

        if mask_status == "MISSING" and self.require_mask:
            risk_score += 15
            risk_reasons.append("Missing Mask")

        if machinery_near:
            risk_score += 25
            risk_reasons.append("Machinery Near")

        if vehicle_near:
            risk_score += 20
            risk_reasons.append("Vehicle Near")

        if is_persistent:
            risk_score += 10
            risk_reasons.append("Persistent Violation")

        # Clamp score to maximum of 100
        risk_score = min(100, max(0, risk_score))

        # 6. Risk Level Categorization
        if risk_score <= 30:
            risk_level = "LOW"
        elif risk_score <= 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return WorkerAssessment(
            worker_id=worker_id,
            hardhat_status=hardhat_status,
            vest_status=vest_status,
            mask_status=mask_status,
            machinery_near=machinery_near,
            vehicle_near=vehicle_near,
            persistent_violation=is_persistent,
            compliance_state=compliance_state,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_reasons=risk_reasons,
            frame_idx=frame_idx,
        )

    # -------------------------------------------------------------------------
    # AUTOMATIC SAFETY EVIDENCE & CSV LOGGING
    # -------------------------------------------------------------------------
    def record_evidence_if_needed(self, assessment, annotated_frame):
        """
        Automatically saves an annotated snapshot and appends an audit record to CSV
        when a worker reaches CONFIRMED VIOLATION status.

        Applies per-worker cooldown logic, with an override if the risk escalates to HIGH.
        """
        if assessment.compliance_state != "CONFIRMED VIOLATION":
            return None

        worker_id = assessment.worker_id
        current_time = time.time()
        record = self.worker_snapshot_records.get(worker_id)

        should_record = False
        if record is None:
            # First confirmed violation for this worker
            should_record = True
        else:
            time_elapsed = current_time - record["time"]
            prev_level = record["risk_level"]
            # Cooldown expired or risk escalated to HIGH
            if time_elapsed >= self.cooldown_seconds:
                should_record = True
            elif assessment.risk_level == "HIGH" and prev_level in ["LOW", "MEDIUM"]:
                should_record = True

        if not should_record:
            return None

        # Generate timestamp and filenames
        now_dt = datetime.now()
        timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S")
        timestamp_iso = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        filename = f"worker_{worker_id}_{timestamp_str}.jpg"
        filepath = os.path.join(self.snapshots_dir, filename)

        # Save snapshot
        cv2.imwrite(filepath, annotated_frame)
        assessment.snapshot_saved = filepath

        # Append CSV event record
        reasons_str = " | ".join(assessment.risk_reasons) if assessment.risk_reasons else "None"
        with open(self.csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_iso,
                f"Worker #{worker_id}",
                assessment.hardhat_status,
                assessment.vest_status,
                assessment.mask_status,
                "NEAR" if assessment.machinery_near else "FAR",
                "NEAR" if assessment.vehicle_near else "FAR",
                "YES" if assessment.persistent_violation else "NO",
                f"{assessment.risk_score}/100",
                assessment.risk_level,
                reasons_str,
                filepath,
            ])

        # Update cooldown state
        self.worker_snapshot_records[worker_id] = {
            "time": current_time,
            "risk_level": assessment.risk_level,
        }

        return filepath

    # -------------------------------------------------------------------------
    # STALE WORKER PRUNING
    # -------------------------------------------------------------------------
    def prune_stale_workers(self, current_frame_idx):
        """
        Removes worker IDs that haven't been observed for STALE_WORKER_TIMEOUT_FRAMES.
        Prevents unbounded memory growth in long-running video or CCTV streams.
        """
        stale_ids = [
            wid for wid, last_frame in self.worker_last_seen.items()
            if current_frame_idx - last_frame > STALE_WORKER_TIMEOUT_FRAMES
        ]
        for wid in stale_ids:
            self.worker_histories.pop(wid, None)
            self.worker_last_seen.pop(wid, None)
            self.worker_snapshot_records.pop(wid, None)
