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
import cv2

from video_processor import VisionXProcessor, DEVICE, MODEL_PATH, CONFIDENCE


# =============================================================================
# CONFIGURATION / DEMO MODE
# =============================================================================

# Video source
VIDEO_SOURCE = "source_files\\video_3.mp4"

# Uncomment this line to use the default webcam
# VIDEO_SOURCE = 0

# Maximum frames to process
# None = continuous/full video
MAX_FRAMES = int(os.environ.get("VISIONX_MAX_FRAMES", "0")) or None


# =============================================================================
# INITIALIZE REUSABLE PROCESSOR
# =============================================================================

print("=" * 60)
print("VisionX - Construction PPE Compliance & Worker Safety Monitoring")
print(f"Compute device: {DEVICE}")

processor = VisionXProcessor(
    model_path=MODEL_PATH,
    device=DEVICE,
    confidence=CONFIDENCE,
)


# =============================================================================
# OPEN VIDEO OR WEBCAM STREAM
# =============================================================================

print(f"Opening video source: {VIDEO_SOURCE} ...")

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"ERROR: Could not open video source: {VIDEO_SOURCE}")

    if VIDEO_SOURCE == 0:
        print(
            "Webcam device 0 not accessible. "
            "Ensure camera permissions or try video path."
        )

    sys.exit(1)


print()
print("VisionX PPE Compliance & Dynamic Risk System started.")
print("Tracking: ByteTrack (persist=True)")
print("Video loop: ENABLED")
print("Keyboard control: Press 'Q' to quit.")
print()


# =============================================================================
# MAIN VIDEO PROCESSING LOOP
# =============================================================================

frame_number = 0

while True:

    # ---------------------------------------------------------
    # Optional maximum-frame limit
    # ---------------------------------------------------------
    if MAX_FRAMES and frame_number >= MAX_FRAMES:
        print(f"Reached configured limit of {MAX_FRAMES} frames. Exiting.")
        break

    # ---------------------------------------------------------
    # Read next frame
    # ---------------------------------------------------------
    ret, frame = cap.read()

    # ---------------------------------------------------------
    # VIDEO LOOP / EOF HANDLING
    # Looping is applied only to video file mode, NOT webcam (VIDEO_SOURCE == 0)
    # ---------------------------------------------------------
    if not ret:
        if VIDEO_SOURCE != 0 and not MAX_FRAMES:
            print("Video ended - looping back to beginning.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_number = 0
            continue
        else:
            print("End of video stream reached or frame read error.")
            break

    frame_number += 1

    # ---------------------------------------------------------
    # Process frame through VisionX engine
    # ---------------------------------------------------------
    annotated_frame, frame_stats = processor.process_frame(
        frame,
        frame_number,
        record_evidence=True
    )

    # ---------------------------------------------------------
    # Display annotated frame
    # ---------------------------------------------------------
    cv2.imshow(
        "VisionX - Construction PPE Compliance V4.1",
        annotated_frame
    )

    # ---------------------------------------------------------
    # Keyboard and window-close control
    # Support 'q', 'Q', ESC (27), and closing the window via [X]
    # ---------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key in [ord("q"), ord("Q"), 27]:
        print("User requested exit.")
        break

    try:
        if cv2.getWindowProperty("VisionX - Construction PPE Compliance V4.1", cv2.WND_PROP_VISIBLE) < 1:
            print("OpenCV window closed by user.")
            break
    except Exception:
        pass


# =============================================================================
# CLEANUP
# =============================================================================

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("VisionX PPE Compliance System stopped.")
print(f"Total frames processed: {frame_number}")
print(
    f"Safety events logged to: "
    f"{processor.safety_engine.csv_file}"
)
print(
    f"Violation snapshots stored in: "
    f"{processor.safety_engine.snapshots_dir}"
)