from ultralytics import YOLO
import cv2
import math

# ============================================================
# VISIONX - OD-04 CONSTRUCTION PPE COMPLIANCE
# ============================================================

MODEL_PATH = "models\\best.pt"
VIDEO_PATH = "source_files\\hardhat.mp4"

# Load trained PPE detection model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

print("VisionX PPE Compliance System started.")
print("Press Q to quit.")

# Class names from the trained model
names = model.names

# PPE classes
POSITIVE_PPE = {
    "Hardhat",
    "Mask",
    "Safety Vest"
}

NEGATIVE_PPE = {
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest"
}

# Required PPE
REQUIRED_PPE = {
    "Hardhat",
    "Mask",
    "Safety Vest"
}


def center(box):
    """Return center point of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def inside_person(ppe_box, person_box):
    """
    Check whether the center of a PPE detection
    lies inside the person's bounding box.
    """
    px, py = center(ppe_box)

    x1, y1, x2, y2 = person_box

    return x1 <= px <= x2 and y1 <= py <= y2


def ppe_position_valid(ppe_name, ppe_box, person_box):
    """
    Additional spatial filtering.

    Hardhat / Mask should normally be in the upper part
    of the worker.

    Safety Vest should normally be around the torso.
    """

    px, py = center(ppe_box)

    x1, y1, x2, y2 = person_box

    person_height = y2 - y1

    relative_y = (py - y1) / max(person_height, 1)

    if ppe_name in ["Hardhat", "NO-Hardhat"]:
        return relative_y <= 0.45

    if ppe_name in ["Mask", "NO-Mask"]:
        return relative_y <= 0.55

    if ppe_name in ["Safety Vest", "NO-Safety Vest"]:
        return 0.20 <= relative_y <= 0.90

    return True


# Process video frame by frame
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO on current frame
    results = model(
        frame,
        device=0,
        verbose=False,
        conf=0.35
    )

    result = results[0]

    # Store persons and PPE detections
    persons = []
    ppe_detections = []

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection_box = (x1, y1, x2, y2)

            if class_name == "Person":

                persons.append({
                    "box": detection_box,
                    "confidence": confidence
                })

            elif class_name in POSITIVE_PPE or class_name in NEGATIVE_PPE:

                ppe_detections.append({
                    "name": class_name,
                    "box": detection_box,
                    "confidence": confidence
                })

    # ========================================================
    # WORKER-PPE ASSOCIATION
    # ========================================================

    compliant_workers = 0
    violation_workers = 0

    for worker_index, person in enumerate(persons):

        person_box = person["box"]

        detected_ppe = set()
        violations = set()

        # Find PPE belonging to this worker
        for ppe in ppe_detections:

            if inside_person(ppe["box"], person_box):

                if ppe_position_valid(
                    ppe["name"],
                    ppe["box"],
                    person_box
                ):

                    detected_ppe.add(ppe["name"])

                    # Explicit missing-PPE classes
                    if ppe["name"] == "NO-Hardhat":
                        violations.add("Hardhat")

                    elif ppe["name"] == "NO-Mask":
                        violations.add("Mask")

                    elif ppe["name"] == "NO-Safety Vest":
                        violations.add("Safety Vest")

        # Determine which required PPE is present
        for required in REQUIRED_PPE:

            if required in detected_ppe:
                continue

            # If explicit negative detection wasn't found,
            # we don't automatically mark it as a violation.
            # This avoids falsely declaring missing PPE just
            # because the detector didn't see it.

        # Worker status
        if violations:

            status = "VIOLATION"
            violation_workers += 1

        else:

            # At least one positive PPE item detected
            positive_found = detected_ppe.intersection(POSITIVE_PPE)

            if positive_found:

                status = "COMPLIANT"
                compliant_workers += 1

            else:

                status = "CHECK PPE"

        # ====================================================
        # DRAW WORKER BOX
        # ====================================================

        x1, y1, x2, y2 = map(int, person_box)

        # Draw person rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        # Worker label
        worker_label = f"Worker #{worker_index + 1}"

        cv2.putText(
            frame,
            worker_label,
            (x1, max(y1 - 35, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Status label
        if status == "VIOLATION":
            status_text = "VIOLATION: " + ", ".join(violations)
            text_y = max(y1 - 10, 50)

        elif status == "COMPLIANT":
            status_text = "COMPLIANT"
            text_y = max(y1 - 10, 50)

        else:
            status_text = "CHECK PPE"
            text_y = max(y1 - 10, 50)

        cv2.putText(
            frame,
            status_text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # ========================================================
    # DASHBOARD OVERLAY
    # ========================================================

    total_workers = len(persons)

    if total_workers > 0:
        compliance_rate = (
            compliant_workers / total_workers
        ) * 100
    else:
        compliance_rate = 0

    cv2.rectangle(
        frame,
        (10, 10),
        (390, 125),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "VISIONX - PPE MONITOR",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Workers: {total_workers}",
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Compliant: {compliant_workers}",
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Violations: {violation_workers}",
        (200, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Compliance: {compliance_rate:.1f}%",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Display
    cv2.imshow(
        "VisionX - Construction PPE Compliance",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
cap.release()
cv2.destroyAllWindows()

print()
print("VisionX PPE Compliance System stopped.")