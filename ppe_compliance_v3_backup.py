from ultralytics import YOLO
import cv2
import math

# ============================================================
# VISIONX - OD-04 CONSTRUCTION PPE COMPLIANCE
# VERSION 3
# ============================================================

MODEL_PATH = "models\\best.pt"
VIDEO_PATH = "source_files\\hardhat.mp4"

CONFIDENCE = 0.30

# Required PPE for this prototype
REQUIRED_PPE = ["Hardhat", "Safety Vest"]

# Tracking settings
MAX_TRACK_DISTANCE = 130
MAX_MISSING_FRAMES = 45


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading VisionX PPE model...")

model = YOLO(MODEL_PATH)

names = model.names

print("Model loaded successfully.")
print("Classes:", names)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

print()
print("VisionX PPE Compliance System started.")
print("Press Q to quit.")
print()


# ============================================================
# TRACKING STORAGE
# ============================================================

workers = {}

next_worker_id = 1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_center(box):
    """Get center point of bounding box."""

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def get_distance(point1, point2):
    """Calculate distance between two points."""

    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def expanded_person_box(person_box, frame_width, frame_height):
    """
    Expand the person region.

    The region is expanded upward because a helmet can
    physically sit above the Person detection box.
    """

    x1, y1, x2, y2 = person_box

    width = x2 - x1
    height = y2 - y1

    # Expand around the worker
    new_x1 = x1 - width * 0.15
    new_y1 = y1 - height * 0.30
    new_x2 = x2 + width * 0.15
    new_y2 = y2 + height * 0.10

    # Keep region inside image
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(frame_width, new_x2)
    new_y2 = min(frame_height, new_y2)

    return (
        new_x1,
        new_y1,
        new_x2,
        new_y2
    )


def center_inside(box, region):
    """Check whether box center is inside a region."""

    cx, cy = get_center(box)

    x1, y1, x2, y2 = region

    return (
        x1 <= cx <= x2 and
        y1 <= cy <= y2
    )


def valid_ppe_position(ppe_name, ppe_box, person_box):
    """
    Check whether PPE is in a reasonable location
    relative to the worker.
    """

    px, py = get_center(ppe_box)

    x1, y1, x2, y2 = person_box

    height = max(y2 - y1, 1)

    relative_y = (py - y1) / height

    # Helmet can be above the Person box
    if ppe_name in ["Hardhat", "NO-Hardhat"]:

        return relative_y <= 0.45

    # Mask should be near upper body
    if ppe_name in ["Mask", "NO-Mask"]:

        return relative_y <= 0.60

    # Vest should be around torso
    if ppe_name in ["Safety Vest", "NO-Safety Vest"]:

        return 0.15 <= relative_y <= 0.95

    return True


def find_worker(person_center):
    """
    Find the closest existing worker.

    This provides lightweight persistent IDs.
    """

    best_worker = None
    best_distance = float("inf")

    for worker_id, worker in workers.items():

        if worker["missing_frames"] > MAX_MISSING_FRAMES:
            continue

        d = get_distance(
            person_center,
            worker["center"]
        )

        if d < best_distance and d < MAX_TRACK_DISTANCE:

            best_distance = d
            best_worker = worker_id

    return best_worker


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    frame_height, frame_width = frame.shape[:2]

    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = model(
        frame,
        device=0,
        verbose=False,
        conf=CONFIDENCE
    )

    result = results[0]

    persons = []
    ppe_detections = []

    # ========================================================
    # EXTRACT DETECTIONS
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection_box = (
                x1,
                y1,
                x2,
                y2
            )

            # -----------------------------------------------
            # PERSON
            # -----------------------------------------------

            if class_name == "Person":

                persons.append({
                    "box": detection_box,
                    "confidence": confidence
                })

            # -----------------------------------------------
            # PPE
            # -----------------------------------------------

            elif class_name in [
                "Hardhat",
                "Mask",
                "NO-Hardhat",
                "NO-Mask",
                "NO-Safety Vest",
                "Safety Vest"
            ]:

                ppe_detections.append({
                    "name": class_name,
                    "box": detection_box,
                    "confidence": confidence
                })


    # ========================================================
    # UPDATE WORKER TRACKING
    # ========================================================

    current_worker_ids = []

    # Sort persons from left to right.
    # This makes IDs more stable when multiple people exist.
    persons.sort(
        key=lambda p: get_center(p["box"])[0]
    )

    for person in persons:

        person_box = person["box"]

        person_center = get_center(person_box)

        worker_id = find_worker(person_center)

        # New worker
        if worker_id is None:

            worker_id = next_worker_id

            next_worker_id += 1

            workers[worker_id] = {
                "center": person_center,
                "box": person_box,
                "missing_frames": 0
            }

        # Existing worker
        else:

            workers[worker_id]["center"] = person_center

            workers[worker_id]["box"] = person_box

            workers[worker_id]["missing_frames"] = 0

        current_worker_ids.append(worker_id)


    # ========================================================
    # UPDATE MISSING WORKERS
    # ========================================================

    for worker_id in list(workers.keys()):

        if worker_id not in current_worker_ids:

            workers[worker_id]["missing_frames"] += 1

        if workers[worker_id]["missing_frames"] > MAX_MISSING_FRAMES:

            del workers[worker_id]


    # ========================================================
    # FRAME COUNTERS
    # ========================================================

    compliant_count = 0
    violation_count = 0
    check_count = 0


    # ========================================================
    # PROCESS EACH WORKER
    # ========================================================

    for worker_id in current_worker_ids:

        worker = workers[worker_id]

        person_box = worker["box"]

        # Expanded association region
        association_region = expanded_person_box(
            person_box,
            frame_width,
            frame_height
        )

        detected_positive = set()
        detected_negative = set()


        # ====================================================
        # ASSOCIATE PPE
        # ====================================================

        for ppe in ppe_detections:

            ppe_name = ppe["name"]

            ppe_box = ppe["box"]

            # First check expanded worker region
            if not center_inside(
                ppe_box,
                association_region
            ):
                continue

            # Then check body position
            if not valid_ppe_position(
                ppe_name,
                ppe_box,
                person_box
            ):
                continue

            # Positive PPE
            if ppe_name in [
                "Hardhat",
                "Mask",
                "Safety Vest"
            ]:

                detected_positive.add(ppe_name)

            # Negative PPE
            elif ppe_name in [
                "NO-Hardhat",
                "NO-Mask",
                "NO-Safety Vest"
            ]:

                detected_negative.add(ppe_name)


        # ====================================================
        # DETERMINE INDIVIDUAL PPE STATUS
        # ====================================================

        # Hardhat
        if "NO-Hardhat" in detected_negative:

            hardhat_status = "MISSING"

        elif "Hardhat" in detected_positive:

            hardhat_status = "YES"

        else:

            hardhat_status = "UNKNOWN"


        # Safety Vest
        if "NO-Safety Vest" in detected_negative:

            vest_status = "MISSING"

        elif "Safety Vest" in detected_positive:

            vest_status = "YES"

        else:

            vest_status = "UNKNOWN"


        # Mask - informational only
        if "NO-Mask" in detected_negative:

            mask_status = "MISSING"

        elif "Mask" in detected_positive:

            mask_status = "YES"

        else:

            mask_status = "UNKNOWN"


        # ====================================================
        # COMPLIANCE DECISION
        # ====================================================

        violations = []

        # Explicit missing PPE
        if hardhat_status == "MISSING":

            violations.append("Hardhat")

        if vest_status == "MISSING":

            violations.append("Safety Vest")


        # Final status
        if len(violations) > 0:

            status = "VIOLATION"

            violation_count += 1

        elif (
            hardhat_status == "YES"
            and
            vest_status == "YES"
        ):

            status = "COMPLIANT"

            compliant_count += 1

        else:

            status = "CHECK PPE"

            check_count += 1


        # ====================================================
        # DRAW PERSON BOX
        # ====================================================

        x1, y1, x2, y2 = map(
            int,
            person_box
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


        # ====================================================
        # DRAW WORKER INFORMATION
        # ====================================================

        label_y = max(y1 - 90, 20)

        cv2.putText(
            frame,
            f"Worker #{worker_id}",
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Hardhat: {hardhat_status}",
            (x1, label_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Vest: {vest_status}",
            (x1, label_y + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Mask: {mask_status}",
            (x1, label_y + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )


        # ====================================================
        # STATUS
        # ====================================================

        if status == "VIOLATION":

            status_text = (
                "VIOLATION - Missing: "
                + ", ".join(violations)
            )

        elif status == "COMPLIANT":

            status_text = "COMPLIANT"

        else:

            status_text = "CHECK PPE"


        cv2.putText(
            frame,
            status_text,
            (x1, min(y2 + 25, frame_height - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


    # ========================================================
    # DASHBOARD
    # ========================================================

    total_workers = len(current_worker_ids)

    if total_workers > 0:

        compliance_rate = (
            compliant_count /
            total_workers
        ) * 100

    else:

        compliance_rate = 0


    # Dashboard
    cv2.rectangle(
        frame,
        (10, 10),
        (430, 175),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "VISIONX - PPE MONITOR",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Workers: {total_workers}",
        (25, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Compliant: {compliant_count}",
        (25, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Violations: {violation_count}",
        (220, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Check: {check_count}",
        (25, 121),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Compliance: {compliance_rate:.1f}%",
        (220, 121),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (25, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (330, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "VisionX - Construction PPE Compliance",
        frame
    )


    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("VisionX PPE Compliance System stopped.")