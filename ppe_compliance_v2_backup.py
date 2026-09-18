from ultralytics import YOLO
import cv2
import math

# ============================================================
# VISIONX - OD-04 CONSTRUCTION PPE COMPLIANCE
# VERSION 2
# ============================================================

MODEL_PATH = "models\\best.pt"
VIDEO_PATH = "source_files\\hardhat.mp4"

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

CONFIDENCE = 0.35

# PPE required for construction-site compliance
REQUIRED_PPE = {
    "Hardhat",
    "Safety Vest"
}

# Optional PPE
OPTIONAL_PPE = {
    "Mask"
}

# Maximum movement allowed between frames for the same worker
MAX_TRACK_DISTANCE = 100

# How many frames a worker can disappear before ID is removed
MAX_MISSING_FRAMES = 30


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("Loading VisionX PPE model...")

model = YOLO(MODEL_PATH)

names = model.names

print("Model loaded successfully.")
print("Classes:", names)


# ------------------------------------------------------------
# OPEN VIDEO
# ------------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

print()
print("VisionX PPE Compliance System started.")
print("Press Q to quit.")
print()


# ------------------------------------------------------------
# WORKER TRACKING DATA
# ------------------------------------------------------------

workers = {}

next_worker_id = 1


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def get_center(box):
    """Return center point of bounding box."""

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def distance(point1, point2):
    """Calculate distance between two points."""

    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def center_inside(box_small, box_large):
    """
    Check whether the center of one bounding box
    lies inside another bounding box.
    """

    cx, cy = get_center(box_small)

    x1, y1, x2, y2 = box_large

    return (
        x1 <= cx <= x2 and
        y1 <= cy <= y2
    )


def valid_ppe_position(ppe_name, ppe_box, person_box):
    """
    Check whether PPE is located in a reasonable
    region of the worker's body.
    """

    px, py = get_center(ppe_box)

    x1, y1, x2, y2 = person_box

    height = max(y2 - y1, 1)

    relative_y = (py - y1) / height

    # Helmet should be near the head
    if ppe_name in ["Hardhat", "NO-Hardhat"]:
        return relative_y <= 0.45

    # Mask should be near the face
    if ppe_name in ["Mask", "NO-Mask"]:
        return relative_y <= 0.55

    # Safety vest should be around torso
    if ppe_name in ["Safety Vest", "NO-Safety Vest"]:
        return 0.20 <= relative_y <= 0.90

    return True


def find_existing_worker(center_point):
    """
    Find the closest existing worker to the current
    person detection.
    """

    best_id = None
    best_distance = float("inf")

    for worker_id, worker in workers.items():

        if worker["missing_frames"] > MAX_MISSING_FRAMES:
            continue

        d = distance(
            center_point,
            worker["center"]
        )

        if d < best_distance and d < MAX_TRACK_DISTANCE:

            best_distance = d
            best_id = worker_id

    return best_id


# ------------------------------------------------------------
# MAIN VIDEO LOOP
# ------------------------------------------------------------

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model(
        frame,
        device=0,
        verbose=False,
        conf=CONFIDENCE
    )

    result = results[0]

    persons = []
    ppe_detections = []

    # --------------------------------------------------------
    # READ DETECTIONS
    # --------------------------------------------------------

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

            # Person
            if class_name == "Person":

                persons.append({
                    "box": detection_box,
                    "confidence": confidence
                })

            # PPE
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


    # --------------------------------------------------------
    # UPDATE WORKER TRACKING
    # --------------------------------------------------------

    current_worker_ids = []

    for person in persons:

        person_box = person["box"]

        person_center = get_center(person_box)

        worker_id = find_existing_worker(person_center)

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


    # --------------------------------------------------------
    # INCREASE MISSING COUNTER
    # --------------------------------------------------------

    for worker_id in list(workers.keys()):

        if worker_id not in current_worker_ids:

            workers[worker_id]["missing_frames"] += 1

        if workers[worker_id]["missing_frames"] > MAX_MISSING_FRAMES:

            del workers[worker_id]


    # --------------------------------------------------------
    # COMPLIANCE COUNTERS
    # --------------------------------------------------------

    compliant_count = 0
    violation_count = 0
    check_count = 0


    # --------------------------------------------------------
    # PROCESS EACH WORKER
    # --------------------------------------------------------

    for worker_id in current_worker_ids:

        worker = workers[worker_id]

        person_box = worker["box"]

        detected_ppe = set()

        negative_ppe = set()


        # ----------------------------------------------------
        # ASSOCIATE PPE WITH WORKER
        # ----------------------------------------------------

        for ppe in ppe_detections:

            ppe_name = ppe["name"]

            ppe_box = ppe["box"]

            # PPE center must be inside worker
            if not center_inside(
                ppe_box,
                person_box
            ):
                continue

            # Check body position
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

                detected_ppe.add(ppe_name)

            # Negative PPE
            elif ppe_name in [
                "NO-Hardhat",
                "NO-Mask",
                "NO-Safety Vest"
            ]:

                negative_ppe.add(ppe_name)


        # ----------------------------------------------------
        # DETERMINE COMPLIANCE
        # ----------------------------------------------------

        violations = []

        # Explicit missing hardhat
        if "NO-Hardhat" in negative_ppe:

            violations.append("Missing Hardhat")

        # Explicit missing safety vest
        if "NO-Safety Vest" in negative_ppe:

            violations.append("Missing Vest")


        # Positive required PPE
        hardhat_present = "Hardhat" in detected_ppe

        vest_present = "Safety Vest" in detected_ppe


        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        if violations:

            status = "VIOLATION"

            violation_count += 1

        elif hardhat_present and vest_present:

            status = "COMPLIANT"

            compliant_count += 1

        else:

            status = "CHECK PPE"

            check_count += 1


        # ----------------------------------------------------
        # DRAW WORKER BOX
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            person_box
        )


        # Worker box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # PPE STATUS TEXT
        # ----------------------------------------------------

        hardhat_symbol = "YES" if hardhat_present else "NO"

        vest_symbol = "YES" if vest_present else "NO"

        mask_symbol = (
            "YES"
            if "Mask" in detected_ppe
            else "N/A"
        )


        # Worker ID
        cv2.putText(
            frame,
            f"Worker #{worker_id}",
            (x1, max(y1 - 70, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # Hardhat
        cv2.putText(
            frame,
            f"Hardhat: {hardhat_symbol}",
            (x1, max(y1 - 45, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


        # Vest
        cv2.putText(
            frame,
            f"Vest: {vest_symbol}",
            (x1, max(y1 - 25, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if status == "VIOLATION":

            status_text = (
                "VIOLATION: "
                + ", ".join(violations)
            )

        elif status == "COMPLIANT":

            status_text = "COMPLIANT"

        else:

            status_text = "CHECK PPE"


        cv2.putText(
            frame,
            status_text,
            (x1, min(y2 + 25, frame.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    total_workers = len(current_worker_ids)

    if total_workers > 0:

        compliance_rate = (
            compliant_count /
            total_workers
        ) * 100

    else:

        compliance_rate = 0


    # Dashboard background
    cv2.rectangle(
        frame,
        (10, 10),
        (410, 155),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "VISIONX - PPE MONITOR",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Workers: {total_workers}",
        (25, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Compliant: {compliant_count}",
        (25, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Violations: {violation_count}",
        (210, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Check: {check_count}",
        (25, 124),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Compliance: {compliance_rate:.1f}%",
        (210, 124),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # Frame number
    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (25, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    # --------------------------------------------------------
    # SHOW VIDEO
    # --------------------------------------------------------

    cv2.imshow(
        "VisionX - Construction PPE Compliance",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------

cap.release()

cv2.destroyAllWindows()

print()
print("VisionX PPE Compliance System stopped.")