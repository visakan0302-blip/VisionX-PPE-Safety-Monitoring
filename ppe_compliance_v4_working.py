from ultralytics import YOLO
import cv2
import math

# ============================================================
# VISIONX - OD-04 CONSTRUCTION PPE COMPLIANCE
# VERSION 4
# YOLO TRACKING + WORKER-PPE ASSOCIATION
# ============================================================

MODEL_PATH = "models\\best.pt"
VIDEO_PATH = "source_files\\hardhat.mp4"

CONFIDENCE = 0.30

# Required PPE
REQUIRED_PPE = [
    "Hardhat",
    "Safety Vest"
]

# PPE classes
PPE_CLASSES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Safety Vest"
]


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
print("VisionX PPE Compliance System V4 started.")
print("YOLO tracking enabled.")
print("Press Q to quit.")
print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def center(box):
    """Return center of bounding box."""

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def point_distance(p1, p2):
    """Distance between two points."""

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def expanded_region(person_box, frame_width, frame_height):
    """
    Expand worker region.

    The top is expanded because a helmet may be
    above the Person bounding box.
    """

    x1, y1, x2, y2 = person_box

    width = x2 - x1
    height = y2 - y1

    return (
        max(0, x1 - width * 0.20),
        max(0, y1 - height * 0.40),
        min(frame_width, x2 + width * 0.20),
        min(frame_height, y2 + height * 0.15)
    )


def inside_region(box, region):
    """Check whether box center is inside region."""

    cx, cy = center(box)

    x1, y1, x2, y2 = region

    return (
        x1 <= cx <= x2 and
        y1 <= cy <= y2
    )


def body_position_ok(ppe_name, ppe_box, person_box):
    """
    Check whether PPE is in a reasonable body position.
    """

    _, py = center(ppe_box)

    x1, y1, x2, y2 = person_box

    height = max(y2 - y1, 1)

    relative_y = (py - y1) / height

    # Helmet
    if ppe_name in ["Hardhat", "NO-Hardhat"]:
        return relative_y <= 0.50

    # Mask
    if ppe_name in ["Mask", "NO-Mask"]:
        return relative_y <= 0.65

    # Vest
    if ppe_name in ["Safety Vest", "NO-Safety Vest"]:
        return 0.10 <= relative_y <= 1.00

    return True


def association_score(ppe_box, person_box):
    """
    Calculate normalized distance between PPE center
    and worker center.

    Lower score = better association.
    """

    ppe_center = center(ppe_box)
    person_center = center(person_box)

    distance_value = point_distance(
        ppe_center,
        person_center
    )

    x1, y1, x2, y2 = person_box

    person_height = max(y2 - y1, 1)

    return distance_value / person_height


# ============================================================
# VIDEO LOOP
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    frame_height, frame_width = frame.shape[:2]


    # ========================================================
    # YOLO TRACKING
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        device=0,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]


    # ========================================================
    # DETECTION STORAGE
    # ========================================================

    persons = []
    ppe_detections = []


    # ========================================================
    # READ YOLO RESULTS
    # ========================================================

    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            class_id = int(boxes.cls[i])

            confidence = float(boxes.conf[i])

            class_name = names[class_id]

            box = boxes.xyxy[i].tolist()

            box = tuple(box)


            # ------------------------------------------------
            # PERSON
            # ------------------------------------------------

            if class_name == "Person":

                # Get persistent tracking ID
                if boxes.id is not None:

                    track_id = int(boxes.id[i])

                else:

                    track_id = None


                persons.append({
                    "id": track_id,
                    "box": box,
                    "confidence": confidence
                })


            # ------------------------------------------------
            # PPE
            # ------------------------------------------------

            elif class_name in PPE_CLASSES:

                ppe_detections.append({
                    "name": class_name,
                    "box": box,
                    "confidence": confidence
                })


    # ========================================================
    # WORKER STATUS
    # ========================================================

    compliant_count = 0
    violation_count = 0
    check_count = 0


    # ========================================================
    # PROCESS EVERY WORKER
    # ========================================================

    for worker_index, person in enumerate(persons):

        person_box = person["box"]

        worker_id = person["id"]

        # Fallback ID if tracker has not assigned one
        if worker_id is None:

            worker_id = worker_index + 1


        # Expanded association area
        association_area = expanded_region(
            person_box,
            frame_width,
            frame_height
        )


        # ----------------------------------------------------
        # FIND PPE BELONGING TO THIS WORKER
        # ----------------------------------------------------

        associated_ppe = []


        for ppe in ppe_detections:

            ppe_box = ppe["box"]

            # PPE must be near/inside worker area
            if not inside_region(
                ppe_box,
                association_area
            ):
                continue


            # Body-position check
            if not body_position_ok(
                ppe["name"],
                ppe_box,
                person_box
            ):
                continue


            score = association_score(
                ppe_box,
                person_box
            )


            associated_ppe.append({
                "name": ppe["name"],
                "box": ppe_box,
                "confidence": ppe["confidence"],
                "score": score
            })


        # ----------------------------------------------------
        # KEEP BEST DETECTION FOR EACH PPE TYPE
        # ----------------------------------------------------

        best_ppe = {}

        for ppe in associated_ppe:

            name = ppe["name"]

            if name not in best_ppe:

                best_ppe[name] = ppe

            else:

                # Keep higher confidence
                if ppe["confidence"] > best_ppe[name]["confidence"]:

                    best_ppe[name] = ppe


        detected_names = set(best_ppe.keys())


        # ====================================================
        # PPE STATUS
        # ====================================================

        # -----------------------------
        # HARDHAT
        # -----------------------------

        if "NO-Hardhat" in detected_names:

            hardhat_status = "MISSING"

        elif "Hardhat" in detected_names:

            hardhat_status = "YES"

        else:

            hardhat_status = "UNKNOWN"


        # -----------------------------
        # SAFETY VEST
        # -----------------------------

        if "NO-Safety Vest" in detected_names:

            vest_status = "MISSING"

        elif "Safety Vest" in detected_names:

            vest_status = "YES"

        else:

            vest_status = "UNKNOWN"


        # -----------------------------
        # MASK
        # -----------------------------

        if "NO-Mask" in detected_names:

            mask_status = "MISSING"

        elif "Mask" in detected_names:

            mask_status = "YES"

        else:

            mask_status = "UNKNOWN"


        # ====================================================
        # COMPLIANCE ENGINE
        # ====================================================

        violations = []


        if hardhat_status == "MISSING":

            violations.append("Hardhat")


        if vest_status == "MISSING":

            violations.append("Safety Vest")


        # Explicit missing PPE = violation
        if violations:

            status = "VIOLATION"

            violation_count += 1


        # Both required PPE confirmed
        elif (
            hardhat_status == "YES"
            and
            vest_status == "YES"
        ):

            status = "COMPLIANT"

            compliant_count += 1


        # Insufficient evidence
        else:

            status = "CHECK PPE"

            check_count += 1


        # ====================================================
        # DRAW WORKER BOX
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
        # WORKER INFORMATION
        # ====================================================

        label_y = max(y1 - 85, 25)


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
            (x1, label_y + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Vest: {vest_status}",
            (x1, label_y + 41),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Mask: {mask_status}",
            (x1, label_y + 61),
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
            (
                x1,
                min(y2 + 25, frame_height - 10)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


    # ========================================================
    # DASHBOARD
    # ========================================================

    total_workers = len(persons)


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
        (445, 175),
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
        "YOLO TRACKING: ON",
        (250, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "VisionX - Construction PPE Compliance V4",
        frame
    )


    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("VisionX PPE Compliance System stopped.")