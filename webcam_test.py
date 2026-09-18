import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam failed")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device=0, conf=0.35, verbose=False)
    annotated = results[0].plot()

    names = results[0].names
    detected = [names[int(c)] for c in results[0].boxes.cls] if results[0].boxes else []

    if "Hardhat" in detected:
        helmet = "HELMET: OK"
    else:
        helmet = "HELMET: MISSING"

    if "Mask" in detected:
        mask = "MASK: OK"
    else:
        mask = "MASK: MISSING"

    cv2.putText(annotated, helmet, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if "OK" in helmet else (0, 0, 255), 2)

    cv2.putText(annotated, mask, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if "OK" in mask else (0, 0, 255), 2)

    cv2.putText(annotated, "GOGGLES: NOT SUPPORTED BY CURRENT MODEL",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow("VisionX - Personal PPE Check", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        break

cap.release()
cv2.destroyAllWindows()