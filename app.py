import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open camera (0 = webcam, or give CCTV video path)
cap = cv2.VideoCapture(0)

# Time tracking
human_start_time = None
ALERT_TIME = 120  # 2 minutes (120 seconds)
alert_sent = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run AI detection
    results = model(frame, stream=True)

    human_detected = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # 0 = person class in YOLO
                human_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "HUMAN ON TRACK",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

    # Time logic
    if human_detected:
        if human_start_time is None:
            human_start_time = time.time()

        elapsed = time.time() - human_start_time

        cv2.putText(frame, f"Time: {int(elapsed)} sec",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        if elapsed > ALERT_TIME and not alert_sent:
            print("🚨 ALERT: Human on railway track for more than 2 minutes!")
            alert_sent = True
            # HERE you can call email/SMS/API
    else:
        human_start_time = None
        alert_sent = False

    cv2.imshow("Railway Safety AI Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()