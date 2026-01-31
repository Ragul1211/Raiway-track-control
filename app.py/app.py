from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONF_THRESHOLD = 0.5
FPS_LIMIT = 15


def generate_frames():
    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # FPS control
        current_time = time.time()
        if current_time - prev_time < 1 / FPS_LIMIT:
            continue
        prev_time = current_time

        # Resize for performance
        frame = cv2.resize(frame, (640, 480))

        # YOLO inference
        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Class 0 = person
                if cls == 0 and conf > CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"HUMAN {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/shutdown")
def shutdown():
    cap.release()
    cv2.destroyAllWindows()
    return "Camera Released"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
