"""
Flask-SocketIO server. Bridges detector + OBD reader to the dashboard.

Latency notes:
  * Detection state is emitted at the detector's native rate (~25–30 Hz on Pi 4).
  * OBD state at ~5 Hz (typical adapter limit).
  * MJPEG endpoint is OPTIONAL — only opened when the dashboard requests it.
  * eventlet monkey-patches the stdlib so the threaded queues integrate cleanly.
"""
import eventlet
eventlet.monkey_patch()

import os
import time
import threading
from pathlib import Path
from queue import Queue, Empty

import cv2
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

from src.detector import DrowsinessDetector, DetectionState
from src.obd_reader import OBDReader, OBDState

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "face_landmarker.task"))
CAM_INDEX  = int(os.getenv("CAM_INDEX", "0"))
HOST       = os.getenv("HOST", "0.0.0.0")
PORT       = int(os.getenv("PORT", "5000"))

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["SECRET_KEY"] = os.urandom(24)
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")

# Queues — sized 1 so consumers always see latest, never stale
det_queue   = Queue(maxsize=2)
obd_queue   = Queue(maxsize=2)
frame_queue = Queue(maxsize=1)

detector: DrowsinessDetector | None = None
obd_reader: OBDReader | None = None


# ---------------------------------------------------------------------------
# Background broadcasters
# ---------------------------------------------------------------------------
def broadcast_detection():
    """Pull from detector queue and emit. Runs as eventlet green thread."""
    while True:
        try:
            state: DetectionState = det_queue.get(timeout=1.0)
            socketio.emit("detection", state.to_dict())
        except Empty:
            continue


def broadcast_obd():
    while True:
        try:
            state: OBDState = obd_queue.get(timeout=1.0)
            socketio.emit("obd", state.to_dict())
        except Empty:
            continue


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/health")
def health():
    return {
        "detector_alive": detector.is_alive() if detector else False,
        "obd_alive":      obd_reader.is_alive() if obd_reader else False,
    }


def _mjpeg_generator():
    """Yield JPEG frames for the optional preview. Uses moderate quality
    to keep bandwidth low — this is a debug aid, not the primary channel."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except Empty:
            continue
        ok, buf = cv2.imencode(".jpg", frame, encode_param)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/stream.mjpg")
def stream():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@socketio.on("connect")
def on_connect():
    print(f"[WS] client connected at {time.strftime('%H:%M:%S')}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    global detector, obd_reader

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Face landmarker model not found at {MODEL_PATH}. "
            "Run scripts/download_model.sh first."
        )

    detector = DrowsinessDetector(
        model_path=MODEL_PATH,
        out_queue=det_queue,
        frame_queue=frame_queue,
        src=CAM_INDEX,
    )
    obd_reader = OBDReader(out_queue=obd_queue)

    detector.start()
    obd_reader.start()

    socketio.start_background_task(broadcast_detection)
    socketio.start_background_task(broadcast_obd)

    try:
        socketio.run(app, host=HOST, port=PORT, debug=False, use_reloader=False)
    finally:
        detector.stop()
        obd_reader.stop()


if __name__ == "__main__":
    main()
