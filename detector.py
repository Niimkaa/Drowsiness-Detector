"""
Drowsiness detector — MediaPipe Face Landmarker (Tasks API).
EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), PERCLOS, head pose.

Threaded: capture loop is decoupled from inference, inference is decoupled
from emission. Latency optimized for Raspberry Pi 4/5.
"""
from __future__ import annotations

import math
import time
import threading
from collections import deque
from dataclasses import dataclass, asdict
from queue import Queue, Empty, Full
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe 478-point model)
# ---------------------------------------------------------------------------
LEFT_EYE  = [33, 160, 158, 133, 153, 144]   # outer, top-left, top-right, inner, btm-right, btm-left
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [78, 81, 13, 311, 308, 402, 14, 178]

# Thresholds — tune per camera/driver
EAR_THRESHOLD     = 0.21    # below = eyes closed
MAR_THRESHOLD     = 0.55    # above = yawning
EYES_CLOSED_SECS  = 1.2     # microsleep threshold
PERCLOS_WINDOW    = 60      # seconds rolling window
PERCLOS_ALERT     = 0.30    # 30% closure = drowsy (industry standard)


@dataclass
class DetectionState:
    """Snapshot sent to dashboard. Compact, JSON-serializable."""
    ts: float = 0.0
    face_detected: bool = False
    ear: float = 0.0
    mar: float = 0.0
    perclos: float = 0.0
    eyes_closed_for: float = 0.0
    yawning: bool = False
    head_down: bool = False
    drowsy: bool = False
    alert_level: int = 0          # 0=ok, 1=warn, 2=alarm
    fps: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _ear(landmarks, indices) -> float:
    p = [landmarks[i] for i in indices]
    # Eye aspect ratio: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    vert = _dist(p[1], p[5]) + _dist(p[2], p[4])
    horz = 2.0 * _dist(p[0], p[3])
    return vert / horz if horz > 1e-6 else 0.0


def _mar(landmarks, indices) -> float:
    p = [landmarks[i] for i in indices]
    # Mouth aspect: vertical opening vs horizontal width
    vert = _dist(p[1], p[7]) + _dist(p[2], p[6]) + _dist(p[3], p[5])
    horz = 3.0 * _dist(p[0], p[4])
    return vert / horz if horz > 1e-6 else 0.0


# ---------------------------------------------------------------------------
# Threaded capture — drops stale frames so inference always gets latest
# ---------------------------------------------------------------------------
class FrameGrabber(threading.Thread):
    def __init__(self, src=0, width=640, height=480, fps=30):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self):
        self._stop.set()
        self.cap.release()


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------
class DrowsinessDetector(threading.Thread):
    def __init__(
        self,
        model_path: str,
        out_queue: Queue,
        frame_queue: Optional[Queue] = None,
        src: int = 0,
    ):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.frame_queue = frame_queue
        self._stop = threading.Event()
        self.grabber = FrameGrabber(src=src)

        # MediaPipe Tasks Face Landmarker — newer & faster than legacy FaceMesh
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # Rolling buffers for PERCLOS
        self.eye_history: deque = deque(maxlen=PERCLOS_WINDOW * 30)  # ~30 FPS
        self.eyes_closed_start: Optional[float] = None

        # FPS smoothing
        self._fps_buf: deque = deque(maxlen=30)

    def run(self):
        self.grabber.start()
        time.sleep(0.5)  # warmup

        while not self._stop.is_set():
            t0 = time.time()
            frame = self.grabber.read()
            if frame is None:
                time.sleep(0.01)
                continue

            state = self._process(frame, ts=t0)

            # Push state to dashboard (drop if consumer slow)
            try:
                self.out_queue.put_nowait(state)
            except Full:
                try:
                    self.out_queue.get_nowait()
                except Empty:
                    pass
                self.out_queue.put_nowait(state)

            # Optionally push frame for MJPEG stream
            if self.frame_queue is not None:
                try:
                    self.frame_queue.put_nowait(frame)
                except Full:
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                    self.frame_queue.put_nowait(frame)

            # FPS
            dt = time.time() - t0
            self._fps_buf.append(1.0 / dt if dt > 0 else 0.0)

    def _process(self, frame: np.ndarray, ts: float) -> DetectionState:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, int(ts * 1000))

        s = DetectionState(ts=ts, fps=round(np.mean(self._fps_buf) if self._fps_buf else 0.0, 1))

        if not result.face_landmarks:
            self.eye_history.append(0)  # no face = treat as not closed
            self.eyes_closed_start = None
            return s

        s.face_detected = True
        h, w = frame.shape[:2]
        pts = [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]]

        ear = (_ear(pts, LEFT_EYE) + _ear(pts, RIGHT_EYE)) / 2.0
        mar = _mar(pts, MOUTH)
        s.ear = round(ear, 3)
        s.mar = round(mar, 3)

        eyes_closed = ear < EAR_THRESHOLD
        self.eye_history.append(1 if eyes_closed else 0)
        s.perclos = round(sum(self.eye_history) / len(self.eye_history), 3)

        # Eyes-closed duration (microsleep detection)
        now = ts
        if eyes_closed:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = now
            s.eyes_closed_for = round(now - self.eyes_closed_start, 2)
        else:
            self.eyes_closed_start = None
            s.eyes_closed_for = 0.0

        s.yawning = mar > MAR_THRESHOLD

        # Head pose from facial transformation matrix (pitch down = head dropping)
        if result.facial_transformation_matrixes:
            mat = np.array(result.facial_transformation_matrixes[0])
            # Extract pitch (rotation about X)
            pitch = math.degrees(math.atan2(-mat[2, 0], math.hypot(mat[2, 1], mat[2, 2])))
            s.head_down = pitch < -15  # tune

        # Alert logic — combine signals
        s.drowsy = (
            s.eyes_closed_for >= EYES_CLOSED_SECS
            or s.perclos >= PERCLOS_ALERT
            or (s.yawning and s.perclos >= 0.15)
        )
        if s.drowsy or s.head_down:
            s.alert_level = 2
        elif s.eyes_closed_for > 0.5 or s.yawning or s.perclos >= 0.20:
            s.alert_level = 1
        else:
            s.alert_level = 0

        return s

    def stop(self):
        self._stop.set()
        self.grabber.stop()
