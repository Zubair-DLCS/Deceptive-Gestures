"""Webcam frames -> numbers. No image is ever stored.

Each frame is reduced to a few numbers per face (head pose, gaze, eye and mouth
opening, where the face sits in the picture) and then dropped. Works for one
student at a laptop (`num_faces=1`) and for a whole class in front of one camera
(`num_faces` up to 40, faces matched to seats by `SeatTracker`).

Uses MediaPipe Face Landmarker (478 landmarks incl. irises). The model file
(~3.7 MB) is downloaded once to ~/.cache/deceptive_gestures/.
"""
from __future__ import annotations

import math
import time
import urllib.request
from pathlib import Path

import numpy as np

HZ = 10                      # samples per second kept
COLUMNS = ["t", "face", "gaze_x", "gaze_y", "yaw", "pitch", "roll",
           "eye_open", "mouth_open", "cx", "cy", "size", "brightness"]
SIGNALS = ["gaze_x", "gaze_y", "yaw", "pitch", "roll", "eye_open", "mouth_open"]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path.home() / ".cache" / "deceptive_gestures" / "face_landmarker.task"

# MediaPipe face-mesh indices: outer corner, inner corner, upper lid, lower lid, iris centre
EYES = {"left": (33, 133, 159, 145, 468), "right": (263, 362, 386, 374, 473)}
NOSE, LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD, CHIN = 1, 234, 454, 10, 152
UPPER_LIP, LOWER_LIP, MOUTH_L, MOUTH_R = 13, 14, 61, 291


def _eye(pts: np.ndarray, idx) -> tuple[float, float, float]:
    outer, inner, top, bottom, iris = (pts[i] for i in idx)
    width = np.linalg.norm(outer - inner) + 1e-6
    height = np.linalg.norm(top - bottom)
    left_x, right_x = min(outer[0], inner[0]), max(outer[0], inner[0])
    gx = 2 * (iris[0] - left_x) / (right_x - left_x + 1e-6) - 1
    mid_y = (top[1] + bottom[1]) / 2
    gy = 2 * (iris[1] - mid_y) / (height + 0.3 * width)
    return float(gx), float(gy), float(height / width)


def head_angles(matrix, pts: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch, roll in degrees from MediaPipe's face transform (or a landmark proxy)."""
    if matrix is not None:
        R = np.asarray(matrix)[:3, :3]
        yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])))
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        return yaw, pitch, roll
    face_w = np.linalg.norm(pts[RIGHT_CHEEK] - pts[LEFT_CHEEK]) + 1e-6
    face_h = np.linalg.norm(pts[CHIN] - pts[FOREHEAD]) + 1e-6
    mid = (pts[RIGHT_CHEEK] + pts[LEFT_CHEEK]) / 2
    yaw = math.degrees(math.asin(np.clip(2 * (pts[NOSE][0] - mid[0]) / face_w, -1, 1)))
    pitch = math.degrees(math.asin(np.clip(2 * (pts[NOSE][1] - (pts[FOREHEAD][1] + pts[CHIN][1]) / 2) / face_h, -1, 1)))
    d = pts[RIGHT_CHEEK] - pts[LEFT_CHEEK]
    roll = math.degrees(math.atan2(d[1], d[0]))
    return yaw, pitch, roll


def face_numbers(pts: np.ndarray, matrix=None) -> dict:
    """pts: (478, 2) normalised landmark x, y for one face -> a dict of numbers."""
    l, r = _eye(pts, EYES["left"]), _eye(pts, EYES["right"])
    yaw, pitch, roll = head_angles(matrix, pts)
    mouth_w = np.linalg.norm(pts[MOUTH_R] - pts[MOUTH_L]) + 1e-6
    xs, ys = pts[:, 0], pts[:, 1]
    return dict(face=1.0, gaze_x=(l[0] + r[0]) / 2, gaze_y=(l[1] + r[1]) / 2, yaw=yaw, pitch=pitch,
                roll=roll, eye_open=(l[2] + r[2]) / 2,
                mouth_open=float(np.linalg.norm(pts[LOWER_LIP] - pts[UPPER_LIP]) / mouth_w),
                cx=float(xs.mean()), cy=float(ys.mean()), size=float(ys.max() - ys.min()))


def empty_numbers() -> dict:
    return {c: (0.0 if c != "t" else np.nan) for c in COLUMNS if c != "t"} | {"face": 0.0}


class SeatTracker:
    """Matches faces to fixed seats across frames by face position.

    Seats are learned from the first `calib_s` seconds (students sit still), then
    each frame's faces are matched to the nearest free seat within `max_dist`
    (in image-normalised units). Seat ids S01, S02, ... run front-to-back by
    picture height, then left-to-right, so the teacher can map them to desks.
    """

    def __init__(self, calib_s: float = 5.0, max_dist: float = 0.06):
        self.calib_s, self.max_dist = calib_s, max_dist
        self.calib: list[np.ndarray] = []
        self.seats: np.ndarray | None = None

    def to_dict(self) -> dict:
        return dict(max_dist=self.max_dist, seats={sid: [float(x), float(y)]
                                                   for sid, (x, y) in zip(self.seat_ids(), self.seats)})

    @classmethod
    def from_dict(cls, d: dict) -> "SeatTracker":
        tr = cls(calib_s=0.0, max_dist=d.get("max_dist", 0.06))
        tr.seats = np.array([d["seats"][k] for k in sorted(d["seats"])], float).reshape(-1, 2)
        return tr

    def seat_ids(self) -> list[str]:
        return [f"S{i + 1:02d}" for i in range(0 if self.seats is None else len(self.seats))]

    def _fit(self):
        pts = np.concatenate(self.calib) if self.calib else np.zeros((0, 2))
        centres: list[np.ndarray] = []
        for p in pts:                           # greedy clustering, enough for still students
            for k, c in enumerate(centres):
                if np.linalg.norm(c[0] - p) < self.max_dist:
                    centres[k] = (c[0] * c[1] + p) / (c[1] + 1), c[1] + 1
                    break
            else:
                centres.append((p, 1))
        min_hits = max(3, int(0.3 * len(self.calib)))
        keep = np.array([c[0] for c in centres if c[1] >= min_hits]).reshape(-1, 2)
        # front of the room = larger faces lower in the picture; sort by y descending, then x
        order = np.lexsort((keep[:, 0], -np.round(keep[:, 1], 1)))
        self.seats = keep[order]

    def assign(self, t: float, positions: np.ndarray) -> list[str | None]:
        """positions: (F, 2) face centres this frame -> a seat id (or None) per face."""
        if self.seats is None:
            self.calib.append(np.asarray(positions).reshape(-1, 2))
            if t >= self.calib_s:
                self._fit()
            return [None] * len(positions)
        out: list[str | None] = [None] * len(positions)
        if len(positions) == 0 or len(self.seats) == 0:
            return out
        d = np.linalg.norm(np.asarray(positions)[:, None, :] - self.seats[None, :, :], axis=2)
        taken = set()
        for flat in np.argsort(d, axis=None):
            f, s = divmod(int(flat), d.shape[1])
            if out[f] is None and s not in taken and d[f, s] <= self.max_dist:
                out[f] = f"S{s + 1:02d}"
                taken.add(s)
        return out


def landmarker(num_faces: int = 1):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading face model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO, num_faces=num_faces,
        min_face_detection_confidence=0.4, output_facial_transformation_matrixes=True)
    return mp, vision.FaceLandmarker.create_from_options(opts)


class Camera:
    """Reads a webcam (index) or a video file, keeps HZ samples a second, and yields
    (t, [face dicts]) with frames dropped immediately. `stop()` ends the loop."""

    def __init__(self, source: int | str = 0, num_faces: int = 1):
        self.source, self.num_faces, self._stop = source, num_faces, False

    def stop(self):
        self._stop = True

    def frames(self, seconds: float | None = None):
        import cv2

        mp, lm = landmarker(self.num_faces)
        cap = cv2.VideoCapture(self.source)
        live = isinstance(self.source, int)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        t0, i, next_t = time.monotonic(), 0, 0.0
        try:
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    break
                t = (time.monotonic() - t0) if live else i / fps
                i += 1
                if t + 1e-9 < next_t:
                    continue
                next_t += 1.0 / HZ
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bright = float(rgb.mean() / 255.0)
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
                del frame, rgb                       # the picture is gone from here on
                faces = []
                for k, lms in enumerate(res.face_landmarks or []):
                    pts = np.array([[p.x, p.y] for p in lms], np.float32)
                    mats = res.facial_transformation_matrixes
                    faces.append(face_numbers(pts, mats[k] if mats else None) | {"brightness": bright})
                yield t, faces, bright
                if seconds is not None and t >= seconds:
                    break
        finally:
            cap.release()
            lm.close()
