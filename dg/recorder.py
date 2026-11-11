"""Recording one segment: camera numbers in, a CSV out, a live attention score on the side.

Kept free of any GUI so it can be tested without a camera: pass any object with a
`frames(seconds)` generator yielding (t, [face dicts], brightness), e.g. `sensing.Camera`.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .sensing import COLUMNS, HZ, SeatTracker, empty_numbers


def open_video(path: str | Path):
    """Start the lecture video in the computer's default player."""
    path = str(path)
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 (local file chosen by the researcher)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def live_score(rows: list[dict], seconds: float = 10.0) -> float | None:
    """Share of the last `seconds` spent facing the screen (the rule attention model's view),
    relative to the student's own median so far. None until 3 s of data exist."""
    if len(rows) < 3 * HZ:
        return None
    df = pd.DataFrame(rows[-int(seconds * HZ):])
    allr = pd.DataFrame(rows)
    seen = allr[allr.face > 0]
    if seen.empty:
        return 0.0
    yaw0, pitch0, gx0 = seen.yaw.median(), seen.pitch.median(), seen.gaze_x.median()
    on = (df.face > 0) & ((df.yaw - yaw0).abs() < 15) & ((df.pitch - pitch0).abs() < 12) & \
         ((df.gaze_x - gx0).abs() < 0.35)
    return float(on.mean())


class SegmentRecorder:
    """Records one person (laptop) for `seconds`, in a background thread."""

    def __init__(self, camera, seconds: float):
        self.camera, self.seconds = camera, seconds
        self.rows: list[dict] = []
        self.thread: threading.Thread | None = None
        self.error: Exception | None = None

    def _run(self):
        try:
            for t, faces, bright in self.camera.frames(self.seconds):
                if faces:  # one student: the largest face is theirs
                    f = max(faces, key=lambda d: d["size"])
                else:
                    f = empty_numbers() | {"brightness": bright}
                self.rows.append({"t": round(t, 3)} | {c: f.get(c, 0.0) for c in COLUMNS if c != "t"})
        except Exception as e:  # surfaced to the app
            self.error = e

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def done(self) -> bool:
        return self.thread is not None and not self.thread.is_alive()

    def stop(self):
        if hasattr(self.camera, "stop"):
            self.camera.stop()
        if self.thread:
            self.thread.join(timeout=5)

    def score(self) -> float | None:
        return live_score(self.rows)

    def save(self, path: Path) -> pd.DataFrame:
        df = pd.DataFrame(self.rows, columns=COLUMNS)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, float_format="%.4f")
        return df


def record_class(camera, seconds: float, tracker: SeatTracker) -> pd.DataFrame:
    """Whole class: every face matched to a seat, one row per seat per sample."""
    rows = []
    for t, faces, bright in camera.frames(seconds):
        pos = np.array([[f["cx"], f["cy"]] for f in faces]).reshape(-1, 2)
        seats = tracker.assign(t, pos)
        seen = set()
        for f, seat in zip(faces, seats):
            if seat is not None:
                rows.append({"t": round(t, 3), "seat": seat} | {c: f.get(c, 0.0) for c in COLUMNS if c != "t"})
                seen.add(seat)
        for seat in tracker.seat_ids():          # absent faces are recorded as absent
            if seat not in seen:
                rows.append({"t": round(t, 3), "seat": seat} | empty_numbers() | {"brightness": bright})
    return pd.DataFrame(rows, columns=["t", "seat"] + [c for c in COLUMNS if c != "t"])


def calibrate_seats(camera, seconds: float = 10.0, max_dist: float = 0.06) -> SeatTracker:
    """Learn seat positions while the class sits still and faces the front."""
    tracker = SeatTracker(calib_s=seconds - 0.5, max_dist=max_dist)
    for t, faces, _ in camera.frames(seconds):
        tracker.assign(t, np.array([[f["cx"], f["cy"]] for f in faces]).reshape(-1, 2))
    if tracker.seats is None:
        tracker._fit()
    return tracker


class FakeCamera:
    """A camera stand-in for tests and dry runs: replays simulated face numbers."""

    def __init__(self, faces_per_frame: list[list[dict]], hz: int = HZ):
        self.data, self.hz, self._stop = faces_per_frame, hz, False

    def stop(self):
        self._stop = True

    def frames(self, seconds: float | None = None):
        for i, faces in enumerate(self.data):
            t = i / self.hz
            if self._stop or (seconds is not None and t > seconds):
                break
            yield t, faces, 0.5
