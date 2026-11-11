"""Study 2: a whole class in front of one camera.

  1. Learn the seats (students sit still and face the front for 10 s):
       python apps/class_app.py seats   --class class1 --session 1
     -> seats.json and seat_map.png (dots and seat codes only: no faces)
  2. Make the sealed cards (print, cut, hand out face down by seat):
       python apps/class_app.py cards   --class class1 --session 1
     -> cards.csv (keep it sealed until the end) and cards.html (to print)
  3. Record each segment while the lecture video plays on the projector:
       python apps/class_app.py record  --class class1 --session 1 --segment 1 --video K11
  4. After the session, type quiz and recall scores into responses.csv (template written in step 2).

Add --dry-run to any step to use simulated faces instead of a camera.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dg.protocol import INSTRUCTIONS, class_cards  # noqa: E402
from dg.recorder import FakeCamera, calibrate_seats, record_class  # noqa: E402
from dg.sensing import SeatTracker  # noqa: E402


def session_dir(a) -> Path:
    d = Path(a.data) / "study2" / a.class_id / f"session{a.session}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fake_class_camera(n: int = 18, seconds: float = 200, seed: int = 0):
    import numpy as np
    rng = np.random.default_rng(seed)
    pos = [((i % 6) / 6 + 0.08, 0.25 + 0.25 * (i // 6)) for i in range(n)]
    frames = []
    for k in range(int(seconds * 10)):
        faces = []
        for (x, y) in pos:
            if rng.random() < 0.95:
                faces.append(dict(face=1.0, gaze_x=rng.normal(0, .1), gaze_y=rng.normal(0, .1),
                                  yaw=rng.normal(0, 5), pitch=rng.normal(-3, 4), roll=0.0,
                                  eye_open=0.28, mouth_open=0.02, cx=x + rng.normal(0, .005),
                                  cy=y + rng.normal(0, .005), size=0.08, brightness=0.5))
        frames.append(faces)
    return FakeCamera(frames)


def camera(a):
    if a.dry_run:
        return fake_class_camera()
    from dg.sensing import Camera
    return Camera(a.camera, num_faces=a.max_faces)


def cmd_seats(a):
    d = session_dir(a)
    print("Ask everyone to sit still and look at the board for 10 seconds ...")
    tracker = calibrate_seats(camera(a), seconds=10.0)
    json.dump(tracker.to_dict(), open(d / "seats.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for sid, (x, y) in tracker.to_dict()["seats"].items():
        ax.scatter(x, y, s=300, color="#2a78d6")
        ax.text(x, y + 0.035, sid, ha="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)                       # picture coordinates: top of the picture at the top
    ax.set_title("Seats as the camera sees them (front of the room at the bottom)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(d / "seat_map.png", dpi=120, bbox_inches="tight")
    print(f"{len(tracker.seat_ids())} seats found. Write each seat code on its desk using {d / 'seat_map.png'}.")


def cmd_cards(a):
    d = session_dir(a)
    seats = SeatTracker.from_dict(json.load(open(d / "seats.json"))).seat_ids()
    cards = pd.DataFrame(class_cards(seats, seed=a.seed))
    cards.to_csv(d / "cards.csv", index=False)
    resp = cards[["segment", "seat"]].assign(quiz="", recall="")
    resp.to_csv(d / "responses.csv", index=False)
    html = ["<html><meta charset='utf-8'><style>.c{border:1px dashed #999;padding:14px;margin:6px;"
            "width:300px;display:inline-block;vertical-align:top;font-family:sans-serif}"
            "h3{margin:0 0 6px}</style><body>"]
    for _, c in cards.iterrows():
        html.append(f"<div class='c'><h3>Seat {c.seat} &middot; part {c.segment}</h3>"
                    f"<p>{INSTRUCTIONS[c.condition]}</p>"
                    + (f"<p><b>Words:</b> {c.words}</p>" if c.words else "") + "</div>")
    html.append("</body></html>")
    (d / "cards.html").write_text("\n".join(html), encoding="utf-8")
    print(f"cards for {len(seats)} seats x {cards.segment.nunique()} parts -> {d / 'cards.html'}")
    print("Keep cards.csv closed until all responses are entered.")


def cmd_record(a):
    d = session_dir(a)
    tracker = SeatTracker.from_dict(json.load(open(d / "seats.json")))
    meta_f = d / "session.json"
    meta = json.load(open(meta_f)) if meta_f.exists() else dict(
        class_id=a.class_id, session=a.session, videos={}, segments={}, seats={s: {} for s in tracker.seat_ids()})
    videos = json.load(open(a.videos, encoding="utf-8"))
    v = videos[a.video]
    print(f"Recording part {a.segment} ({a.video}, {v['duration']} s). Start the video on the projector now.")
    t0 = time.time()
    df = record_class(camera(a), v["duration"], tracker)
    df.to_csv(d / f"segment{a.segment:02d}.csv", index=False, float_format="%.4f")
    meta["videos"][a.video] = {"duration": v["duration"], "events": v.get("events", [])}
    meta["segments"][str(a.segment)] = a.video
    json.dump(meta, open(meta_f, "w"), indent=1)
    seen = df[df.face > 0].groupby("seat").size() / max(df.t.nunique(), 1)
    print(f"done in {time.time() - t0:.0f} s; faces found per seat: median {seen.median():.0%}, "
          f"worst {seen.min():.0%} ({seen.idxmin() if len(seen) else '-'})")
    print("Add each seat's groups (row, glasses, face_covering, lowered_gaze_habit 1-5) to session.json.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["seats", "cards", "record"])
    ap.add_argument("--class", dest="class_id", required=True)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--segment", type=int)
    ap.add_argument("--video")
    ap.add_argument("--videos", default="materials/videos.json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--max-faces", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.step == "record" and (a.segment is None or a.video is None):
        ap.error("record needs --segment and --video")
    {"seats": cmd_seats, "cards": cmd_cards, "record": cmd_record}[a.step](a)


if __name__ == "__main__":
    main()
