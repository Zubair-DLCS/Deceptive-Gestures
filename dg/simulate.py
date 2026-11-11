"""Simulated students, so the whole pipeline can be built and tested before anyone is recorded.

EVERYTHING HERE IS A HAND-SET ASSUMPTION. The simulator writes data in exactly the format
the recording apps write, so the analysis can be checked end to end. Numbers it produces
show that the code works; they say nothing about how real students behave.

Assumptions built in (each one is a hypothesis the real study tests, not a fact):
  * genuine attention follows the lesson: gaze moves with the slides and reorients after
    slide changes, with small natural head corrections and irregular blinks;
  * genuine off-task rehearsal drifts away from the lesson, looks down and away, blinks more;
  * faked attention holds a stare at the screen: still, centred, weakly coupled to the lesson;
    better-informed fakers leak fewer look-aways, suppress blinks, and (with the dashboard)
    add deliberate movements that are not tied to the lesson;
  * faked distraction looks away a lot but still glances at slide changes to learn;
  * some honest students lower their gaze while listening, wear glasses, cover their face,
    or sit in poor light or at the back; these change the signals, not the attention.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import class_cards, individual_plan
from .sensing import COLUMNS, HZ

SEG_S = 180


def _stable(video: str) -> int:
    return sum(ord(ch) * (i + 1) for i, ch in enumerate(video))


def _drive(video: str, seconds: float = SEG_S) -> dict:
    # hash() is salted per process; use a stable seed so videos look the same in every run
    rng = np.random.default_rng(_stable(video))
    n = int(seconds * HZ)
    t = np.arange(n) / HZ
    events, e = [], rng.uniform(8, 20)
    while e < seconds - 5:
        events.append(round(float(e), 1))
        e += rng.uniform(15, 35)
    seg = np.searchsorted(events, t, side="right")
    tx = rng.uniform(-0.25, 0.25, len(events) + 1)
    ty = rng.uniform(-0.15, 0.15, len(events) + 1)
    gx = tx[seg] + 0.12 * ((t * 0.35) % 1 - 0.5)
    gy = ty[seg] + 0.04 * np.sin(t * 0.9)
    return dict(t=t, gx=gx, gy=gy, events=events)


def person(rng, pid: str, row: str = "front") -> dict:
    """A student's own traits. Group flags are what RQ3 checks for false accusations."""
    return dict(
        pid=pid,
        yaw0=rng.normal(0, 5), pitch0=rng.normal(-4, 4), gaze0=rng.normal(0, 0.05),
        eye0=rng.normal(0.28, 0.03), lag=rng.uniform(0.2, 0.5),
        glasses=bool(rng.random() < 0.3), covered=bool(rng.random() < 0.15),
        lowered_gaze=bool(rng.random() < 0.25), low_light=bool(rng.random() < 0.2),
        row=row,
        # how strongly this student follows a lesson, how well they act, how much they fidget
        follow_k=rng.uniform(0.55, 1.1), fake_follow=rng.uniform(0.05, 0.5),
        fidget=rng.uniform(0.6, 1.6),
    )


STATE = {  # follow = coupling to the lesson; away = look-aways per minute; blink = per minute
    ("A", None): dict(follow=0.85, away=1.5, away_s=1.2, blink=11, jitter=0.5, events=1.0, regular=0.0),
    ("B", None): dict(follow=0.10, away=8.0, away_s=4.0, blink=18, jitter=0.6, events=0.1, regular=0.0),
    ("D", None): dict(follow=0.35, away=5.0, away_s=3.0, blink=14, jitter=0.6, events=0.7, regular=0.0),
    ("C", "naive"): dict(follow=0.15, away=2.0, away_s=1.5, blink=16, jitter=0.20, events=0.15, regular=0.0),
    ("C", "told"): dict(follow=0.15, away=0.6, away_s=1.0, blink=6, jitter=0.15, events=0.15, regular=0.6),
    ("C", "dashboard"): dict(follow=0.20, away=0.4, away_s=1.0, blink=9, jitter=0.45, events=0.2, regular=0.3),
}


def behave(rng, p: dict, cond: str, block: str | None, drive: dict, noise: float = 1.0,
           face_p: float = 0.98) -> pd.DataFrame:
    """One person, one segment -> samples in the recording format."""
    s = STATE[(cond, block if cond == "C" else None)]
    n = len(drive["t"])
    t = drive["t"]
    lag = int(p["lag"] * HZ)
    lesson_x = np.roll(drive["gx"], lag)
    lesson_y = np.roll(drive["gy"], lag)
    follow = p["fake_follow"] if cond == "C" else min(s["follow"] * p["follow_k"], 0.95)
    s = dict(s, jitter=s["jitter"] * p["fidget"])
    # own wandering path (low-pass noise)
    own_x = np.convolve(rng.normal(0, 0.08, n), np.ones(15) / 15, "same") * 4
    own_y = np.convolve(rng.normal(0, 0.06, n), np.ones(15) / 15, "same") * 4
    if cond == "C":                       # a stare at the middle of the screen
        own_x *= 0.25
        own_y *= 0.25
    gx = follow * lesson_x + (1 - follow) * own_x
    gy = follow * lesson_y + (1 - follow) * own_y
    yaw = 9 * gx + np.cumsum(rng.normal(0, s["jitter"], n)) * 0.15
    yaw -= np.convolve(yaw, np.ones(50) / 50, "same") * 0.9          # mean-reverting head
    pitch = -6 * gy + np.cumsum(rng.normal(0, s["jitter"], n)) * 0.12
    pitch -= np.convolve(pitch, np.ones(50) / 50, "same") * 0.9
    if cond == "C" and block == "dashboard":                          # deliberate, uncoupled moves
        for k in rng.choice(n, size=int(n / HZ / 12), replace=False):
            yaw[k:k + 15] += rng.normal(0, 4)
    # reorienting to slide changes
    for e in drive["events"]:
        if rng.random() < s["events"]:
            k = int((e + p["lag"]) * HZ)
            yaw[k:k + 8] += rng.normal(0, 3)
            pitch[k:k + 8] += rng.normal(1.5, 1.5)
    # look-aways (down to notes, to the side)
    n_away = rng.poisson(s["away"] * n / HZ / 60)
    for k in rng.integers(0, n, n_away):
        d = int(rng.exponential(s["away_s"]) * HZ) + 3
        if rng.random() < 0.5:
            pitch[k:k + d] -= rng.uniform(15, 30)
        else:
            yaw[k:k + d] += rng.choice([-1, 1]) * rng.uniform(20, 40)
        gx[k:k + d] += rng.normal(0, 0.4)
    if p["lowered_gaze"] and cond in "AD":                            # honest habit, not inattention
        down = np.convolve(rng.random(n) < 0.02, np.ones(40), "same") > 0
        pitch[down] -= 18
    # blinks
    eye = np.full(n, p["eye0"]) + rng.normal(0, 0.01, n)
    rate = s["blink"] / 60 / HZ
    if s["regular"] > 0:
        period = max(int(1 / rate), 5)                    # deliberate, evenly spaced blinks
        starts = np.arange(int(rng.integers(0, period)), n, period)
        starts = starts + rng.integers(-3, 4, len(starts))
    else:
        starts = np.flatnonzero(rng.random(n) < rate)
    for k in starts:
        eye[max(k, 0): max(k, 0) + 2] *= 0.3
    mouth = np.abs(rng.normal(0.02, 0.01, n))
    if cond in "BC":                                                  # silent rehearsal
        mouth += 0.03 * (np.sin(t * 2 * np.pi * 1.5) > 0.6) * (rng.random() < 0.5)
    # sensing noise from the person's situation
    k_noise = noise * (2.0 if p["glasses"] else 1.0) * (1.5 if p["low_light"] else 1.0)
    gx = gx + rng.normal(0, 0.03 * k_noise, n) + p["gaze0"]
    gy = gy + rng.normal(0, 0.03 * k_noise, n)
    yaw = yaw + rng.normal(0, 0.8 * noise, n) + p["yaw0"]
    pitch = pitch + rng.normal(0, 0.8 * noise, n) + p["pitch0"]
    if p["glasses"]:
        eye -= 0.03
    found = rng.random(n) < face_p * (0.85 if p["covered"] else 1.0) * (0.92 if p["low_light"] else 1.0)
    found &= ~((np.abs(yaw) > 45) & (rng.random(n) < 0.7))
    bright = 0.25 if p["low_light"] else 0.55
    df = pd.DataFrame(dict(t=t, face=found.astype(float), gaze_x=gx, gaze_y=gy, yaw=yaw, pitch=pitch,
                           roll=rng.normal(0, 2, n), eye_open=np.clip(eye, 0.02, None),
                           mouth_open=mouth, cx=0.5, cy=0.5, size=0.35, brightness=bright))
    df.loc[~found, ["gaze_x", "gaze_y", "yaw", "pitch", "roll", "eye_open", "mouth_open"]] = 0.0
    return df[COLUMNS]


def responses(rng, cond: str) -> dict:
    learn = {"A": 0.78, "D": 0.68, "B": 0.30, "C": 0.30}[cond]
    words = {"A": 0.0, "D": 0.0, "B": 0.72, "C": 0.65}[cond]
    return dict(quiz=int(rng.binomial(5, learn)), recall=int(rng.binomial(12, words)) if words else 0)


def make_study1(root: Path, n: int = 40, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root) / "study1"
    videos = {f"V{i:02d}": _drive(f"V{i:02d}") for i in range(1, 11)}
    (root / "materials").mkdir(parents=True, exist_ok=True)
    json.dump({v: {"duration": SEG_S, "events": d["events"]} for v, d in videos.items()},
              open(root / "materials" / "videos.json", "w"), indent=1)
    for i in range(1, n + 1):
        pid = f"P{i:03d}"
        p = person(rng, pid)
        d = root / pid
        (d / "segments").mkdir(parents=True, exist_ok=True)
        resp = {}
        for seg in individual_plan(i):
            df = behave(rng, p, seg.condition, seg.block, videos[seg.video])
            df.to_csv(d / "segments" / f"seg{seg.index:02d}.csv", index=False, float_format="%.4f")
            resp[seg.index] = responses(rng, seg.condition)
        meta = dict(participant=pid, number=i, simulated=True,
                    groups=dict(glasses=p["glasses"], face_covering=p["covered"],
                                lowered_gaze_habit=4 if p["lowered_gaze"] else 2),
                    plan=[dict(index=s.index, block=s.block, condition=s.condition, video=s.video)
                          for s in individual_plan(i)])
        json.dump(meta, open(d / "session.json", "w"), indent=1)
        json.dump(resp, open(d / "responses.json", "w"), indent=1)
    return root


def make_study2(root: Path, classes: int = 3, size: int = 24, sessions: int = 2, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root) / "study2"
    rows = ["front"] * (size // 3) + ["middle"] * (size // 3) + ["back"] * (size - 2 * (size // 3))
    noise_of = {"front": 1.0, "middle": 1.6, "back": 2.4}
    face_of = {"front": 0.97, "middle": 0.93, "back": 0.85}
    for c in range(1, classes + 1):
        people = [person(rng, f"C{c}S{j + 1:02d}", rows[j]) for j in range(size)]
        seats = [f"S{j + 1:02d}" for j in range(size)]
        for s in range(1, sessions + 1):
            d = root / f"class{c}" / f"session{s}"
            d.mkdir(parents=True, exist_ok=True)
            cards = pd.DataFrame(class_cards(seats, seed=100 * c + s))
            cards.to_csv(d / "cards.csv", index=False)
            vids = {}
            resp = []
            for k in sorted(cards.segment.unique()):
                vid = f"K{c}{s}{k}"
                drive = _drive(vid)
                vids[vid] = {"duration": SEG_S, "events": drive["events"]}
                parts = []
                for j, (seat, p) in enumerate(zip(seats, people)):
                    cond = cards[(cards.segment == k) & (cards.seat == seat)].condition.item()
                    df = behave(rng, p, cond, "naive", drive, noise=noise_of[p["row"]], face_p=face_of[p["row"]])
                    df["seat"] = seat
                    df["cx"], df["cy"] = (j % 8) / 8 + 0.06, 0.2 + 0.25 * ["back", "middle", "front"].index(p["row"])
                    parts.append(df)
                    resp.append(dict(segment=int(k), seat=seat) | responses(rng, cond))
                pd.concat(parts).to_csv(d / f"segment{k:02d}.csv", index=False, float_format="%.4f")
            json.dump(dict(class_id=f"class{c}", session=s, simulated=True, videos=vids,
                           segments={int(k): f"K{c}{s}{k}" for k in sorted(cards.segment.unique())},
                           seats={seat: dict(row=p["row"], glasses=p["glasses"], face_covering=p["covered"],
                                             lowered_gaze_habit=4 if p["lowered_gaze"] else 2)
                                  for seat, p in zip(seats, people)}),
                      open(d / "session.json", "w"), indent=1)
            pd.DataFrame(resp).to_csv(d / "responses.csv", index=False)
    return root
