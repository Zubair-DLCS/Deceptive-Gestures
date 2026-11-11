"""Load recorded (or simulated) sessions into one table of 10-second windows.

Folder layout written by the apps (and by dg.simulate):

  study1/materials/videos.json            {video: {duration, events, ...}}
  study1/P001/session.json                participant, groups, plan (index, block, condition, video)
  study1/P001/segments/seg01.csv          face numbers, one row per sample
  study1/P001/responses.json              {segment index: {quiz, recall}}

  study2/<class>/<session>/session.json   videos, segments {k: video}, seats {seat: groups}
  study2/<class>/<session>/segmentKK.csv  face numbers with a `seat` column
  study2/<class>/<session>/cards.csv      segment, faker_share, seat, condition
  study2/<class>/<session>/responses.csv  segment, seat, quiz, recall
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (centre, person_baseline, reference, synchrony, to_grid,
                       window_table, zsignals)

QUIZ_PASS = 3      # of 5: the student followed the lesson
RECALL_PASS = 6    # of 12: the student really rehearsed the words
LOW_LIGHT = 0.35   # mean frame brightness below this counts as poor light


def proof_ok(condition: str, quiz: int, recall: int) -> bool:
    """A segment counts only if what the student did matches the condition."""
    if condition in ("A", "D"):
        return quiz >= QUIZ_PASS
    return recall >= RECALL_PASS


def _groups(g: dict, brightness: float) -> dict:
    return dict(glasses=bool(g.get("glasses", False)),
                face_covering=bool(g.get("face_covering", False)),
                lowered_gaze=int(g.get("lowered_gaze_habit", 0)) >= 4,
                low_light=bool(brightness < LOW_LIGHT))


def load_study1(root: Path) -> pd.DataFrame:
    root = Path(root)
    videos = json.load(open(root / "materials" / "videos.json"))
    people = []
    for d in sorted(p for p in root.iterdir() if (p / "session.json").exists()):
        meta = json.load(open(d / "session.json"))
        resp = json.load(open(d / "responses.json")) if (d / "responses.json").exists() else {}
        segs = []
        for s in meta["plan"]:
            f = d / "segments" / f"seg{s['index']:02d}.csv"
            if not f.exists():
                continue
            raw = pd.read_csv(f)
            grid = to_grid(raw, videos[s["video"]]["duration"])
            r = resp.get(str(s["index"]), {"quiz": 0, "recall": 0})
            segs.append(dict(s, grid=grid, quiz=r["quiz"], recall=r["recall"],
                             brightness=float(raw["brightness"].mean()) if "brightness" in raw else 0.5))
        if not segs:
            continue
        base = person_baseline([s["grid"] for s in segs])
        for s in segs:
            s["c"] = centre(s["grid"], base)
            s["z"] = zsignals(s["c"])
        people.append(dict(person=meta["participant"], groups=meta.get("groups", {}), segs=segs, base=base))

    by_video: dict[str, list] = {}
    for p in people:
        for s in p["segs"]:
            by_video.setdefault(s["video"], []).append((p["person"], s["z"]))

    rows = []
    for p in people:
        for s in p["segs"]:
            others = [g for who, g in by_video[s["video"]] if who != p["person"]]
            sync = synchrony(s["c"], reference(others))[0] if others else None
            w = window_table(s["c"], p["base"]["eye_open"], videos[s["video"]]["events"], sync)
            w = w.assign(study=1, person=p["person"], segment=s["index"], block=s["block"],
                         condition=s["condition"], video=s["video"], quiz=s["quiz"], recall=s["recall"],
                         valid=proof_ok(s["condition"], s["quiz"], s["recall"]),
                         **_groups(p["groups"], s["brightness"]), row="laptop")
            rows.append(w)
    return pd.concat(rows, ignore_index=True)


def load_study2(root: Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for sess in sorted(root.glob("*/*/session.json")):
        d = sess.parent
        meta = json.load(open(sess))
        cards = pd.read_csv(d / "cards.csv")
        resp = pd.read_csv(d / "responses.csv")
        segs = {}
        for k, video in meta["segments"].items():
            f = d / f"segment{int(k):02d}.csv"
            if f.exists():
                raw = pd.read_csv(f)
                dur = meta["videos"][video]["duration"]
                segs[int(k)] = (video, {seat: (to_grid(g, dur), float(g["brightness"].mean()))
                                        for seat, g in raw.groupby("seat")})
        seats = sorted({seat for _, gs in segs.values() for seat in gs})
        base = {seat: person_baseline([gs[seat][0] for _, gs in segs.values() if seat in gs]) for seat in seats}
        for k, (video, gs) in segs.items():
            centred = {seat: centre(g, base[seat]) for seat, (g, _) in gs.items()}
            zs = {seat: zsignals(g) for seat, g in centred.items()}
            for seat, g in centred.items():
                others = [z for s2, z in zs.items() if s2 != seat]
                sync = synchrony(g, reference(others))[0] if others else None
                card = cards[(cards.segment == k) & (cards.seat == seat)]
                if card.empty:
                    continue
                cond = card.condition.item()
                r = resp[(resp.segment == k) & (resp.seat == seat)]
                quiz, recall = (int(r.quiz.item()), int(r.recall.item())) if len(r) else (0, 0)
                w = window_table(g, base[seat]["eye_open"], meta["videos"][video]["events"], sync)
                sg = meta["seats"].get(seat, {})
                w = w.assign(study=2, person=f"{meta['class_id']}-{seat}", segment=k,
                             session=f"{meta['class_id']}/s{meta['session']}", block="naive",
                             condition=cond, video=video, faker_share=float(card.faker_share.item()),
                             quiz=quiz, recall=recall, valid=proof_ok(cond, quiz, recall),
                             **_groups(sg, gs[seat][1]), row=sg.get("row", "unknown"))
                rows.append(w)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
