import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dg import analysis, data, features, models, protocol, recorder, sensing, simulate  # noqa: E402


# ---- protocol -------------------------------------------------------------------------------
def test_latin_square_is_balanced():
    sq = protocol.balanced_latin_square(4)
    assert all(sorted(r) == [0, 1, 2, 3] for r in sq)
    assert all(sorted(col) == [0, 1, 2, 3] for col in zip(*sq))
    pairs = Counter((r[i], r[i + 1]) for r in sq for i in range(3))
    assert set(pairs.values()) == {1} and len(pairs) == 12      # every ordered pair once


def test_individual_plan_shape():
    for n in range(1, 13):
        plan = protocol.individual_plan(n)
        assert [s.block for s in plan] == ["naive"] * 4 + ["told"] * 3 + ["dashboard"] * 3
        assert sorted(s.condition for s in plan[:4]) == list("ABCD")
        assert sorted(s.condition for s in plan[4:7]) == list("ACD")
        assert [s.video for s in plan] == [f"V{i:02d}" for i in range(1, 11)]
        assert all(len(s.words) == 12 for s in plan if s.condition in "BC")
        assert all(not s.words for s in plan if s.condition in "AD")
    firsts = Counter(protocol.individual_plan(n)[0].condition for n in range(1, 9))
    assert set(firsts) == set("ABCD")


def test_word_lists_do_not_overlap():
    lists = protocol.word_lists()
    flat = [w for l in lists for w in l]
    assert len(flat) == len(set(flat)) == 120


def test_class_cards_hit_the_faker_share():
    seats = [f"S{i:02d}" for i in range(1, 21)]
    cards = pd.DataFrame(protocol.class_cards(seats))
    assert sorted(cards.segment.unique()) == list(range(1, 7))
    for k, g in cards.groupby("segment"):
        assert len(g) == 20
        assert (g.condition == "C").mean() == pytest.approx(g.faker_share.iloc[0])
        if g.faker_share.iloc[0] == 0:
            assert not (g.condition.isin(["C", "D"])).any()


# ---- sensing --------------------------------------------------------------------------------
def test_face_numbers_on_a_flat_face():
    pts = np.zeros((478, 2), np.float32) + 0.5
    for side, (o, i, top, bot, iris) in sensing.EYES.items():
        x = 0.42 if side == "left" else 0.58
        pts[o], pts[i] = (x - 0.03, 0.45), (x + 0.03, 0.45)
        pts[top], pts[bot], pts[iris] = (x, 0.44), (x, 0.46), (x, 0.45)
    pts[sensing.LEFT_CHEEK], pts[sensing.RIGHT_CHEEK] = (0.35, 0.5), (0.65, 0.5)
    pts[sensing.FOREHEAD], pts[sensing.CHIN], pts[sensing.NOSE] = (0.5, 0.3), (0.5, 0.7), (0.5, 0.5)
    f = sensing.face_numbers(pts)
    assert abs(f["gaze_x"]) < 1e-3 and abs(f["yaw"]) < 1e-3 and abs(f["roll"]) < 1e-3
    assert f["eye_open"] == pytest.approx(0.02 / 0.06, rel=1e-3)


def test_seat_tracker_keeps_seats_when_faces_reorder():
    tr = sensing.SeatTracker(calib_s=1.0)
    seats = np.array([[0.2, 0.8], [0.6, 0.8], [0.4, 0.3]])
    for k in range(12):
        tr.assign(k / 10, seats + np.random.default_rng(k).normal(0, 0.003, seats.shape))
    assert len(tr.seat_ids()) == 3
    a = tr.assign(2.0, seats[[2, 0, 1]])
    b = tr.assign(2.1, seats)
    assert a == [b[2], b[0], b[1]] and len(set(b)) == 3
    assert b[2] == "S03"                                      # the back-row face gets the last code
    tr2 = sensing.SeatTracker.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert tr2.assign(0, seats) == b


# ---- features -------------------------------------------------------------------------------
def test_grid_marks_missing_faces():
    df = pd.DataFrame({c: 0.0 for c in sensing.COLUMNS} | {"t": np.arange(100) / 10})
    df["face"] = 1.0
    df.loc[40:59, "face"] = 0.0
    g = features.to_grid(df, 10)
    assert len(g) == 100 and g.face.sum() == 80
    assert g.yaw.isna().sum() == 20            # a 2 s absence stays missing
    df.loc[40:59, "face"] = 1.0
    df.loc[40:43, "face"] = 0.0                # a 0.4 s blink-length gap is filled
    assert features.to_grid(df, 10).yaw.notna().all()


def test_lag_is_recovered():
    rng = np.random.default_rng(1)
    ref = np.convolve(rng.normal(size=1800), np.ones(12) / 12, "same")
    for lag in (-9, 0, 6):
        x = np.roll(ref, lag) + rng.normal(0, 0.05, ref.size)
        assert features.best_lag(x, ref) == lag


def test_window_table_has_every_family():
    rng = np.random.default_rng(0)
    p = simulate.person(rng, "t")
    drive = simulate._drive("V01")
    g = features.centre(features.to_grid(simulate.behave(rng, p, "A", "naive", drive), 180))
    w = features.window_table(g, 0.28, drive["events"], np.zeros(18))
    assert len(w) == 18
    assert all(c in w for c in features.ATT + features.CPL)
    assert any(c.startswith("dyn_") for c in w)


# ---- recorder -------------------------------------------------------------------------------
def test_segment_recorder_writes_the_recording_format(tmp_path):
    rng = np.random.default_rng(0)
    df = simulate.behave(rng, simulate.person(rng, "r"), "C", "told", simulate._drive("V02"))
    frames = [[r] if r["face"] > 0 else [] for r in df.drop(columns="t").to_dict("records")]
    rec = recorder.SegmentRecorder(recorder.FakeCamera(frames), seconds=60).start()
    rec.thread.join(5)
    out = rec.save(tmp_path / "seg.csv")
    assert list(out.columns) == sensing.COLUMNS
    assert 590 <= len(out) <= 602
    assert 0.5 < rec.score() <= 1.0                       # a faker facing the screen scores high


def test_class_recording_tracks_every_seat():
    from apps.class_app import fake_class_camera
    tracker = recorder.calibrate_seats(fake_class_camera(n=12, seconds=12), seconds=10)
    assert len(tracker.seat_ids()) == 12
    df = recorder.record_class(fake_class_camera(n=12, seconds=20, seed=3), 15, tracker)
    per_seat = df.groupby("seat").t.nunique()
    assert len(per_seat) == 12 and per_seat.min() >= 140


# ---- end to end -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    simulate.make_study1(root, n=8)
    simulate.make_study2(root, classes=1, size=12, sessions=1)
    w1 = data.load_study1(root / "study1")
    w2 = data.load_study2(root / "study2")
    return w1, w2


def test_loading_counts_and_proofs(small_study):
    w1, w2 = small_study
    assert w1.person.nunique() == 8 and len(w1) == 8 * 10 * 18
    assert set(w1.condition) == set("ABCD") and 0.5 < w1.valid.mean() <= 1
    assert w2.person.nunique() == 12 and w2.cpl_synchrony.notna().mean() > 0.9


def test_scoring_and_analysis_run(small_study):
    w1, w2 = small_study
    s1 = models.score_study1(w1)
    s2 = models.score_transfer(w1, w2)
    att = analysis.rq1_attack(s1)
    det = analysis.detection_rates(s1, ["block"])
    col = analysis.rq3_collateral(s1)
    assert {"rule", "learned"} == set(att.model)
    assert det.auroc.between(0, 1).all()
    assert set(col.measure) == {"accused of faking", "rated not attentive"}
    assert "flag_combined" in s2
    h3 = analysis.h3_synchrony(s1)
    assert h3["students"] == 8


def test_camera_loop_samples_at_10hz_and_keeps_no_frames(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = str(tmp_path / "clip.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for _ in range(90):                                  # 3 s at 30 fps
        vw.write(np.full((120, 160, 3), 100, np.uint8))
    vw.release()

    class Pt:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class Res:
        face_landmarks = [[Pt(0.5 + 0.001 * (i % 7), 0.5 + 0.001 * (i % 5)) for i in range(478)]]
        facial_transformation_matrixes = [np.eye(4)]

    class Landmarker:
        def detect_for_video(self, image, ms):
            return Res()

        def close(self):
            pass

    class MP:
        class ImageFormat:
            SRGB = 1

        @staticmethod
        def Image(image_format, data):
            return data

    monkeypatch.setattr(sensing, "landmarker", lambda n: (MP, Landmarker()))
    out = list(sensing.Camera(path).frames())
    assert 29 <= len(out) <= 31                          # 10 samples a second from 30 fps
    t, faces, bright = out[5]
    assert set(faces[0]) >= {"yaw", "pitch", "gaze_x", "eye_open", "cx", "brightness"}
    assert bright == pytest.approx(100 / 255, abs=0.04)          # video compression shifts it a little
    assert not any(isinstance(v, np.ndarray) for v in faces[0].values())   # numbers only
