"""Study 1: one student at a laptop. Walks the session from consent to the last quiz.

    python apps/session_app.py                      # real webcam
    python apps/session_app.py --dry-run            # no camera: replays a simulated student

For each of the 10 segments: instruction -> (words to rehearse, 30 s) -> the lecture video
plays in the computer's own player while the webcam is read -> word recall -> 5-question quiz.
Only numbers are saved, to data/study1/P###/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dg.protocol import CONDITIONS, individual_plan  # noqa: E402
from dg.recorder import FakeCamera, SegmentRecorder, open_video  # noqa: E402

WORDS_SECONDS = 30
BIG = ("Segoe UI", 16)
MID = ("Segoe UI", 12)


def score_recall(typed: str, words: list[str]) -> int:
    said = {w.strip(" ,.;").lower() for w in typed.replace(",", " ").split()}
    return sum(w.lower() in said for w in words)


class SessionApp:
    def __init__(self, root: tk.Misc, args):
        """`root` may be the main Tk window or a Toplevel opened by the launcher."""
        self.root, self.args = root, args
        self.videos = json.load(open(args.videos, encoding="utf-8"))
        self.base = Path(args.videos).parent
        root.title("Deceptive Gestures: study session")
        root.geometry("900x620")
        self.frame = None
        self.setup_screen()

    # ---- helpers -------------------------------------------------------------
    def screen(self):
        if self.frame:
            self.frame.destroy()
        self.frame = ttk.Frame(self.root, padding=30)
        self.frame.pack(fill="both", expand=True)
        return self.frame

    def text(self, parent, words, font=MID, **kw):
        lab = ttk.Label(parent, text=words, font=font, wraplength=820, justify="left", **kw)
        lab.pack(anchor="w", pady=6)
        return lab

    def camera(self):
        if self.args.dry_run:
            from dg import simulate
            import numpy as np
            rng = np.random.default_rng(int(time.time()))
            p = simulate.person(rng, "dry")
            seg = self.plan[self.k]
            df = simulate.behave(rng, p, seg.condition, seg.block, simulate._drive(seg.video))
            frames = [[r._asdict() | {}] if r.face > 0 else [] for r in df.drop(columns="t").itertuples(index=False)]
            return FakeCamera(frames)
        from dg.sensing import Camera
        return Camera(self.args.camera, num_faces=1)

    # ---- screens -------------------------------------------------------------
    def setup_screen(self):
        f = self.screen()
        self.text(f, "New session", BIG)
        self.v_num = tk.StringVar()
        self.v_glasses = tk.BooleanVar()
        self.v_cover = tk.BooleanVar()
        self.v_down = tk.IntVar(value=3)
        self.v_consent = tk.BooleanVar()
        self.v_minor = tk.BooleanVar()
        self.v_parent = tk.BooleanVar()
        row = ttk.Frame(f); row.pack(anchor="w", pady=4)
        ttk.Label(row, text="Participant number (1, 2, 3 ...):", font=MID).pack(side="left")
        ttk.Entry(row, textvariable=self.v_num, width=8, font=MID).pack(side="left", padx=8)
        ttk.Checkbutton(f, text="Wears glasses today", variable=self.v_glasses).pack(anchor="w")
        ttk.Checkbutton(f, text="Face partly covered today (student's own choice to report)",
                        variable=self.v_cover).pack(anchor="w")
        row = ttk.Frame(f); row.pack(anchor="w", pady=4)
        ttk.Label(row, text="'When I listen to a teacher, I usually look down.'  1 = never ... 5 = always:",
                  font=MID).pack(side="left")
        ttk.Spinbox(row, from_=1, to=5, textvariable=self.v_down, width=3).pack(side="left", padx=8)
        ttk.Separator(f).pack(fill="x", pady=10)
        ttk.Checkbutton(f, text="The student has read the information sheet and signed the consent form",
                        variable=self.v_consent).pack(anchor="w")
        ttk.Checkbutton(f, text="The student is under 18", variable=self.v_minor).pack(anchor="w")
        ttk.Checkbutton(f, text="... and a parent has signed the consent form", variable=self.v_parent).pack(anchor="w")
        ttk.Button(f, text="Start session", command=self.begin).pack(anchor="w", pady=16)

    def begin(self):
        try:
            n = int(self.v_num.get())
        except ValueError:
            return messagebox.showerror("Missing", "Enter the participant number.")
        if not self.v_consent.get() or (self.v_minor.get() and not self.v_parent.get()):
            return messagebox.showerror("Consent", "No recording without the consent forms.")
        self.pid = f"P{n:03d}"
        self.dir = Path(self.args.data) / "study1" / self.pid
        if (self.dir / "session.json").exists() and not messagebox.askyesno(
                "Exists", f"{self.pid} already has data. Continue and overwrite?"):
            return
        self.plan = individual_plan(n, sorted(self.videos)[:10])
        self.meta = dict(participant=self.pid, number=n, started=time.strftime("%Y-%m-%d %H:%M"),
                         dry_run=self.args.dry_run,
                         groups=dict(glasses=self.v_glasses.get(), face_covering=self.v_cover.get(),
                                     lowered_gaze_habit=self.v_down.get()),
                         plan=[dict(index=s.index, block=s.block, condition=s.condition, video=s.video)
                               for s in self.plan])
        self.dir.mkdir(parents=True, exist_ok=True)
        snap = self.dir.parent / "materials" / "videos.json"     # the event times the analysis uses
        snap.parent.mkdir(parents=True, exist_ok=True)
        if not snap.exists():
            json.dump(self.videos, open(snap, "w", encoding="utf-8"), indent=1)
        json.dump(self.meta, open(self.dir / "session.json", "w"), indent=1)
        self.responses = {}
        self.k = 0
        self.instruction_screen()

    def instruction_screen(self):
        seg = self.plan[self.k]
        f = self.screen()
        self.text(f, f"Part {seg.index} of {len(self.plan)}", BIG)
        self.text(f, seg.instruction(), ("Segoe UI", 15))
        ttk.Button(f, text="Continue", command=self.words_screen if seg.needs_words else self.video_screen).pack(
            anchor="w", pady=20)

    def words_screen(self):
        seg = self.plan[self.k]
        f = self.screen()
        self.text(f, "Remember these words. Keep rehearsing them silently.", BIG)
        self.text(f, "     ".join(seg.words), ("Segoe UI", 20))
        left = self.text(f, "")

        def tick(s=WORDS_SECONDS):
            if s <= 0:
                return self.video_screen()
            left.config(text=f"{s} s")
            self.root.after(1000, tick, s - 1)
        tick()

    def video_screen(self):
        seg = self.plan[self.k]
        v = self.videos[seg.video]
        f = self.screen()
        self.text(f, "The lecture will now play. Sit as you normally would.", BIG)
        self.status = self.text(f, "")
        self.live = None
        self.rec = SegmentRecorder(self.camera(), v["duration"]).start()
        self.video_started = time.time()
        if not self.args.dry_run:
            open_video(self.base / v["file"])
        if seg.block == "dashboard" and seg.condition in "CD":
            self.live = tk.Toplevel(self.root)
            self.live.title("Attention score")
            self.live.attributes("-topmost", True)
            self.live.geometry("260x110+20+20")
            self.live_lab = ttk.Label(self.live, text="...", font=("Segoe UI", 28))
            self.live_lab.pack(expand=True)
        self.poll()

    def poll(self):
        rec = self.rec
        if rec.error:
            messagebox.showerror("Camera", f"Camera problem: {rec.error}")
        elapsed = time.time() - self.video_started
        dur = self.videos[self.plan[self.k].video]["duration"]
        self.status.config(text=f"Recording ... {max(0, int(dur - elapsed))} s left")
        if self.live is not None:
            s = rec.score()
            self.live_lab.config(text="..." if s is None else f"{100 * s:.0f}%")
        if rec.done() or elapsed > dur + 5:
            rec.stop()
            rec.save(self.dir / "segments" / f"seg{self.plan[self.k].index:02d}.csv")
            if self.live is not None:
                self.live.destroy()
            return self.recall_screen()
        self.root.after(500 if not self.args.dry_run else 20, self.poll)

    def recall_screen(self):
        seg = self.plan[self.k]
        if not seg.needs_words:
            self.recall = 0
            return self.quiz_screen()
        f = self.screen()
        self.text(f, "Write down every word you remember (any order, separated by spaces).", BIG)
        box = tk.Text(f, height=5, width=70, font=MID)
        box.pack(anchor="w")

        def done():
            self.recall = score_recall(box.get("1.0", "end"), seg.words)
            self.quiz_screen()
        ttk.Button(f, text="Next", command=done).pack(anchor="w", pady=12)

    def quiz_screen(self):
        seg = self.plan[self.k]
        qs = self.videos[seg.video].get("quiz", [])
        f = self.screen()
        self.text(f, "Quiz on the part you just saw. Guess if you are unsure.", BIG)
        answers = []
        for i, q in enumerate(qs):
            var = tk.IntVar(value=-1)
            answers.append(var)
            self.text(f, f"{i + 1}. {q['q']}", MID)
            row = ttk.Frame(f); row.pack(anchor="w")
            for j, opt in enumerate(q["options"]):
                ttk.Radiobutton(row, text=opt, variable=var, value=j).pack(side="left", padx=6)

        def done():
            quiz = sum(a.get() == q["answer"] for a, q in zip(answers, qs))
            self.responses[seg.index] = dict(quiz=int(quiz), recall=int(self.recall),
                                             condition=seg.condition, block=seg.block)
            json.dump(self.responses, open(self.dir / "responses.json", "w"), indent=1)
            self.k += 1
            if self.k < len(self.plan):
                self.instruction_screen()
            else:
                self.end_screen()
        ttk.Button(f, text="Next", command=done).pack(anchor="w", pady=12)

    def end_screen(self):
        f = self.screen()
        self.text(f, "Thank you. The session is complete.", BIG)
        self.text(f, "Nothing you did today affects your marks. No pictures or video of you were "
                     "saved, only numbers such as head angle and blink timing. You may ask for your "
                     "numbers to be deleted at any time before publication, using your participant "
                     f"number {self.pid}.")
        ttk.Button(f, text="Close", command=self.root.destroy).pack(anchor="w", pady=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="materials/videos.json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="no camera, no video: replay a simulated student")
    args = ap.parse_args()
    root = tk.Tk()
    SessionApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    print("Conditions:", ", ".join(f"{k} = {v}" for k, v in CONDITIONS.items()))
    main()
