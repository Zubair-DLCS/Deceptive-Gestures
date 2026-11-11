"""Deceptive Gestures desktop app: one start window for both studies.

Double-click the desktop icon. Data and materials live in Documents/DeceptiveGestures:
  materials/videos.json   lecture clips, slide-change times and quiz questions (edit this)
  materials/videos/       the clips themselves (V01.mp4 ... V10.mp4)
  data/                   everything recorded (numbers only, never images)

Build checks (used by the Windows build, not by people):
  DeceptiveGestures.exe --check-window   open the start window, close it, report
  DeceptiveGestures.exe --self-test      run the bundled face model on a generated clip
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# A windowed .exe has no console: give libraries somewhere harmless to print.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = "DeceptiveGestures"
CREDIT = "Developed by Malak Zubair, Assistant Professor, G.C.M.S Timergara"
HOME = Path.home() / "Documents" / APP


def bundled(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", ROOT))
    return base / name


def workspace() -> tuple[Path, Path]:
    """Documents/DeceptiveGestures with materials copied in on first run."""
    mat, data = HOME / "materials", HOME / "data"
    if not (mat / "videos.json").exists():
        shutil.copytree(bundled("materials"), mat, dirs_exist_ok=True)
    (mat / "videos").mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    return mat, data


def open_folder(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(str(p))  # noqa: S606
    else:
        subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", str(p)])


def log_error(text: str):
    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / "last_error.txt").write_text(text, encoding="utf-8")


def launcher(auto_close_ms: int | None = None):
    import tkinter as tk
    from tkinter import messagebox, ttk

    from apps import class_app
    from apps.session_app import SessionApp

    mat, data = workspace()
    root = tk.Tk()
    root.title("Deceptive Gestures")
    root.geometry("640x600")
    ico = bundled("packaging/deceptive_gestures.ico")
    if ico.exists() and sys.platform.startswith("win"):
        root.iconbitmap(str(ico))
    f = ttk.Frame(root, padding=22)
    f.pack(fill="both", expand=True)
    ttk.Label(f, text="Deceptive Gestures", font=("Segoe UI", 18, "bold")).pack(anchor="w")
    ttk.Label(f, text="Can students fool classroom attention AI? No images are ever saved.",
              font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 12))

    def session(dry: bool):
        args = argparse.Namespace(videos=str(mat / "videos.json"), camera=int(cam.get()),
                                  data=str(data if not dry else HOME / "data_dryrun"), dry_run=dry)
        win = tk.Toplevel(root)
        try:
            SessionApp(win, args)
        except Exception as e:
            win.destroy()
            log_error(repr(e))
            messagebox.showerror("Problem", f"{type(e).__name__}: {e}")

    box = ttk.LabelFrame(f, text="Study 1: one student at this laptop", padding=12)
    box.pack(fill="x", pady=6)
    row = ttk.Frame(box)
    row.pack(anchor="w")
    ttk.Label(row, text="Camera number:").pack(side="left")
    cam = tk.StringVar(value="0")
    ttk.Spinbox(row, from_=0, to=5, textvariable=cam, width=4).pack(side="left", padx=6)
    ttk.Button(box, text="Start a session", command=lambda: session(False)).pack(anchor="w", pady=4)
    ttk.Button(box, text="Try a session without the camera (practice)",
               command=lambda: session(True)).pack(anchor="w")

    box2 = ttk.LabelFrame(f, text="Study 2: a whole class, one camera", padding=12)
    box2.pack(fill="x", pady=6)
    vals = {k: tk.StringVar(value=v) for k, v in
            dict(cls="class1", session="1", segment="1", video="V01").items()}
    row = ttk.Frame(box2)
    row.pack(anchor="w")
    for label, key, w in [("Class", "cls", 9), ("Session", "session", 4), ("Part", "segment", 4), ("Video", "video", 6)]:
        ttk.Label(row, text=label).pack(side="left")
        ttk.Entry(row, textvariable=vals[key], width=w).pack(side="left", padx=(4, 12))
    status = ttk.Label(box2, text="", wraplength=560, font=("Segoe UI", 9))

    def class_step(step: str):
        a = argparse.Namespace(step=step, class_id=vals["cls"].get(), session=int(vals["session"].get()),
                               segment=int(vals["segment"].get()), video=vals["video"].get(),
                               videos=str(mat / "videos.json"), data=str(data), camera=int(cam.get()),
                               max_faces=40, seed=int(time.time()) % 10000, dry_run=False)
        status.config(text={"seats": "Learning seats: everyone sits still and faces the board (10 s) ...",
                            "record": "Recording this part. Start the video on the projector now ..."}.get(step, "..."))
        out = io.StringIO()

        def work():
            try:
                with contextlib.redirect_stdout(out):
                    {"seats": class_app.cmd_seats, "cards": class_app.cmd_cards,
                     "record": class_app.cmd_record}[step](a)
                msg = out.getvalue().strip()
            except Exception as e:
                log_error(repr(e))
                msg = f"Problem: {type(e).__name__}: {e}"
            root.after(0, lambda: status.config(text=msg))
        threading.Thread(target=work, daemon=True).start()

    row = ttk.Frame(box2)
    row.pack(anchor="w", pady=6)
    ttk.Button(row, text="1. Learn seats", command=lambda: class_step("seats")).pack(side="left")
    ttk.Button(row, text="2. Make cards", command=lambda: class_step("cards")).pack(side="left", padx=6)
    ttk.Button(row, text="3. Record this part", command=lambda: class_step("record")).pack(side="left")
    status.pack(anchor="w")

    row = ttk.Frame(f)
    row.pack(anchor="w", pady=10)
    ttk.Button(row, text="Open materials folder", command=lambda: open_folder(mat)).pack(side="left")
    ttk.Button(row, text="Open data folder", command=lambda: open_folder(data)).pack(side="left", padx=6)
    ttk.Label(f, text="Put the lecture clips in materials\\videos and edit materials\\videos.json "
                      "(slide-change times and quiz questions) before the first real session.",
              wraplength=580, font=("Segoe UI", 9)).pack(anchor="w")
    ttk.Label(f, text=CREDIT, font=("Segoe UI", 8)).pack(side="bottom", anchor="w")

    if auto_close_ms:
        def snap_and_close():
            try:
                from PIL import ImageGrab
                root.update()
                x, y, w, h = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
                ImageGrab.grab(bbox=(x - 8, y - 32, x + w + 8, y + h + 8)).save(HOME / "start_window.png")
            except Exception:
                pass
            root.destroy()
        root.after(auto_close_ms, snap_and_close)
    root.mainloop()


def self_test() -> int:
    """Run the bundled face model on a generated clip: proves the model and camera loop
    work inside the built app (no face in the clip, so none is expected)."""
    import cv2
    import numpy as np

    from dg.sensing import Camera, model_file
    clip = HOME / "self_test.mp4"
    HOME.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(60):
        vw.write(np.full((240, 320, 3), 80 + i, np.uint8))
    vw.release()
    n = sum(1 for _ in Camera(str(clip)).frames())
    ok = model_file().exists() and 18 <= n <= 22
    (HOME / "self_test.txt").write_text(f"model={model_file()} samples={n} ok={ok}\n")
    return 0 if ok else 1


def main():
    try:
        if sys.argv[1:] == ["--check-window"]:
            launcher(auto_close_ms=1500)
            (HOME / "window_check_ok.txt").write_text("ok")
            return 0
        if sys.argv[1:] == ["--self-test"]:
            return self_test()
        launcher()
        return 0
    except Exception:
        import traceback
        log_error(traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
