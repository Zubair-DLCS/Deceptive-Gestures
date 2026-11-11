"""Face numbers -> 10-second windows -> three families of features.

  attention   what typical engagement models use: facing the screen, eyes open, blinks
  dynamics    HOW the face moves: stillness, variability, small corrections, abrupt onsets
  coupling    whether the face follows the LESSON: response to lesson events and
              synchrony with other viewers of the same video (or the rest of the room)

Every signal is compared with the person's own median over the whole session (no labels
needed), so someone who sits slightly turned or blinks a lot is not penalised for it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sensing import HZ, SIGNALS

WIN_S = 10
WIN = WIN_S * HZ
SYNC_SIGNALS = ["gaze_x", "gaze_y", "yaw", "pitch"]
MAX_LAG_S = 2.0          # timing slack between the video player and the recording


def to_grid(df: pd.DataFrame, duration: float) -> pd.DataFrame:
    """Resample one person's samples onto an exact 10 Hz grid from 0 to `duration` s.
    Each grid point takes the nearest sample within half a step; missing faces become NaN.
    Gaps of up to 0.5 s are filled by interpolation; longer gaps stay missing."""
    grid = np.arange(int(round(duration * HZ))) / HZ
    out = pd.DataFrame({"t": grid})
    d = df.sort_values("t")
    ts = d["t"].to_numpy(float)
    if len(ts) == 0:
        for c in SIGNALS:
            out[c] = np.nan
        out["face"] = 0.0
        return out
    j = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)
    j = np.where(np.abs(ts[j - 1] - grid) <= np.abs(ts[j] - grid), j - 1, j)
    close = np.abs(ts[j] - grid) <= 0.5 / HZ + 1e-9
    seen = close & (d["face"].to_numpy()[j] > 0)
    for c in SIGNALS:
        out[c] = np.where(seen, d[c].to_numpy(float)[j], np.nan)
    out["face"] = seen.astype(float)
    # fill only short gaps (blinks, a missed frame); a longer absence stays missing
    run = (~seen).astype(int)
    gid = np.cumsum(np.r_[0, np.diff(run) != 0])
    gap_len = pd.Series(run).groupby(gid).transform("sum").to_numpy()
    filled = out[SIGNALS].interpolate(limit_area="inside")
    short = (~seen) & (gap_len <= int(0.5 * HZ))
    out.loc[short, SIGNALS] = filled.loc[short, SIGNALS]
    if "brightness" in d:
        out["brightness"] = float(d["brightness"].mean())
    return out


def centre(grid: pd.DataFrame, baseline: dict | None = None) -> pd.DataFrame:
    """Subtract the person's own median (or a given baseline) from each signal."""
    g = grid.copy()
    base = baseline or person_baseline([g])
    for c in SIGNALS:
        g[c] = g[c] - base[c]
    return g


def person_baseline(grids: list[pd.DataFrame]) -> dict:
    allg = pd.concat(grids, ignore_index=True)
    return {c: float(np.nanmedian(allg[c])) if allg[c].notna().any() else 0.0 for c in SIGNALS}


def _blinks(eye: np.ndarray) -> np.ndarray:
    """Frame indices where a blink starts (eye opening dips below 60% of its median)."""
    e = np.nan_to_num(eye, nan=np.nanmedian(eye) if np.isfinite(eye).any() else 0.0)
    shut = e < 0.6 * np.median(e[e > 0]) if (e > 0).any() else np.zeros_like(e, bool)
    return np.flatnonzero(shut[1:] & ~shut[:-1]) + 1


def _arrays(w) -> dict:
    return w if isinstance(w, dict) else {c: w[c].to_numpy(float) for c in w.columns}


def attention_features(w, eye_ref: float) -> dict:
    w = _arrays(w)
    face = w["face"] > 0
    yaw, pitch, gx = w["yaw"], w["pitch"], w["gaze_x"]
    with np.errstate(invalid="ignore"):
        on = face & (np.abs(yaw) < 15) & (np.abs(pitch) < 12) & (np.abs(gx) < 0.35)
    seen = face.any() and np.isfinite(yaw).any()
    return dict(
        on_screen=float(on.mean()),
        face_found=float(face.mean()),
        abs_yaw=float(np.nanmean(np.abs(yaw))) if seen else 90.0,
        abs_pitch=float(np.nanmean(np.abs(pitch))) if seen else 90.0,
        eye_open=float(np.nanmean(w["eye_open"])) if seen else -1.0,
        blink_rate=len(_blinks(w["eye_open"] + eye_ref)) * 60 / WIN_S,
        gaze_spread=float(np.nanstd(gx)) if seen else 1.0,
    )


SMOOTH = 5   # 0.5 s moving average before measuring motion, so sensor jitter is not counted


def _smooth(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rolling(SMOOTH, center=True, min_periods=2).mean().to_numpy()


def dynamics_features(w, eye_ref: float, smoothed: dict | None = None) -> dict:
    """`smoothed` = the same window's signals already smoothed over the whole segment."""
    w = _arrays(w)
    f = {}
    vel = {}
    for c in SYNC_SIGNALS:
        x = smoothed[c] if smoothed is not None else _smooth(w[c])
        if not np.isfinite(x).any():
            x = np.zeros_like(x)
        dx = np.diff(x)
        vel[c] = np.abs(dx)
        f[f"std_{c}"] = float(np.nanstd(x))
        f[f"speed_{c}"] = float(np.nanmean(np.abs(dx))) if np.isfinite(dx).any() else 0.0
        acc = np.abs(np.diff(dx))
        sd = np.nanstd(dx) + 1e-6 if np.isfinite(dx).any() else 1.0
        f[f"onset_{c}"] = float(np.nanpercentile(acc, 95) / sd) if np.isfinite(acc).any() else 0.0
    head = np.nan_to_num(vel["yaw"]) + np.nan_to_num(vel["pitch"])
    gaze = np.nan_to_num(vel["gaze_x"]) + np.nan_to_num(vel["gaze_y"])
    # relative measures: robust to how noisy a given camera is
    med = np.median(head) + 1e-6
    f["head_burstiness"] = float(np.percentile(head, 90) / med)
    f["head_quiet_frac"] = float((head < 0.5 * med).mean())
    f["big_moves"] = float((head >= 1.0).sum() * 60 / WIN_S)
    f["saccades"] = float((gaze >= 0.05).sum() * 60 / WIN_S)
    starts = _blinks(w["eye_open"] + eye_ref)
    f["blink_rate"] = len(starts) * 60 / WIN_S
    ibi = np.diff(starts) / HZ
    f["blink_regularity"] = float(np.std(ibi) / (np.mean(ibi) + 1e-6)) if len(ibi) >= 2 else 0.0
    return f


def event_response(g, events: list[float], lo: float, hi: float) -> float:
    """Mean change in orientation speed in the 2 s after each lesson event in [lo, hi)
    minus the 2 s before. NaN if the window holds no event."""
    g = _arrays(g)
    t = g["t"]
    sp = np.nan_to_num(np.abs(np.diff(g["gaze_x"], prepend=np.nan))) \
        + np.nan_to_num(np.abs(np.diff(g["yaw"], prepend=np.nan))) / 10
    vals = []
    for e in events:
        if lo <= e < hi:
            after = sp[(t >= e) & (t < e + 2)]
            before = sp[(t >= e - 2) & (t < e)]
            if len(after) and len(before):
                vals.append(after.mean() - before.mean())
    return float(np.mean(vals)) if vals else np.nan


def _z(x: np.ndarray) -> np.ndarray:
    x = x - np.nanmean(x)
    s = np.nanstd(x)
    return x / s if s > 0 else x * 0


def best_lag(x: np.ndarray, ref: np.ndarray, max_lag: int = int(MAX_LAG_S * HZ)) -> int:
    """Shift (in samples) of x against ref that maximises their correlation over a segment.
    Positive lag = x lags behind ref."""
    a, b = np.nan_to_num(_z(x)), np.nan_to_num(_z(ref))
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 3 * max_lag or not a.any() or not b.any():
        return 0
    lags = np.arange(-max_lag, max_lag + 1)
    full = np.correlate(a, b, mode="full")[n - 1 - max_lag: n + max_lag]
    overlap = n - np.abs(lags)
    return int(lags[np.argmax(full / overlap)])


def shift(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if lag >= 0:
        out[: len(x) - lag] = x[lag:]
    else:
        out[-lag:] = x[: len(x) + lag]
    return out


def synchrony(g: pd.DataFrame, ref: pd.DataFrame, lags: dict | None = None) -> tuple[np.ndarray, dict]:
    """Per-window mean correlation with a reference across SYNC_SIGNALS.
    Lags are fitted once per segment (timing slack), then held fixed."""
    lags = lags or {}
    per_sig = []
    fitted = {}
    n_win = len(g) // WIN
    for c in SYNC_SIGNALS:
        x, r = _z(g[c].to_numpy(float)), _z(ref[c].to_numpy(float))
        lag = lags.get(c, best_lag(x, r))
        fitted[c] = lag
        xs = shift(x, lag)
        vals = []
        for k in range(n_win):
            a, b = xs[k * WIN:(k + 1) * WIN], r[k * WIN:(k + 1) * WIN]
            ok = np.isfinite(a) & np.isfinite(b)
            vals.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 20 and np.std(a[ok]) > 0
                        and np.std(b[ok]) > 0 else np.nan)
        per_sig.append(vals)
    return np.nanmean(np.array(per_sig, float), axis=0) if n_win else np.array([]), fitted


def zsignals(g: pd.DataFrame) -> dict:
    """Each sync signal z-scored over the segment (what the reference is built from)."""
    return {c: _z(g[c].to_numpy(float)) for c in SYNC_SIGNALS}


def reference(others: list) -> pd.DataFrame:
    """Median of other viewers' z-scored signals, sample by sample (label-free).
    `others` = grids (DataFrames) or precomputed `zsignals` dicts."""
    zs = [o if isinstance(o, dict) else zsignals(o) for o in others]
    n = min(len(z[SYNC_SIGNALS[0]]) for z in zs)
    ref = {}
    for c in SYNC_SIGNALS:
        stack = np.vstack([z[c][:n] for z in zs])
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            ref[c] = np.nanmedian(stack, axis=0)
    return pd.DataFrame(ref)


def window_table(g: pd.DataFrame, eye_ref: float, events: list[float],
                 sync: np.ndarray | None = None) -> pd.DataFrame:
    """One row per complete 10-s window with all three feature families."""
    arr = _arrays(g)
    sm = {c: _smooth(arr[c]) for c in SYNC_SIGNALS}
    rows = []
    for k in range(len(arr["t"]) // WIN):
        sl = slice(k * WIN, (k + 1) * WIN)
        w = {c: v[sl] for c, v in arr.items()}
        lo, hi = k * WIN_S, (k + 1) * WIN_S
        row = {"window": k}
        row |= {f"att_{a}": v for a, v in attention_features(w, eye_ref).items()}
        row |= {f"dyn_{a}": v for a, v in dynamics_features(w, eye_ref, {c: v[sl] for c, v in sm.items()}).items()}
        row["cpl_event_response"] = event_response(arr, events, lo, hi)
        row["cpl_synchrony"] = float(sync[k]) if sync is not None and k < len(sync) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


ATT = ["att_on_screen", "att_face_found", "att_abs_yaw", "att_abs_pitch", "att_eye_open",
       "att_blink_rate", "att_gaze_spread"]
DYN_PREFIX, CPL = "dyn_", ["cpl_event_response", "cpl_synchrony"]
