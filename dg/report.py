"""Figures for the README and the paper. Static PNGs, one message each."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
                     "legend.frameon": False})


def _dots(ax, x, m, lo, hi, color, label=None):
    ax.errorbar(x, m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)], fmt="o",
                color=color, ms=7, lw=1.6, capsize=0, label=label, zorder=3)


def _pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.8)


def fig_attack(t: pd.DataFrame, out: Path, title_note: str = ""):
    """How often each attention model rates a window attentive, per condition."""
    order = [("A", "naive"), ("B", "naive"), ("C", "naive"), ("C", "told"), ("C", "dashboard"), ("D", "naive")]
    labels = ["Genuine\nattention", "Genuine\noff-task", "Faked att.\nnaive", "Faked att.\ntold",
              "Faked att.\ndashboard", "Faked\ndistraction"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, model, name in zip(axes, ["rule", "learned"], ["Rule model (facing the screen)", "Learned model"]):
        sub = t[t.model == model].set_index(["condition", "block"])
        for i, key in enumerate(order):
            if key not in sub.index:
                continue
            r = sub.loc[key]
            faked = key[0] in "CD"
            _dots(ax, [i], [r.rated_attentive], [r.lo], [r.hi], ORANGE if faked else BLUE,
                  ("acted" if faked else "genuine") if i in (0, 2) else None)
        ax.set_xticks(range(len(order)), labels, fontsize=8.5)
        ax.set_title(name, fontsize=10, color=INK)
        _pct(ax)
    axes[0].set_ylabel("Windows rated attentive")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Faked attention is rated attentive as often as the real thing" + title_note,
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_detection(d: pd.DataFrame, out: Path, title_note: str = ""):
    """Share of faked-attention windows caught at a 5% false-accusation rate."""
    dets = ["dynamics", "coupling", "combined"]
    blocks = [b for b in ["naive", "told", "dashboard"] if b in set(d.block)]
    colors = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for j, (b, c) in enumerate(zip(blocks, colors)):
        sub = d[d.block == b].set_index("detector")
        xs = [i + (j - 1) * 0.22 for i in range(len(dets))]
        _dots(ax, xs, sub.loc[dets, "caught"], sub.loc[dets, "lo"], sub.loc[dets, "hi"], c,
              f"fakers {b}")
    ax.set_xticks(range(len(dets)), ["How the face moves", "Following the lesson", "Both"])
    ax.set_ylabel("Faked windows caught")
    _pct(ax)
    ax.axhline(0.05, color=INK2, lw=0.8, ls=":")
    ax.text(2.45, 0.07, "chance (5%)", color=INK2, fontsize=8, ha="right")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    ax.set_title("Caught at a 5% false-accusation rate" + title_note, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_synchrony(per: pd.DataFrame, out: Path, title_note: str = ""):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for _, r in per.iterrows():
        ax.plot([0, 1], [r["A"], r["C"]], color=GRID, lw=1, zorder=1)
    ax.plot([0, 1], [per["A"].mean(), per["C"].mean()], color=ORANGE, lw=2.2, marker="o", ms=8, zorder=3)
    ax.set_xticks([0, 1], ["Genuine\nattention", "Faked\nattention"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("Synchrony with other viewers (r)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("Each line is one student" + title_note, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_collateral(c: pd.DataFrame, out: Path, title_note: str = ""):
    names = {"glasses": "Glasses", "face_covering": "Face covered", "lowered_gaze": "Lowers gaze to listen",
             "low_light": "Poor light", "row": "Seat row"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, measure in zip(axes, ["accused of faking", "rated not attentive"]):
        sub = c[(c.measure == measure) & (c.group != "row")].assign(value=lambda x: x.value.astype(str))
        ys, labels = [], []
        for i, g in enumerate([g for g in names if g in set(sub.group)]):
            for val, color, off in [("False", BLUE, -0.15), ("True", ORANGE, 0.15)]:
                r = sub[(sub.group == g) & (sub.value == val)]
                if len(r):
                    r = r.iloc[0]
                    ax.errorbar([r.rate], [i + off], xerr=[[r.rate - r.lo], [r.hi - r.rate]], fmt="o",
                                color=color, ms=6, lw=1.4,
                                label=("no" if val == "False" else "yes") if i == 0 else None)
            ys.append(i)
            labels.append(names[g])
        ax.set_yticks(ys, labels)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_title(f"Honest attention {measure}", fontsize=10, color=INK)
    fig.suptitle("Who pays for the detector: false accusations by group" + title_note, x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8.5,
               title="student has the condition", title_fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_share(d: pd.DataFrame, out: Path, title_note: str = ""):
    sub = d[d.detector == "combined"].sort_values("faker_share")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(sub.faker_share, sub.caught, color=BLUE, lw=2, marker="o", ms=7)
    ax.fill_between(sub.faker_share, sub.lo, sub.hi, color=BLUE, alpha=0.12, lw=0)
    ax.set_xticks(sub.faker_share, [f"{s:.0%}" for s in sub.faker_share])
    ax.set_xlabel("Share of the class faking attention")
    ax.set_ylabel("Faked windows caught")
    _pct(ax)
    ax.set_title("Classroom, both detectors" + title_note, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
