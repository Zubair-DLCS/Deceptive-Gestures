"""Run every analysis on recorded data, or on simulated students.

  python scripts/run_analysis.py --data data/           # real sessions (study1/ and study2/)
  python scripts/run_analysis.py --simulate             # 40 simulated students + 3 classes
  python scripts/run_analysis.py --simulate --quick     # small, ~1 minute, for a smoke test

Writes results/*.csv, results/summary.json and results/fig*.png.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dg import analysis, data, models, report, simulate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()

    root = a.data
    note = ""
    if a.simulate:
        root = Path("data_simulated")
        shutil.rmtree(root, ignore_errors=True)          # never mix with an earlier simulation
        n, classes = (12, 1) if a.quick else (40, 3)
        print(f"simulating {n} students and {classes} classes in {root}/ ...")
        simulate.make_study1(root, n=n)
        simulate.make_study2(root, classes=classes, size=18 if a.quick else 24)
        note = " (SIMULATED)"
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    print("loading Study 1 ...")
    w1 = data.load_study1(root / "study1")
    print("scoring Study 1 (leave one student out) ...")
    s1 = models.score_study1(w1)
    w2 = data.load_study2(root / "study2") if (root / "study2").exists() else None
    s2 = models.score_transfer(w1, w2) if w2 is not None and len(w2) else None

    attack = analysis.rq1_attack(s1)
    tests = analysis.rq1_test(s1)
    det = analysis.detection_rates(s1, ["block"])
    coll = analysis.rq3_collateral(s1)
    h3 = analysis.h3_synchrony(s1)
    attack.to_csv(out / "rq1_attack.csv", index=False)
    tests.to_csv(out / "rq1_tests.csv", index=False)
    det.to_csv(out / "rq2_detection.csv", index=False)
    coll.to_csv(out / "rq3_collateral.csv", index=False)

    report.fig_attack(attack, out / "fig1_attack.png", note)
    report.fig_detection(det, out / "fig2_detection.png", note)
    if "per_student" in h3:
        report.fig_synchrony(h3["per_student"], out / "fig3_synchrony.png", note)
    report.fig_collateral(coll, out / "fig4_collateral.png", note)

    summary = analysis.summary(s1, s2)
    summary["simulated"] = bool(a.simulate)
    if s2 is not None:
        share = analysis.detection_rates(s2, ["faker_share"])
        share.to_csv(out / "rq2c_faker_share.csv", index=False)
        coll2 = analysis.rq3_collateral(s2)
        coll2.to_csv(out / "rq3_collateral_classroom.csv", index=False)
        analysis.rq1_attack(s2).to_csv(out / "rq1_attack_classroom.csv", index=False)
        report.fig_share(share, out / "fig5_faker_share.png", note)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)

    print("\nRQ1  windows rated attentive (rule model):")
    print(attack[attack.model == "rule"][["label", "block", "rated_attentive", "lo", "hi"]].round(2).to_string(index=False))
    print("\nRQ2  faked windows caught at 5% false accusation:")
    print(det[["block", "detector", "auroc", "caught", "caught_after_fooling", "false_accusation"]].round(2).to_string(index=False))
    print("\nH3   synchrony:", {k: round(v, 3) if isinstance(v, float) else v for k, v in summary["h3"].items()})
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
