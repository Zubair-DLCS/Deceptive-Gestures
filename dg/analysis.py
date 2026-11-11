"""One function per research question. Intervals are 95% bootstraps over STUDENTS
(windows of one student are not independent)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .models import FALSE_ACCUSATION, feature_sets

SEED = 0
COND_LABEL = {"A": "Genuine attention", "B": "Genuine off-task", "C": "Faked attention",
              "D": "Faked distraction"}
BLOCKS = ["naive", "told", "dashboard"]
GROUPS = ["glasses", "face_covering", "lowered_gaze", "low_light"]


def boot_rate(df: pd.DataFrame, col: str, n: int = 1000, seed: int = SEED) -> tuple[float, float, float]:
    """Mean of a 0/1 column with a 95% bootstrap interval, resampling students."""
    per = df.groupby("person")[col].agg(["sum", "count"])
    if per.empty:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    s, c = per["sum"].to_numpy(float), per["count"].to_numpy(float)
    idx = rng.integers(0, len(per), (n, len(per)))
    draws = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def rq1_attack(w: pd.DataFrame) -> pd.DataFrame:
    """Share of windows each attention model rates ATTENTIVE, by condition and knowledge."""
    rows = []
    v = w[w.valid]
    for model in ["rule_attentive", "learned_attentive"]:
        for (cond, block), g in v.groupby(["condition", "block"]):
            m, lo, hi = boot_rate(g, model)
            rows.append(dict(model=model.replace("_attentive", ""), condition=cond, block=block,
                             label=COND_LABEL[cond], rated_attentive=m, lo=lo, hi=hi,
                             students=g.person.nunique(), windows=len(g)))
    return pd.DataFrame(rows)


def rq1_test(w: pd.DataFrame, model: str = "rule_attentive") -> pd.DataFrame:
    """Faked vs genuine attention within each knowledge block: logistic GEE, clustered by student."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    out = []
    v = w[w.valid & w.condition.isin(["A", "C"])].copy()
    v["y"] = v[model].astype(int)
    v["faked"] = (v.condition == "C").astype(int)
    for block, g in v.groupby("block"):
        if g.y.nunique() < 2 or g.faked.nunique() < 2:
            continue
        try:
            fit = smf.gee("y ~ faked", groups="person", data=g, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
            out.append(dict(model=model, block=block, odds_ratio=float(np.exp(fit.params["faked"])),
                            p=float(fit.pvalues["faked"])))
        except Exception as e:  # perfect separation etc.
            out.append(dict(model=model, block=block, odds_ratio=np.nan, p=np.nan, note=str(e)[:80]))
    return pd.DataFrame(out)


def detection_rates(w: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """For each detector: AUROC (genuine vs faked attention) and the share of faked-attention
    windows flagged at the 5% false-accusation threshold. Also the share caught among faked
    windows that FOOLED the rule attention model (the case that matters)."""
    rows = []
    v = w[w.valid & w.condition.isin(["A", "C"])]
    for det in feature_sets(w.columns):
        for key, g in v.groupby(by):
            key = key if isinstance(key, tuple) else (key,)
            fake, real = g[g.condition == "C"], g[g.condition == "A"]
            if fake.empty or real.empty:
                continue
            y = (g.condition == "C").to_numpy()
            auc = roc_auc_score(y, g[f"det_{det}"]) if 0 < y.sum() < len(y) else np.nan
            caught, lo, hi = boot_rate(fake, f"flag_{det}")
            fooled = fake[fake.rule_attentive]
            rows.append(dict(zip(by, key)) | dict(
                detector=det, auroc=auc, caught=caught, lo=lo, hi=hi,
                caught_after_fooling=float(fooled[f"flag_{det}"].mean()) if len(fooled) else np.nan,
                false_accusation=float(real[f"flag_{det}"].mean())))
    return pd.DataFrame(rows)


def h3_synchrony(w: pd.DataFrame) -> dict:
    """Paired, within student: mean synchrony during faked vs genuine attention."""
    v = w[w.valid & w.condition.isin(["A", "C"])]
    per = v.groupby(["person", "condition"])["cpl_synchrony"].mean().unstack().dropna()
    if len(per) < 3:
        return dict(students=len(per))
    d = per["A"] - per["C"]
    t = stats.ttest_rel(per["A"], per["C"])
    wil = stats.wilcoxon(per["A"], per["C"])
    return dict(students=int(len(per)), genuine=float(per["A"].mean()), faked=float(per["C"].mean()),
                difference=float(d.mean()), dz=float(d.mean() / d.std(ddof=1)), t=float(t.statistic),
                p_t=float(t.pvalue), p_wilcoxon=float(wil.pvalue), per_student=per.reset_index())


def rq3_collateral(w: pd.DataFrame, detector: str = "combined") -> pd.DataFrame:
    """Who gets wrongly accused: on GENUINE attention windows, the detector's flag rate and the
    rule attention model's 'not attentive' rate, for students with vs without each condition."""
    v = w[w.valid & (w.condition == "A")].copy()
    v["not_attentive"] = ~v.rule_attentive
    rows = []
    for grp in GROUPS + (["row"] if v.get("row") is not None and v.row.nunique() > 1 else []):
        for val, g in v.groupby(grp):
            for col, what in [(f"flag_{detector}", "accused of faking"), ("not_attentive", "rated not attentive")]:
                m, lo, hi = boot_rate(g, col)
                rows.append(dict(group=grp, value=str(val), measure=what, rate=m, lo=lo, hi=hi,
                                 students=g.person.nunique()))
    return pd.DataFrame(rows)


def summary(w1: pd.DataFrame, w2: pd.DataFrame | None = None) -> dict:
    s = dict(
        study1_students=int(w1.person.nunique()),
        study1_windows=int(len(w1)),
        excluded_segments=float(1 - w1.drop_duplicates(["person", "segment"]).valid.mean()),
        false_accusation_target=FALSE_ACCUSATION,
    )
    h3 = h3_synchrony(w1)
    s["h3"] = {k: v for k, v in h3.items() if k != "per_student"}
    if w2 is not None and len(w2):
        s["study2_students"] = int(w2.person.nunique())
        s["study2_windows"] = int(len(w2))
    return s
