"""The models students try to fool, and the detectors that try to catch them.

Targets (attention models), trained only on GENUINE behaviour, as a real deployment would be:
  rule      attentive if facing the screen most of the window and the face is found
  learned   logistic regression on the attention features, trained on genuine attention (A)
            vs genuine off-task (B); it never sees acting during training

Detectors (deception), trained on genuine attention (A) vs faked attention (C):
  dynamics  how the face moves
  coupling  response to lesson events + synchrony with other viewers
  combined  both

Every Study 1 score is out-of-person: the model is trained on all other students
(leave-one-student-out). Study 2 is scored by models trained on all of Study 1, i.e. the
lab-to-classroom transfer a real deployment would face.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import ATT, CPL, DYN_PREFIX

FALSE_ACCUSATION = 0.05     # detector threshold: flag at most 5% of genuine-attention windows


def feature_sets(cols) -> dict[str, list[str]]:
    dyn = [c for c in cols if c.startswith(DYN_PREFIX)]
    return {"dynamics": dyn, "coupling": list(CPL), "combined": dyn + list(CPL)}


def rule_attention(w: pd.DataFrame) -> np.ndarray:
    return ((w["att_on_screen"] >= 0.6) & (w["att_face_found"] >= 0.7)).to_numpy()


class LearnedAttention:
    def fit(self, w: pd.DataFrame):
        tr = w[w.valid & w.condition.isin(["A", "B"])]
        X = tr[ATT].fillna(tr[ATT].median())
        self.fill = tr[ATT].median()
        self.m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, tr.condition == "A")
        return self

    def predict(self, w: pd.DataFrame) -> np.ndarray:
        return self.m.predict_proba(w[ATT].fillna(self.fill))[:, 1] >= 0.5


class Detector:
    def __init__(self, cols: list[str], seed: int = 0):
        self.cols, self.seed = cols, seed

    def fit(self, w: pd.DataFrame):
        tr = w[w.valid & w.condition.isin(["A", "C"])]
        self.m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                l2_regularization=1.0, random_state=self.seed)
        self.m.fit(tr[self.cols], tr.condition == "C")
        genuine = self.score(tr[tr.condition == "A"])
        self.threshold = float(np.quantile(genuine, 1 - FALSE_ACCUSATION))
        return self

    def score(self, w: pd.DataFrame) -> np.ndarray:
        return self.m.predict_proba(w[self.cols])[:, 1]


def score_study1(w: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-student-out scores for every window: attention verdicts and detector flags."""
    w = w.copy()
    w["rule_attentive"] = rule_attention(w)
    sets = feature_sets(w.columns)
    for name in ["learned_attentive"] + [f"det_{k}" for k in sets] + [f"flag_{k}" for k in sets]:
        w[name] = np.nan
    for person in w.person.unique():
        test = w.person == person
        train = w[~test]
        w.loc[test, "learned_attentive"] = LearnedAttention().fit(train).predict(w[test]).astype(float)
        for k, cols in sets.items():
            det = Detector(cols).fit(train)
            s = det.score(w[test])
            w.loc[test, f"det_{k}"] = s
            w.loc[test, f"flag_{k}"] = (s >= det.threshold).astype(float)
    return _bools(w)


def score_transfer(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Models trained on Study 1 (lab), applied to Study 2 (classroom)."""
    test = test.copy()
    test["rule_attentive"] = rule_attention(test)
    test["learned_attentive"] = LearnedAttention().fit(train).predict(test)
    for k, cols in feature_sets(train.columns).items():
        det = Detector(cols).fit(train)
        test[f"det_{k}"] = det.score(test)
        test[f"flag_{k}"] = test[f"det_{k}"] >= det.threshold
    return _bools(test)


def _bools(w: pd.DataFrame) -> pd.DataFrame:
    for c in w.columns:
        if c.endswith("_attentive") or c.startswith("flag_"):
            w[c] = w[c].astype(bool)
    return w
