#!/usr/bin/env python3
"""
Export trained classifiers for the dashboard app.

Mirrors the sklearn_eval.py protocol exactly (5-fold inner GridSearchCV on
the official train split, official test scored once) and saves the tuned
pipelines with joblib so the Streamlit app can classify uploaded photos
with the SAME models the reports describe.

Saved to ../models/:
    family_pipeline.joblib   warm/cool (binary logreg, hsv-lab)
    modifier_pipeline.joblib 5 modifiers (multinomial logreg, hsv-lab)
    season_pipeline.joblib   12-season (multinomial logreg, hsv-lab)
    season4_pipeline.joblib  4-season (multinomial logreg, hsv-lab)
    feature_config.json      column order + test-set stats per model

Usage:
    python3 export_models.py
"""

import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kfold_validation import CLASS_TO_SEASON
from lab_feature_test import lab_features_of
from loocv_validation import FAMILY_FEATURES, wilson

SRC_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SRC_DIR.parent / "results" / "armocromia" / "features.csv"
MODEL_DIR = SRC_DIR.parent / "models"

HSV_ALL = FAMILY_FEATURES["all-signals"] + ["contrast"]
LAB_COLS = ["skin_L", "skin_a", "skin_b", "iris_L", "iris_a", "iris_b",
            "hair_L", "hair_a", "hair_b"]
HSLAB_COLS = HSV_ALL + LAB_COLS  # 20 columns, the winning feature set

# Expected official-test numbers from the 2026-11-05 sklearn_eval.py run
# (results/armocromia/sklearn_summary.csv) — used to verify reproducibility.
EXPECTED = {
    ("family", "logreg"): (616, 910),
    ("season", "logreg"): None,   # captured from run, compared for drift only
    ("modifier", "logreg"): None,
    ("4-season", "logreg"): None,
}


def build_features(rows):
    hsv = np.array([[float(r[c]) for c in HSV_ALL] for r in rows])
    lab = {c: np.array([lab_features_of(r)[c] for r in rows])
           for c in LAB_COLS}
    return np.column_stack([hsv] + [lab[c] for c in LAB_COLS])


def make_pipeline(multi):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial" if multi else "auto",
            max_iter=2000, solver="lbfgs", random_state=42)),
    ])


def main():
    rows = [r for r in csv.DictReader(open(FEATURES_CSV))
            if r["status"] == "ok"]
    fam = np.array([r["family"] for r in rows])
    mod = np.array([r["modifier"] for r in rows])
    season = np.array([r["season"] for r in rows])
    class4 = np.array([CLASS_TO_SEASON[r["class"]] for r in rows])
    partition = np.array([r["partition"] for r in rows])
    X = build_features(rows)
    tr = np.where(partition == "train")[0]
    te = np.where(partition == "test")[0]
    inner_cv = StratifiedKFold(5, shuffle=True, random_state=42)

    targets = {"family": (fam, False), "modifier": (mod, True),
               "season": (season, True), "4-season": (class4, True)}
    config = {"columns": HSLAB_COLS, "n_train": int(len(tr)),
              "n_test": int(len(te)), "models": {}}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} faces | train {len(tr)} / test {len(te)} | "
          f"features {X.shape[1]}\n")

    for tname, (y, multi) in targets.items():
        gs = GridSearchCV(make_pipeline(multi),
                          {"clf__C": [0.01, 0.1, 1.0, 10.0]},
                          cv=inner_cv, scoring="accuracy", refit=True)
        gs.fit(X[tr], y[tr])
        test_pred = gs.predict(X[te])
        ok = int(np.sum(test_pred == y[te]))
        lo, hi = wilson(ok, len(te))
        acc = ok / len(te)
        out = MODEL_DIR / f"{tname}_pipeline.joblib"
        joblib.dump(gs.best_estimator_, out)
        classes = list(gs.best_estimator_.classes_)
        info = {"file": out.name, "best_C": float(gs.best_params_["clf__C"]),
                "test_ok": ok, "test_n": int(len(te)), "test_acc": acc,
                "ci_lo": lo, "ci_hi": hi,
                "train_cv": float(gs.best_score_), "classes": classes}
        config["models"][tname] = info
        exp = EXPECTED.get((tname, "logreg"))
        drift = ""
        if exp is not None:
            drift = "MATCH" if ok == exp[0] else \
                f"DRIFT from expected {exp[0]}/{exp[1]}"
        print(f"{tname:9s} C={info['best_C']}: test {ok}/{len(te)} "
              f"({acc:.1%}, CI {lo:.0%}-{hi:.0%}) | trainCV "
              f"{gs.best_score_:.0%} | {drift}")

    (MODEL_DIR / "feature_config.json").write_text(
        json.dumps(config, indent=2))
    print(f"\nsaved -> {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
