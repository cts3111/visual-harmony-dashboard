#!/usr/bin/env python3
"""
LAB feature test: does CIELAB beat HSV for the undertone (warm/cool) axis?

Background: the 2025 LIACS thesis reusing Deep Armocromia reports LAB
features as best for the undertone layer. The project roadmap (Priority 2)
already planned a CIELAB + ITA upgrade. This script tests the claim with
OUR pipeline: it derives LAB features from the already-extracted features.csv
(skin LAB is exact via the stored dominant hex; iris/hair round-trip
HSV->RGB->LAB with <=1/255 error — no 25-minute re-extraction needed) and
re-runs the family evaluation (official split + 10-fold) against the HSV
baseline numbers from kfold_validation.py.

Usage:
    python3 lab_feature_test.py

Outputs:
    ../results/armocromia/lab_summary.csv
    ../../../output/lab-vs-hsv-report.md
"""

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from loocv_validation import knn_predict, logreg_predict, standardize, wilson

SRC_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SRC_DIR.parent / "results" / "armocromia" / "features.csv"
VAL_DIR = SRC_DIR.parent / "results" / "armocromia"
REPORT_PATH = SRC_DIR.parents[2] / "output" / "lab-vs-hsv-report.md"

# HSV results from kfold_validation.py (2026-11-05 run) for comparison.
HSV_BASELINE = {
    ("official-test", "logreg[skin-hsv]"): 569,
    ("official-test", "logreg[all-signals]"): 605,
    ("official-test", "knn-3[skin-hsv]"): 501,
    ("10-fold", "logreg[skin-hsv]"): 2891,
    ("10-fold", "logreg[all-signals]"): 3124,
    ("10-fold", "knn-3[skin-hsv]"): 2938,
}
N_OFFICIAL, N_FOLD = 910, 4911

LAB_COLS = ["skin_L", "skin_a", "skin_b", "iris_L", "iris_a", "iris_b",
            "hair_L", "hair_a", "hair_b"]

LAB_FEATURES = {
    "lab-b": ["skin_b"],
    "lab-ab": ["skin_a", "skin_b"],
    "lab-skin": ["skin_L", "skin_a", "skin_b"],
    "lab-skin-ita": ["skin_L", "skin_a", "skin_b", "skin_ita"],
    "lab-all": ["skin_L", "skin_a", "skin_b",
                "iris_L", "iris_a", "iris_b",
                "hair_L", "hair_a", "hair_b", "has_iris", "has_hair"],
    "hsv-lab-all": ["skin_red_dist", "skin_s", "skin_v",
                    "skin_L", "skin_a", "skin_b",
                    "iris_h", "iris_s", "iris_v",
                    "iris_L", "iris_a", "iris_b",
                    "hair_h", "hair_s", "hair_v",
                    "hair_L", "hair_a", "hair_b",
                    "has_iris", "has_hair"],
}


def hex_to_rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def rgb_to_lab(rgb):
    """OpenCV LAB: L 0-255, a/b centered at 128."""
    return [int(v) for v in
            cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0]]


def hsv_to_lab(h_raw, s_frac, v_frac):
    # (1,1,3) shape: OpenCV 5 rejects the ambiguous (1,3) for HSV2RGB.
    hsv = np.uint8([[[h_raw, s_frac * 255, v_frac * 255]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return rgb_to_lab(rgb)


def skin_ita(lab):
    """ITA angle in degrees from OpenCV LAB units (dermatology-standard)."""
    l_star, b_star = lab[0] * 100 / 255, lab[2] - 128
    b_star = b_star if abs(b_star) > 1e-6 else 1e-6
    return math.degrees(math.atan((l_star - 50) / b_star))


def lab_features_of(row):
    skin = rgb_to_lab(hex_to_rgb(row["dominant_hex"]))
    f = {"skin_L": skin[0], "skin_a": skin[1], "skin_b": skin[2],
         "skin_ita": skin_ita(skin),
         "has_iris": float(row["has_iris"]),
         "has_hair": float(row["has_hair"]),
         # HSV features too, so the combined set can mix both spaces
         "skin_red_dist": float(row["skin_red_dist"]),
         "skin_s": float(row["skin_s"]), "skin_v": float(row["skin_v"]),
         "iris_h": float(row["iris_h"]), "iris_s": float(row["iris_s"]),
         "iris_v": float(row["iris_v"]),
         "hair_h": float(row["hair_h"]), "hair_s": float(row["hair_s"]),
         "hair_v": float(row["hair_v"])}
    if float(row["has_iris"]) == 1.0:
        ir = hsv_to_lab(float(row["iris_h"]), float(row["iris_s"]),
                        float(row["iris_v"]))
        f.update(iris_L=ir[0], iris_a=ir[1], iris_b=ir[2])
    else:
        f.update(iris_L=128, iris_a=128, iris_b=128)  # neutral gray
    if float(row["has_hair"]) == 1.0:
        hr = hsv_to_lab(float(row["hair_h"]), float(row["hair_s"]),
                        float(row["hair_v"]))
        f.update(hair_L=hr[0], hair_a=hr[1], hair_b=hr[2])
    else:
        f.update(hair_L=128, hair_a=128, hair_b=128)
    return f


def stratified_folds(seasons, k, seed=42):
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(seasons), dtype=int)
    for cls in sorted(set(seasons)):
        idx = np.where(seasons == cls)[0]
        idx = idx[rng.permutation(len(idx))]
        for f in range(k):
            folds[idx[f::k]] = f
    return folds


def main():
    rows = [r for r in csv.DictReader(open(FEATURES_CSV))
            if r["status"] == "ok"]
    n = len(rows)
    fam = np.array([r["family"] for r in rows])
    season = np.array([r["season"] for r in rows])
    partition = np.array([r["partition"] for r in rows])
    lab_rows = [lab_features_of(r) for r in rows]

    tr = np.where(partition == "train")[0]
    te = np.where(partition == "test")[0]
    y01 = np.array([1 if t == "Warm" else 0 for t in fam])
    folds = stratified_folds(season, 10)

    print(f"{n} faces | LAB derived from stored hex/HSV\n")
    results = []  # (fs, model, split, ok, judged, lo, hi)

    for fs, cols in LAB_FEATURES.items():
        X = np.array([[lab_rows[i][c] for c in cols] for i in range(n)])
        # official split
        X_tr, X_te = standardize(X[tr], X[te])
        p01 = logreg_predict(X_tr, y01[tr], X_te, 0.5)
        p = np.where(p01 == 1, "Warm", "Cool")
        ok = int(np.sum(p == fam[te]))
        results.append((fs, "logreg", "official-test", ok, N_OFFICIAL,
                        *wilson(ok, N_OFFICIAL)))
        # 10-fold
        preds = np.empty(n, dtype=object)
        for f in range(10):
            ftr = np.where(folds != f)[0]
            fte = np.where(folds == f)[0]
            X_tr, X_te = standardize(X[ftr], X[fte])
            p01 = logreg_predict(X_tr, y01[ftr], X_te, 0.5)
            preds[fte] = np.where(p01 == 1, "Warm", "Cool")
        ok = int(np.sum(preds == fam))
        results.append((fs, "logreg", "10-fold", ok, N_FOLD,
                        *wilson(ok, N_FOLD)))
    for fs in ("lab-skin", "lab-all"):
        X = np.array([[lab_rows[i][c] for c in LAB_FEATURES[fs]]
                      for i in range(n)])
        X_tr, X_te = standardize(X[tr], X[te])
        p = knn_predict(X_tr, fam[tr], X_te, 3)
        ok = int(np.sum(np.array(p) == fam[te]))
        results.append((fs, "knn-3", "official-test", ok, N_OFFICIAL,
                        *wilson(ok, N_OFFICIAL)))
        preds = np.empty(n, dtype=object)
        for f in range(10):
            ftr = np.where(folds != f)[0]
            fte = np.where(folds == f)[0]
            X_tr, X_te = standardize(X[ftr], X[fte])
            preds[fte] = knn_predict(X_tr, fam[ftr], X_te, 3)
        ok = int(np.sum(preds == fam))
        results.append((fs, "knn-3", "10-fold", ok, N_FOLD,
                        *wilson(ok, N_FOLD)))

    lines = ["# LAB vs HSV — Undertone (Warm/Cool) Feature Test\n",
             f"**Date:** 2026-11-05 | **Tool:** lab_feature_test.py | "
             f"**n:** {n} faces (Deep Armocromia)\n",
             "| Split | Features | Model | Accuracy |",
             "|---|---|---|---|"]
    print(f"{'split':12s} {'features':14s} {'model':6s}  accuracy")
    for fs, model, split, ok, judged, lo, hi in results:
        lines.append(f"| {split} | {fs} | {model} | "
                     f"{ok}/{judged} ({ok / judged:.0%}, 95% CI "
                     f"{lo:.0%}-{hi:.0%}) |")
        print(f"{split:12s} {fs:14s} {model:6s}  {ok}/{judged} "
              f"({ok / judged:.0%})")
    for (split, key), ok in HSV_BASELINE.items():
        judged = N_OFFICIAL if split == "official-test" else N_FOLD
        lo, hi = wilson(ok, judged)
        lines.append(f"| {split} | {key} (HSV) | — | "
                     f"{ok}/{judged} ({ok / judged:.0%}, 95% CI "
                     f"{lo:.0%}-{hi:.0%}) |")
        print(f"{split:12s} {key:14s} {'HSV':6s}  {ok}/{judged} "
              f"({ok / judged:.0%})")
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("- skin_b is the CIELAB yellow-blue axis — the theoretical "
                 "undertone signal; ITA is the dermatology-standard "
                 "lightness metric")
    lines.append("- Decision rule: promote LAB into production if it beats "
                 "the HSV baselines on the official test split by a "
                 "meaningful margin")
    REPORT_PATH.write_text("\n".join(lines))

    with open(VAL_DIR / "lab_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "features", "model", "correct", "judged",
                    "acc", "ci_lo", "ci_hi"])
        for fs, model, split, ok, judged, lo, hi in results:
            w.writerow([split, fs, model, ok, judged, f"{ok / judged:.4f}",
                        f"{lo:.4f}", f"{hi:.4f}"])
    print(f"\nsummary -> {VAL_DIR / 'lab_summary.csv'}")
    print(f"report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
