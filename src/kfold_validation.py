#!/usr/bin/env python3
"""
Deep Armocromia evaluation: official train/test split (headline) +
10-fold cross-validation (supporting numbers).

Consumes results/armocromia/features.csv (written by
extract_armocromia_features.py). Models (numpy-only, from
loocv_validation): logistic regression (binary family), softmax regression
(modifier + 12-season), k-NN.

Headline: train on the official train partition (4,008 faces), score the
held-out test partition (912 faces) — the same split the Deep Armocromia
authors' CNN baselines (~55% 4-season / ~32% 12-subtype) use, so numbers
are directly comparable. The fixed deterministic rule gets its FIRST
genuinely out-of-sample test here (its thresholds were calibrated on 14
unrelated faces).

Supporting: 10-fold stratified CV (by 12-season label, seed 42) over all
extracted faces, for accuracy spread.

Metrics: accuracy with Wilson 95% CI; for 12-season targets also
family-correct and exact-or-adjacent rates (adjacent = wheel neighbors).

Usage:
    python3 kfold_validation.py [--folds 10]

Outputs:
    ../results/armocromia/kfold_summary.csv     official-split + CV rows
    ../../../output/kfold-validation-report.md  summary report
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from loocv_validation import (FAMILY_FEATURES, MOD_FEATURES, knn_predict,
                              logreg_predict, softmax_predict, standardize,
                              wilson)

SRC_DIR = Path(__file__).resolve().parent
FEATURES_CSV = SRC_DIR.parent / "results" / "armocromia" / "features.csv"
VAL_DIR = SRC_DIR.parent / "results" / "armocromia"
REPORT_PATH = SRC_DIR.parents[2] / "output" / "kfold-validation-report.md"

WHEEL = ["BRIGHT SPRING", "WARM SPRING", "LIGHT SPRING", "LIGHT SUMMER",
         "COOL SUMMER", "SOFT SUMMER", "SOFT AUTUMN", "WARM AUTUMN",
         "DEEP AUTUMN", "DEEP WINTER", "COOL WINTER", "BRIGHT WINTER"]
WHEEL_IDX = {s: i for i, s in enumerate(WHEEL)}

CLASS_TO_SEASON = {"primavera": "SPRING", "estate": "SUMMER",
                   "autunno": "AUTUMN", "inverno": "WINTER"}
SEASON_FEATURES = FAMILY_FEATURES["all-signals"] + ["contrast"]


def load_rows():
    return [r for r in csv.DictReader(open(FEATURES_CSV))
            if r["status"] == "ok"]


def is_warm_label(s):
    return s.split()[1] in ("SPRING", "AUTUMN")


def adjacent(pred, true):
    i, j = WHEEL_IDX[pred], WHEEL_IDX[true]
    return i == j or (i - j) % 12 in (1, 11)


def fit_eval(X, y, tr, te, fit_predict):
    """Train on tr, predict te. Returns (preds, truth)."""
    X_tr, X_te = standardize(X[tr], X[te])
    return fit_predict(X_tr, y[tr], X_te), y[te]


def fmt_ci(ok, n):
    lo, hi = wilson(ok, n)
    return f"{ok}/{n} ({ok / n:.0%}, 95% CI {lo:.0%}-{hi:.0%})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()
    k = args.folds

    rows = load_rows()
    n = len(rows)
    fam = np.array([r["family"] for r in rows])
    mod = np.array([r["modifier"] for r in rows])
    season = np.array([r["season"] for r in rows])
    class4 = np.array([CLASS_TO_SEASON[r["class"]] for r in rows])
    partition = np.array([r["partition"] for r in rows])
    tr = np.where(partition == "train")[0]
    te = np.where(partition == "test")[0]
    print(f"{n} extracted faces | official split: train {len(tr)} / "
          f"test {len(te)} | {k}-fold CV")

    results = []  # (split, target, model, ok, judged, lo, hi, extra)

    # ================= official split (headline) =================
    split = "official-test"
    y01 = np.array([1 if t == "Warm" else 0 for t in fam])
    for fs in ("skin-hsv", "all-signals"):
        X = np.array([[float(r[c]) for c in FAMILY_FEATURES[fs]]
                      for r in rows])
        p, _ = fit_eval(X, y01, tr, te,
                        lambda Xtr, ytr, Xte: logreg_predict(
                            Xtr, ytr, Xte, 0.5))
        p = np.where(p == 1, "Warm", "Cool")
        ok = int(np.sum(p == fam[te]))
        results.append((split, "family", f"logreg[{fs}]", ok, len(te),
                        *wilson(ok, len(te)), None))
    X = np.array([[float(r[c]) for c in FAMILY_FEATURES["skin-hsv"]]
                  for r in rows])
    p, truth = fit_eval(X, fam, tr, te,
                        lambda Xtr, ytr, Xte: knn_predict(Xtr, ytr, Xte, 3))
    ok = int(np.sum(p == truth))
    results.append((split, "family", "knn-3[skin-hsv]", ok, len(te),
                    *wilson(ok, len(te)), None))

    Xm = np.array([[float(r[c]) for c in MOD_FEATURES] for r in rows])
    mclasses = sorted(set(mod))
    p, truth = fit_eval(Xm, mod, tr, te,
                        lambda Xtr, ytr, Xte: np.array(softmax_predict(
                            Xtr, ytr, Xte, mclasses, 1.0)))
    ok = int(np.sum(p == truth))
    results.append((split, "modifier", "softmax", ok, len(te),
                    *wilson(ok, len(te)), None))
    p, truth = fit_eval(Xm, mod, tr, te,
                        lambda Xtr, ytr, Xte: knn_predict(Xtr, ytr, Xte, 3))
    ok = int(np.sum(p == truth))
    results.append((split, "modifier", "knn-3", ok, len(te),
                    *wilson(ok, len(te)), None))

    Xs = np.array([[float(r[c]) for c in SEASON_FEATURES] for r in rows])
    sclasses = sorted(set(season))
    p, truth = fit_eval(Xs, season, tr, te,
                        lambda Xtr, ytr, Xte: np.array(softmax_predict(
                            Xtr, ytr, Xte, sclasses, 1.0)))
    ok = int(np.sum(p == truth))
    famc = int(np.sum([is_warm_label(a) == is_warm_label(b)
                       for a, b in zip(p, truth)]))
    adjc = int(np.sum([adjacent(a, b) for a, b in zip(p, truth)]))
    results.append((split, "season", "softmax", ok, len(te),
                    *wilson(ok, len(te)),
                    f"fam {famc}/{len(te)}, adj {adjc}/{len(te)}"))
    p4 = class4[te]
    c4_soft = int(np.sum(np.array([x.split()[1] for x in p]) == p4))
    p, truth = fit_eval(Xs, season, tr, te,
                        lambda Xtr, ytr, Xte: knn_predict(Xtr, ytr, Xte, 5))
    ok = int(np.sum(p == truth))
    results.append((split, "season", "knn-5", ok, len(te),
                    *wilson(ok, len(te)), None))

    # fixed rule on the held-out test set (first out-of-sample test)
    rule_s = np.array([r["rule_season"] for r in rows], dtype=object)[te]
    judged = rule_s != ""
    ok = int(np.sum(rule_s[judged] == season[te][judged]))
    lo, hi = wilson(ok, int(judged.sum()))
    results.append((split, "season", "rule", ok, int(judged.sum()), lo, hi,
                    f"{int((~judged).sum())} abstain"))
    r4 = np.array([x.split()[1] if x else "" for x in rule_s])
    ok4 = int(np.sum(r4[judged] == p4[judged]))
    results.append((split, "4-season", "rule", ok4, int(judged.sum()),
                    *wilson(ok4, int(judged.sum())), None))
    results.append((split, "4-season", "softmax", c4_soft, len(te),
                    *wilson(c4_soft, len(te)), None))
    rule_m = np.array([r["rule_modifier"] for r in rows], dtype=object)[te]
    ok = int(np.sum(rule_m == mod[te]))
    results.append((split, "modifier", "rule", ok, len(te),
                    *wilson(ok, len(te)), None))
    rule_f = np.array([r["rule_family"] for r in rows], dtype=object)[te]
    judged = np.isin(rule_f, ["Warm", "Cool"])
    ok = int(np.sum(rule_f[judged] == fam[te][judged]))
    results.append((split, "family", "rule", ok, int(judged.sum()),
                    *wilson(ok, int(judged.sum())),
                    f"{int((~judged).sum())} abstain"))

    # ================= 10-fold CV (supporting) =================
    def stratified_folds(seasons, kk, seed=42):
        rng = np.random.default_rng(seed)
        folds = np.zeros(len(seasons), dtype=int)
        for cls in sorted(set(seasons)):
            idx = np.where(seasons == cls)[0]
            idx = idx[rng.permutation(len(idx))]
            for f in range(kk):
                folds[idx[f::kk]] = f
        return folds

    def run_cv(X, y, folds, fit_predict):
        preds = np.empty(len(y), dtype=object)
        per_fold = []
        for f in range(k):
            ftr = np.where(folds != f)[0]
            fte = np.where(folds == f)[0]
            p, truth = fit_eval(X, y, ftr, fte, fit_predict)
            preds[fte] = p
            per_fold.append(float(np.mean(p == truth)))
        return preds, per_fold

    folds = stratified_folds(season, k)
    for fs in ("skin-hsv", "all-signals"):
        X = np.array([[float(r[c]) for c in FAMILY_FEATURES[fs]]
                      for r in rows])
        p, pf = run_cv(X, y01, folds,
                       lambda Xtr, ytr, Xte: logreg_predict(Xtr, ytr, Xte, 0.5))
        p = np.where(p == 1, "Warm", "Cool")
        ok = int(np.sum(p == fam))
        results.append((f"{k}-fold", "family", f"logreg[{fs}]", ok, n,
                        *wilson(ok, n),
                        f"folds {min(pf):.0%}-{max(pf):.0%}"))
    p, pf = run_cv(X, fam, folds,
                   lambda Xtr, ytr, Xte: knn_predict(Xtr, ytr, Xte, 3))
    ok = int(np.sum(p == fam))
    results.append((f"{k}-fold", "family", "knn-3[skin-hsv]", ok, n,
                    *wilson(ok, n), f"folds {min(pf):.0%}-{max(pf):.0%}"))
    p, pf = run_cv(Xm, mod, folds,
                   lambda Xtr, ytr, Xte: np.array(softmax_predict(
                       Xtr, ytr, Xte, mclasses, 1.0)))
    ok = int(np.sum(p == mod))
    results.append((f"{k}-fold", "modifier", "softmax", ok, n,
                    *wilson(ok, n), f"folds {min(pf):.0%}-{max(pf):.0%}"))
    p, pf = run_cv(Xs, season, folds,
                   lambda Xtr, ytr, Xte: np.array(softmax_predict(
                       Xtr, ytr, Xte, sclasses, 1.0)))
    ok = int(np.sum(p == season))
    results.append((f"{k}-fold", "season", "softmax", ok, n,
                    *wilson(ok, n), f"folds {min(pf):.0%}-{max(pf):.0%}"))

    # ================= output =================
    print()
    cur = None
    for split_, target, model, ok, judged, lo, hi, extra in results:
        if split_ != cur:
            cur = split_
            print(f"--- {split_} ---")
        extra_s = f"  [{extra}]" if extra else ""
        print(f"  {target:9s} {model:18s}: {fmt_ci(ok, judged)}{extra_s}")

    with open(VAL_DIR / "kfold_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "target", "model", "correct", "judged", "acc",
                    "ci_lo", "ci_hi", "extra"])
        for split_, target, model, ok, judged, lo, hi, extra in results:
            w.writerow([split_, target, model, ok, judged, f"{ok / judged:.4f}",
                        f"{lo:.4f}", f"{hi:.4f}", extra or ""])

    lines = ["# Deep Armocromia Evaluation — Official Split + 10-Fold CV\n",
             f"**Date:** 2026-11-05 | **Tool:** kfold_validation.py | "
             f"**n:** {n} faces\n",
             "## Official split (train 4,008 -> held-out test 912)\n",
             "| Target | Model | Accuracy | Notes |", "|---|---|---|---|"]
    for split_, target, model, ok, judged, lo, hi, extra in results:
        if split_ != "official-test":
            continue
        lines.append(f"| {target} | {model} | {fmt_ci(ok, judged)} | "
                     f"{extra or ''} |")
    lines.append("")
    lines.append("## 10-fold CV (supporting, stratified by season, seed 42)\n")
    lines.append("| Target | Model | Accuracy | Fold spread |")
    lines.append("|---|---|---|---|")
    for split_, target, model, ok, judged, lo, hi, extra in results:
        if split_ != f"{k}-fold":
            continue
        lines.append(f"| {target} | {model} | {fmt_ci(ok, judged)} | "
                     f"{extra or ''} |")
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("- Official split = the Deep Armocromia benchmark split; "
                 "the authors' CNN baselines are ~55% (4-season) and ~32% "
                 "(12-subtype) — our numbers are directly comparable")
    lines.append("- The fixed rule's official-test row is its first genuine "
                 "out-of-sample test (thresholds calibrated on 14 unrelated "
                 "faces)")
    lines.append("- Adjacent = wheel neighbors; published work reports ~96% "
                 "of automated errors land in adjacent categories")
    lines.append("- Models: numpy-only logistic/softmax regression and k-NN "
                 "on the deterministic pipeline's HSV features")
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nsummary -> {VAL_DIR / 'kfold_summary.csv'}")
    print(f"report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
