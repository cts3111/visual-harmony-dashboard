#!/usr/bin/env python3
"""
Leave-one-out cross-validation of the extracted color features against the
14 hand-labeled faces (12 from the season wheel + 2 independent controls).

Targets (what the data can support):
    FAMILY    warm/cool — 7 warm / 7 cool. Chance = majority = 50%.
    MODIFIER  5 classes (Bright/Light/Deep/Soft/True) — counts 4/2/2/2/4.
              Chance = majority = 28.6%.

Exact 12-season LOOCV is structurally impossible here: 12 of the 14 labels
are singleton classes, so a held-out season usually has zero training
examples and can never be predicted. That gap is the argument for growing
the labeled set (see report).

Models (numpy-only, deterministic, no new dependencies):
    logistic regression (L2, binary for family, softmax for modifier)
    k-NN (k=1 and k=3, Euclidean on per-fold standardized features)

The fixed deterministic rule from compare_wheel (skin-hue family axis,
hair+eyes modifier tree) is scored on the same 14 faces as the reference.

Usage:
    python3 loocv_validation.py

Outputs:
    ../results/validation/loocv_folds.csv        per-face, per-model predictions
    ../../../output/loocv-validation-report.md   summary report
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from compare_wheel import modifier_tree, predict_season, skin_vote

SRC_DIR = Path(__file__).resolve().parent
WHEEL_DIR = SRC_DIR.parent / "results" / "wheel"
BW_DIR = SRC_DIR.parent / "results" / "bright_winter"
VAL_DIR = SRC_DIR.parent / "results" / "validation"
REPORT_PATH = SRC_DIR.parents[2] / "output" / "loocv-validation-report.md"

CONTROL_LABELS = {
    "bright_winter_example": "BRIGHT WINTER",
    "bright_spring_control": "BRIGHT SPRING",
}
TRUE_MODIFIERS = ("WARM", "COOL")

FAMILY_FEATURES = {
    "skin-hue": ["skin_red_dist"],
    "skin-hsv": ["skin_red_dist", "skin_s", "skin_v"],
    "all-signals": ["skin_red_dist", "skin_s", "skin_v",
                    "iris_h", "iris_s", "iris_v",
                    "hair_h", "hair_s", "hair_v", "has_iris", "has_hair"],
}
MOD_FEATURES = ["skin_v", "skin_s", "hair_v", "iris_v", "iris_s", "contrast"]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def season_from_stem(stem):
    if stem in CONTROL_LABELS:
        return CONTROL_LABELS[stem]
    return stem.replace("_", " ").upper()


def load_faces():
    """Return a list of (label, family, modifier, feature_dict, raw_json)."""
    faces = []
    for jf in sorted(list(WHEEL_DIR.glob("*_results.json"))
                     + list(BW_DIR.glob("*_results.json"))):
        d = json.loads(jf.read_text())
        label = season_from_stem(jf.stem.replace("_results", ""))
        family = "Warm" if any(s in label
                               for s in ("SPRING", "AUTUMN")) else "Cool"
        mod = label.split()[0]
        if mod in TRUE_MODIFIERS:
            mod = "TRUE"
        faces.append((label, family, mod, extract_features(d), d))
    return faces


def extract_features(d):
    sk = d["hsv_opencv"]
    f = {"skin_red_dist": min(sk["h"], 180 - sk["h"]),
         "skin_s": sk["s"] / 255, "skin_v": sk["v"] / 255}
    ir = d.get("iris_hsv_opencv")
    if ir:
        f.update(iris_h=ir["h"], iris_s=ir["s"] / 255,
                 iris_v=ir["v"] / 255, has_iris=1.0)
    else:
        f.update(iris_h=0.0, iris_s=0.0, iris_v=0.0, has_iris=0.0)
    hr = d.get("hair_hsv_opencv")
    if hr:
        f.update(hair_h=hr["h"], hair_s=hr["s"] / 255,
                 hair_v=hr["v"] / 255, has_hair=1.0)
    else:
        f.update(hair_h=0.0, hair_s=0.0, hair_v=0.0, has_hair=0.0)
    f["contrast"] = f["skin_v"] - f["hair_v"]
    return f


# ---------------------------------------------------------------------------
# Models (numpy-only)
# ---------------------------------------------------------------------------
def standardize(X_tr, X_te):
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X_tr - mu) / sd, (X_te - mu) / sd


def logreg_fit(X, y, lam, iters=5000, lr=0.5, tol=1e-6):
    """Binary logistic regression via full-batch GD. y in {0, 1}.

    The gradient is norm-clipped at 5 so early oscillation from extreme
    outliers cannot blow the weights up; clipping is inactive near the
    optimum, so the converged solution is unchanged.
    """
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    prev = math.inf
    for _ in range(iters):
        z = np.clip(X @ w + b, -30, 30)  # keep np.exp finite under outliers
        p = 1.0 / (1.0 + np.exp(-z))
        loss = -(y @ np.log(p + 1e-12) + (1 - y) @ np.log(1 - p + 1e-12)) / n \
            + 0.5 * lam * (w @ w)
        if abs(prev - loss) < tol:
            break
        prev = loss
        g = X.T @ (p - y) / n + lam * w
        gnorm = float(np.linalg.norm(g))
        if gnorm > 5.0:
            g = g * (5.0 / gnorm)
        w -= lr * g
        b -= lr * (p - y).mean()
    return w, b


def logreg_predict(X_tr, y_tr, X_te, lam, iters=5000):
    w, b = logreg_fit(X_tr, y_tr, lam=lam, iters=iters)
    z = np.clip(X_te @ w + b, -30, 30)
    p = 1.0 / (1.0 + np.exp(-z))
    return (p >= 0.5).astype(int)


def softmax_fit(X, Y, lam, iters=5000, lr=0.5, tol=1e-6):
    """Multinomial (softmax) regression. Y is (n, K) one-hot."""
    n, d = X.shape
    K = Y.shape[1]
    W, b = np.zeros((d, K)), np.zeros(K)
    prev = math.inf
    for _ in range(iters):
        z = np.clip(X @ W + b, -30, 30)  # keep np.exp finite under outliers
        z = z - z.max(axis=1, keepdims=True)
        P = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        loss = -np.sum(Y * np.log(P + 1e-12)) / n + 0.5 * lam * (W * W).sum()
        if abs(prev - loss) < tol:
            break
        prev = loss
        g = X.T @ (P - Y) / n + lam * W
        gnorm = float(np.linalg.norm(g))
        if gnorm > 5.0:
            g = g * (5.0 / gnorm)
        W -= lr * g
        b -= lr * (P - Y).mean(axis=0)
    return W, b


def softmax_predict(X_tr, y_tr, X_te, classes, lam, iters=8000):
    Y = np.zeros((len(y_tr), len(classes)))
    for i, c in enumerate(classes):
        Y[:, i] = (y_tr == c)
    W, b = softmax_fit(X_tr, Y, lam=lam, iters=iters)
    z = np.clip(X_te @ W + b, -30, 30)
    z = z - z.max(axis=1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return [classes[i] for i in np.argmax(P, axis=1)]


def knn_predict(X_tr, y_tr, X_te, k):
    """k-NN majority vote; distance breaks ties. Inputs pre-standardized."""
    preds = []
    for x in X_te:
        dist = ((X_tr - x) ** 2).sum(axis=1)
        order = np.argsort(dist)[:k]
        votes = {}
        for i in order:
            votes.setdefault(y_tr[i], []).append(dist[i])
        ranked = sorted(votes, key=lambda c: (-len(votes[c]), min(votes[c])))
        preds.append(ranked[0])
    return preds


def loocv_predict(X, y, fit_predict):
    """LOOCV: fit_predict receives per-fold standardized (X_tr, y_tr, X_te)."""
    n = len(y)
    preds = [None] * n
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        X_tr, X_te = standardize(X[tr], X[i:i + 1])
        y_tr = np.array([y[j] for j in tr])
        preds[i] = fit_predict(X_tr, y_tr, X_te)[0]
    return preds


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def accuracy(preds, truth):
    ok = sum(p == t for p, t in zip(preds, truth))
    lo, hi = wilson(ok, len(preds))
    return ok, len(preds), lo, hi


def main():
    faces = load_faces()
    n = len(faces)
    labels = [f[0] for f in faces]
    fam_true = np.array([f[1] for f in faces])
    mod_true = np.array([f[2] for f in faces])

    # ---- fitted models: family (binary) ----
    fam_models = []
    for fs_name, fs in FAMILY_FEATURES.items():
        X = np.array([[f[3][k] for k in fs] for f in faces])
        y = np.array([1 if t == "Warm" else 0 for t in fam_true])
        preds01 = loocv_predict(
            X, y, lambda Xtr, ytr, Xte: logreg_predict(Xtr, ytr, Xte, 0.5))
        preds = ["Warm" if p else "Cool" for p in preds01]
        fam_models.append(("logreg", fs_name, preds))
    X = np.array([[f[3][k] for k in FAMILY_FEATURES["skin-hsv"]]
                  for f in faces])
    preds = loocv_predict(
        X, fam_true, lambda Xtr, ytr, Xte: knn_predict(Xtr, ytr, Xte, 3))
    fam_models.append(("knn-3", "skin-hsv", preds))

    # ---- fitted models: modifier (5-class) ----
    mod_models = []
    classes = sorted(set(mod_true))
    Xm = np.array([[f[3][k] for k in MOD_FEATURES] for f in faces])
    preds = loocv_predict(
        Xm, mod_true,
        lambda Xtr, ytr, Xte: softmax_predict(Xtr, ytr, Xte, classes, 1.0))
    mod_models.append(("softmax", "mod-features", preds))
    for k in (1, 3):
        preds = loocv_predict(
            Xm, mod_true, lambda Xtr, ytr, Xte: knn_predict(Xtr, ytr, Xte, k))
        mod_models.append((f"knn-{k}", "mod-features", preds))

    # ---- reference: the fixed deterministic rule ----
    rule_fam, rule_mod, rule_exact = [], [], []
    for label, family, mod, feats, d in faces:
        sv = skin_vote(d)
        rule_fam.append(sv if sv in ("Warm", "Cool") else None)
        mt = modifier_tree(d)
        # The tree's WARM/COOL/None outputs are all the True axis.
        rule_mod.append("TRUE" if mt in (None, "WARM", "COOL") else mt)
        rule_exact.append(predict_season(d, sv)
                          if sv in ("Warm", "Cool") else None)

    # ---- results tables ----
    fam_chance = max(np.mean(fam_true == "Warm"), np.mean(fam_true == "Cool"))
    mod_chance = max(np.mean(mod_true == c) for c in classes)
    print(f"n = {n} faces | family chance (majority) = {fam_chance:.0%} | "
          f"modifier chance = {mod_chance:.0%}\n")

    print("FAMILY (warm/cool) — LOOCV")
    fam_summary = []
    for name, fs, preds in fam_models:
        ok, tot, lo, hi = accuracy(preds, fam_true)
        fam_summary.append((name, fs, ok, tot, lo, hi))
        print(f"  {name:7s} [{fs:12s}]: {ok}/{tot}  "
              f"({ok / tot:.0%}, Wilson 95% [{lo:.0%}-{hi:.0%}])")
    judged = [p for p in rule_fam if p is not None]
    rule_fam_ok = sum(p == t for p, t in zip(rule_fam, fam_true)
                      if p is not None)
    print(f"  rule    [skin-hue axis]: {rule_fam_ok}/{len(judged)} judged "
          f"({len(faces) - len(judged)} abstain)\n")

    print("MODIFIER (5-class) — LOOCV")
    mod_summary = []
    for name, fs, preds in mod_models:
        ok, tot, lo, hi = accuracy(preds, mod_true)
        mod_summary.append((name, fs, ok, tot, lo, hi))
        print(f"  {name:7s} [{fs:12s}]: {ok}/{tot}  "
              f"({ok / tot:.0%}, Wilson 95% [{lo:.0%}-{hi:.0%}])")
    rule_mod_ok = sum(p == t for p, t in zip(rule_mod, mod_true))
    lo, hi = wilson(rule_mod_ok, n)
    print(f"  rule    [hair+eyes tree]: {rule_mod_ok}/{n}  "
          f"({rule_mod_ok / n:.0%}, Wilson 95% [{lo:.0%}-{hi:.0%}])")
    rule_exact_ok = sum(p == t for p, t in zip(rule_exact, labels)
                        if p is not None)
    rule_exact_n = sum(1 for p in rule_exact if p is not None)
    print(f"  rule    [exact 12-season]: {rule_exact_ok}/{rule_exact_n} "
          f"correct\n")

    # ---- per-fold CSV ----
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, label in enumerate(labels):
        row = {"face": label, "family_true": fam_true[i],
               "modifier_true": mod_true[i]}
        for name, fs, preds in fam_models:
            row[f"fam_{name}_{fs}"] = preds[i]
        for name, fs, preds in mod_models:
            row[f"mod_{name}_{fs}"] = preds[i]
        row["rule_family"] = rule_fam[i]
        row["rule_modifier"] = rule_mod[i]
        row["rule_exact"] = rule_exact[i]
        rows.append(row)
    with open(VAL_DIR / "loocv_folds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- report ----
    lines = ["# LOOCV Validation — Predictive Power on 14 Labeled Faces\n",
             f"**Date:** 2026-11-05 | **Tool:** loocv_validation.py | "
             f"**n:** {n} (12 wheel faces + 2 independent controls)\n",
             "## TL;DR\n"]
    for name, fs, ok, tot, lo, hi in fam_summary:
        lines.append(f"- Family (warm/cool) — {name} [{fs}]: **{ok}/{tot} "
                     f"({ok / tot:.0%})**, Wilson 95% [{lo:.0%}-{hi:.0%}] "
                     f"vs {fam_chance:.0%} chance")
    lines.append(f"- Family — fixed rule (skin-hue axis): {rule_fam_ok}/"
                 f"{len(judged)} judged (in-sample reference)")
    for name, fs, ok, tot, lo, hi in mod_summary:
        lines.append(f"- Modifier — {name} [{fs}]: **{ok}/{tot} "
                     f"({ok / tot:.0%})**, Wilson 95% [{lo:.0%}-{hi:.0%}] "
                     f"vs {mod_chance:.0%} chance")
    lines.append(f"- Modifier — fixed rule (hair+eyes tree): "
                 f"{rule_mod_ok}/{n} (in-sample reference)")
    lines.append(f"- Exact 12-season — fixed rule: {rule_exact_ok}/"
                 f"{rule_exact_n} (in-sample reference)")
    lines.append(f"- Exact 12-season LOOCV: **not possible with n={n}** — "
                 f"12 singleton classes (see below)\n")
    lines.append("## Interpretation\n")
    lines.append("- **Family axis has real signal.** LOOCV 79-86% against a "
                 "50% chance baseline — skin hue separates warm from cool "
                 "even on held-out faces. This is the one out-of-sample "
                 "predictive-power claim the current data supports.")
    lines.append("- **Modifier axis is not learnable at n=14.** The fitted "
                 "models score 0-7%, below the 29% chance baseline: with "
                 "2-4 examples per class the features interleave (a BRIGHT "
                 "face's nearest neighbors are TRUE faces), so no model can "
                 "generalize. The 5-class structure needs more data, not "
                 "better math.")
    lines.append("- **The fixed rule's numbers are in-sample.** Its family "
                 "13/14, modifier 13/14 and exact 12/14 were produced by "
                 "thresholds calibrated on these same faces — they are an "
                 "upper bound, not a validated accuracy. The k-NN family "
                 "result is the fair out-of-sample comparison.")
    lines.append("- **Conclusion:** real predictive power requires the "
                 "dataset expansion (Path B) — several faces per season, "
                 "then a train/test split (the roadmap's Priority 5).\n")
    lines.append("## Method\n")
    lines.append("- Leave-one-out CV; features standardized per fold on the "
                 "training split only")
    lines.append("- Models: L2 logistic regression (binary + softmax) and "
                 "k-NN — numpy-only, deterministic, no new dependencies")
    lines.append("- Wilson 95% interval on accuracy; folds share 12/13 "
                 "training faces, so the interval is approximate")
    lines.append("- The fixed deterministic rule is scored on the same 14 "
                 "faces as the reference point\n")
    lines.append("## Why exact-season LOOCV is impossible right now\n")
    lines.append("- 12 of the 14 labels are singleton classes (one face per "
                 "season). A held-out season has zero training examples, so "
                 "no model can ever predict it — exact-season LOOCV is "
                 "structurally 0% for most folds.")
    lines.append("- Fixing this needs several faces per season — the dataset "
                 "expansion (Path B).\n")
    lines.append("## Caveats\n")
    lines.append("- All 14 faces come from two tightly controlled sources "
                 "(one graphic, one pair of example photos): same lighting, "
                 "same artist — a biased sample, not a population")
    lines.append("- n=14: treat these as directional evidence, not a "
                 "validated accuracy claim")
    lines.append("- Per-face predictions: results/validation/loocv_folds.csv")
    REPORT_PATH.write_text("\n".join(lines))
    print(f"CSV -> {VAL_DIR / 'loocv_folds.csv'}")
    print(f"report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
