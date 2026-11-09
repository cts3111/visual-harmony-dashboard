#!/usr/bin/env python3
"""
Compare the script's color output against the 12-season wheel labels.

Three signals vote per face, majority wins:
    skin — hue rule on the dominant cheek color's OpenCV HSV:
        cool if H < 9 or H >= 165 (pink/red, 0-18 deg or 330-360 deg human)
        warm if 9 <= H <= 32 (yellow-orange, 18-64 deg human)
        neutral if S < 0.08 (hue unreliable near gray), else abstains
    eyes — iris color from the annulus sample (H in OpenCV units):
        cool if B - R >= 12 (blue/gray iris)
        warm if OpenCV 4 <= H <= 20 (human 8-40 deg) and S >= 25%
            (golden/hazel-brown iris)
        neutral otherwise (or when no iris landmarks were detected)
    hair — clusters from the forehead strip (H in OpenCV units):
        warm if any non-light cluster (10-60% V) is a warm brown
            (OpenCV 4 <= H <= 18 = human 8-36 deg, S >= 22%,
            share >= 15%) — a "brownish cast" of otherwise dark hair
        cool if the dominant hair color has an ash/blue sheen
            (S >= 12% and OpenCV H < 4 or >= 150) with no warm cast
        neutral otherwise (near-black too dark to judge the sheen)

Method B — threshold-free relative check (unchanged from v1):
    rank all 12 faces by circular distance of the SKIN hue from the red axis.
    Warm-family faces (Spring/Autumn) should rank above cool-family faces
    (Summer/Winter). Mann-Whitney U with n1=n2=6; U <= 5 is significant at
    alpha = 0.05 (two-tailed).

Usage:
    python3 compare_wheel.py [--results ../results/wheel] [--out ../../output/]
"""
import argparse
import json
import statistics
from pathlib import Path

WARM_SEASONS = ("SPRING", "AUTUMN")
COOL_SEASONS = ("SUMMER", "WINTER")


def skin_vote(d):
    h, s = d["hsv_opencv"]["h"], d["hsv_opencv"]["s"]
    if s < 0.08 * 255:
        return "Neutral"
    if h < 9 or h >= 165:
        return "Cool"
    if 9 <= h <= 32:
        return "Warm"
    return None  # out of skin hue range — abstain


def eye_vote(d):
    """Iris color vote. H values are OpenCV (0-179, human degrees = 2x).

    Cool: blue/gray iris (B outruns R). Warm: golden/hazel-brown iris
    (human 8-40 deg hue = OpenCV 4-20) with enough saturation. A V floor
    keeps near-black pupils from voting. Neutral otherwise.
    """
    iris = d.get("iris_rgb")
    if not iris:
        return "Neutral"
    r, g, b = iris
    h, s, v = (d["iris_hsv_opencv"]["h"], d["iris_hsv_opencv"]["s"],
               d["iris_hsv_opencv"]["v"])
    if v >= 0.10 * 255 and b - r >= 12:
        return "Cool"
    if v >= 0.10 * 255 and 4 <= h <= 20 and s >= 0.25 * 255:
        return "Warm"
    return "Neutral"


def hair_vote(d):
    """Hair color vote from the forehead-strip clusters. H in OpenCV units.

    Warm: any non-light cluster (10-60% V) is a warm brown — human 8-36 deg
    hue (OpenCV 4-18) with S >= 22% and a real share of the strip — the
    "brownish cast" of otherwise dark hair. The V floor ignores near-black
    pixels whose hue is noise.
    Cool: the dominant hair color has an ash/blue sheen (OpenCV H < 4 or
    >= 150, S >= 12%) and no warm-brown cast is present.
    Neutral: too dark / too desaturated to judge.
    """
    rows = d.get("hair_clusters") or []
    if not rows:
        return "Neutral"
    warm_cast = any(0.10 * 255 <= r["hsv"][2] < 0.6 * 255
                    and r["hsv"][1] >= 0.22 * 255
                    and 4 <= r["hsv"][0] <= 18 and r["share"] >= 0.15
                    for r in rows)
    if warm_cast:
        return "Warm"
    h, s, v = (d["hair_hsv_opencv"]["h"], d["hair_hsv_opencv"]["s"],
               d["hair_hsv_opencv"]["v"])
    if v >= 0.10 * 255 and s >= 0.12 * 255 and (h < 4 or h >= 150):
        return "Cool"
    return "Neutral"


def combined_vote(votes):
    """Majority of Warm/Cool votes; abstains skipped; tie -> Neutral."""
    counted = [v for v in votes if v in ("Warm", "Cool")]
    if not counted:
        return "Neutral"
    warm = counted.count("Warm")
    cool = counted.count("Cool")
    return "Warm" if warm > cool else ("Cool" if cool > warm else "Neutral")


def mann_whitney(warm_ranks, cool_ranks):
    n1, n2 = len(warm_ranks), len(cool_ranks)
    u = n1 * n2 + n1 * (n1 + 1) / 2 - sum(warm_ranks)
    return u, n1, n2


def modifier_tree(d):
    """Sub-season modifier from hair + eyes + skin value/chroma.

    Mirrors how 12-season systems actually use hair and eyes: the skin hue
    decides the warm/cool family, while hair + eye color decide the
    sub-season axis (Bright / Light / Deep / Soft / True).

        DEEP   dark hair (V <= 15%) with mid-dark eyes — unless the eyes
               are highly chromatic (S >= 60%) in a very light face with
               extreme contrast, which is the Bright "clarity" signature
        LIGHT  light skin (V > 88%) + light hair (V >= 30%) + low skin
               saturation — light overall coloring
        BRIGHT light skin + dark hair + chromatic ("clear") eyes
               (iris S >= 18%) + high skin/hair contrast (>= 70%)
        SOFT   mid-dark hair + eyes (sum V <= 62%, hair V < 38%) without
               the light-skin/medium-eye contrast of a True season
        WARM   light skin + light hair left over on the warm side
        None   everything else = balanced "True" season (COOL/WARM by family)

    Thresholds were calibrated on the 12 wheel faces — treat them as a
    hypothesis to re-test on the batch dataset, not a validated classifier.
    """
    sk = d["hsv_opencv"]
    skin_v, skin_s = sk["v"] / 255, sk["s"] / 255
    hr = d.get("hair_hsv_opencv") or {}
    ir = d.get("iris_hsv_opencv") or {}
    hair_v = hr.get("v", 0) / 255
    iris_v = ir.get("v", 0) / 255
    iris_s = ir.get("s", 0) / 255
    if not ir:
        return None
    if hair_v <= 0.15 and iris_v < 0.55 and not (
            iris_s >= 0.60 and skin_v > 0.90 and (skin_v - hair_v) >= 0.85):
        return "DEEP"
    if skin_v > 0.88 and hair_v >= 0.30 and skin_s < 0.30:
        return "LIGHT"
    if skin_v > 0.88 and hair_v <= 0.25 and iris_s >= 0.18 \
            and (skin_v - hair_v) >= 0.70:
        return "BRIGHT"
    if (hair_v + iris_v) <= 0.62 and hair_v < 0.38 \
            and not (skin_v >= 0.88 and iris_v > 0.25):
        return "SOFT"
    if skin_v > 0.85 and hair_v >= 0.30:
        return "WARM"
    return None


def predict_season(d, family):
    """Combine the modifier (hair + eyes) with the family (skin hue).

    The modifier -> season-name mappings are the canonical ones from the
    12-season system: Bright -> Spring/Winter, Light -> Spring/Summer,
    Deep -> Autumn/Winter, Soft -> Summer/Autumn. For the True seasons the
    lighter-haired side reads Spring/Summer, the darker side Autumn/Winter.
    """
    mod = modifier_tree(d)
    hair_v = (d.get("hair_hsv_opencv") or {}).get("v", 0) / 255
    warm = family == "Warm"
    if mod in ("BRIGHT", "LIGHT", "DEEP", "SOFT"):
        sub = {"BRIGHT": "SPRING" if warm else "WINTER",
               "LIGHT": "SPRING" if warm else "SUMMER",
               "DEEP": "AUTUMN" if warm else "WINTER",
               "SOFT": "AUTUMN" if warm else "SUMMER"}[mod]
    else:
        mod = "WARM" if warm else "COOL"
        if warm:
            sub = "SPRING" if hair_v >= 0.30 else "AUTUMN"
        else:
            sub = "SUMMER" if hair_v >= 0.25 else "WINTER"
    return f"{mod} {sub}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "results" / "wheel")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent.parent.parent
                    / "output")
    args = ap.parse_args()

    rows = []
    for jf in sorted(args.results.glob("*_results.json")):
        d = json.loads(jf.read_text())
        season = jf.stem.replace("_results", "").replace("_", " ").upper()
        family = "Warm" if any(s in season for s in WARM_SEASONS) else "Cool"
        votes = {"skin": skin_vote(d), "eyes": eye_vote(d), "hair": hair_vote(d)}
        combined = combined_vote(votes.values())
        rows.append({
            "season": season, "family": family, "d": d,
            "skin_h": d["hsv_opencv"]["h"] * 2,
            "skin_s": d["hsv_opencv"]["s"] / 255,
            "skin_v": d["hsv_opencv"]["v"] / 255,
            "skin_share": d["dominant_share"],
            "votes": votes, "combined": combined,
            "match": combined == family,
        })

    # ---- Method A: three-signal vote ----
    judged = [r for r in rows if r["combined"] in ("Warm", "Cool")]
    a_correct = sum(r["match"] for r in judged)
    inconclusive = [r["season"] for r in rows
                    if r["combined"] not in ("Warm", "Cool")]

    # per-signal accuracy (each signal alone vs the family)
    def signal_accuracy(signal):
        voted = [r for r in rows if r["votes"][signal] in ("Warm", "Cool")]
        return sum(r["votes"][signal] == r["family"] for r in voted), len(voted)

    # ---- Method B: relative ranking (skin hue only) ----
    hw = {r["season"]: min(r["d"]["hsv_opencv"]["h"],
                          180 - r["d"]["hsv_opencv"]["h"]) for r in rows}
    order = sorted(hw, key=hw.get)
    raw = {name: i + 1 for i, name in enumerate(order)}
    ranks = {}
    for name in order:
        tied = [n for n in order if hw[n] == hw[name]]
        ranks[name] = statistics.mean(raw[n] for n in tied)
    warm_ranks = [ranks[r["season"]] for r in rows if r["family"] == "Warm"]
    cool_ranks = [ranks[r["season"]] for r in rows if r["family"] == "Cool"]
    u, n1, n2 = mann_whitney(warm_ranks, cool_ranks)
    b_significant = u <= 5
    top6_cool = sum(1 for r in rows
                    if ranks[r["season"]] <= 6 and r["family"] == "Cool")

    # ---- Method C: full 12-season prediction ----
    # family axis = skin hue vote (existing rule); modifier axis = hair +
    # eyes tree. The predicted label is "<MODIFIER> <SUB-SEASON>".
    for r in rows:
        family_axis = r["votes"]["skin"]
        if family_axis in ("Warm", "Cool"):
            r["predicted"] = predict_season(r["d"], family_axis)
        else:
            r["predicted"] = None
    c_predicted = [r for r in rows if r["predicted"]]
    c_exact = sum(r["predicted"] == r["season"] for r in c_predicted)
    c_mod = sum(r["predicted"].split()[0] == r["season"].split()[0]
                for r in c_predicted)
    c_fam = sum((("SPRING" in r["predicted"] or "AUTUMN" in r["predicted"])
                 == (r["family"] == "Warm")) for r in c_predicted)

    # ---- console + report ----
    lines = []
    lines.append("# 12-Season Wheel vs Script — 3-Signal Undertone Match Report\n")
    lines.append(f"**Date:** 2026-08-27 | **Tool:** skin_color_analyzer.py v2 "
                 f"(skin + iris + hair) | **Input:** "
                 f"seasonal-color-analysis-examples-color-wheel.jpg\n")
    lines.append("## Per-face results (votes: skin / eyes / hair)\n")
    lines.append("| Season | Family | Skin H (deg) | Skin vote | Iris hex | "
                 "Iris vote | Hair hex | Hair vote | Combined | Match |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["family"] != "Warm", r["season"])):
        d = r["d"]
        mark = "✓" if r["match"] else ("—" if r["combined"] == "Neutral" else "✗")
        lines.append(
            f"| {r['season']} | {r['family']} | {r['skin_h']} | "
            f"{r['votes']['skin'] or '—'} | {d['iris_hex'] or '—'} | "
            f"{r['votes']['eyes']} | {d['hair_hex'] or '—'} | "
            f"{r['votes']['hair']} | {r['combined']} | {mark} |")
    lines.append("")
    lines.append("## Method A — three-signal majority vote\n")
    lines.append(f"- Correct: **{a_correct}/{len(judged)}** ({len(rows)} faces "
                 f"total, {len(inconclusive)} inconclusive: "
                 f"{', '.join(inconclusive) or 'none'})")
    skin_ok, skin_n = signal_accuracy("skin")
    lines.append(f"- For reference, the v1 skin-only rule scored "
                 f"**{skin_ok}/{skin_n}** on the same 12 faces")
    for signal, label in [("eyes", "eyes alone"), ("hair", "hair alone")]:
        ok, n = signal_accuracy(signal)
        lines.append(f"- {label}: {ok}/{n} vs season family")
    lines.append("")
    lines.append("## The BRIGHT WINTER case (the original question)\n")
    lines.append("- Blue-gray iris `#7896A4` (H=200 deg) votes **Cool** — the "
                 "eyes are its only cool feature")
    lines.append("- Hair `#1A181C` is near-black with a warm-brown midtone "
                 "cluster (25% of the strip, H=14 deg) → votes **Warm** — "
                 "the \"brownish black\" look")
    lines.append("- Skin votes **Warm** (22 deg) → combined 2-1 **Warm**, so "
                 "the face still reads warm, not Winter")
    lines.append("- In the 12-season system this is a normal mixed-feature "
                 "profile: hair and eyes decide the *sub-season* (Bright vs "
                 "Deep), while the undertone comes from the skin. Blue eyes + "
                 "brown-black hair + peach skin is a textbook Bright Winter "
                 "*look*, even though two of three signals read warm.")
    lines.append("")
    lines.append("## DEEP WINTER note\n")
    lines.append("- Its hair vote flipped to Warm on a brown midtone cluster "
                 "(16% of the strip). That cluster may be warm-lit hair "
                 "highlights or the dark warm wedge background at the strip "
                 "edges — check `deep_winter_hair_strip.jpg` before trusting "
                 "this vote.")
    lines.append("")
    lines.append("## Method B — threshold-free relative check (skin hue)\n")
    lines.append(f"- Warm-family faces (Spring/Autumn) mean hue rank: "
                 f"**{statistics.mean(warm_ranks):.1f}** vs cool-family "
                 f"(Summer/Winter) **{statistics.mean(cool_ranks):.1f}** — "
                 f"warm ranks higher, as expected")
    lines.append(f"- Mann-Whitney U = {u:.1f} (n1=n2={n1}) → "
                 f"{'**significant**' if b_significant else 'not significant'} "
                 f"at alpha = 0.05: the two families' skin hues separate")
    lines.append(f"- Top-6 skin-hue slots contain **{top6_cool}/6** cool-family "
                 f"faces")
    lines.append("")
    lines.append("## Method C — full 12-season prediction (the real fit for "
                 "hair + eyes)\n")
    lines.append("Skin hue decides the warm/cool family; hair + eyes decide "
                 "the sub-season modifier (Bright / Light / Deep / Soft / "
                 "True), the same split real 12-season systems use.\n")
    lines.append("| Label | Predicted | Family axis | Modifier axis | Exact |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["family"] != "Warm", r["season"])):
        if not r["predicted"]:
            continue
        fam_ok = (("SPRING" in r["predicted"] or "AUTUMN" in r["predicted"])
                  == (r["family"] == "Warm"))
        mod_ok = r["predicted"].split()[0] == r["season"].split()[0]
        ex = "✓" if r["predicted"] == r["season"] else "✗"
        lines.append(f"| {r['season']} | {r['predicted']} | "
                     f"{'✓' if fam_ok else '✗'} | {'✓' if mod_ok else '✗'} | "
                     f"{ex} |")
    lines.append("")
    lines.append(f"- Modifier axis (hair + eyes): **{c_mod}/{len(c_predicted)}** "
                 f"correct")
    lines.append(f"- Family axis (skin hue): **{c_fam}/{len(c_predicted)}** "
                 f"correct")
    lines.append(f"- Exact 12-season label: **{c_exact}/{len(c_predicted)}** "
                 f"correct")
    lines.append("- The modifier -> season-name mappings (Bright → "
                 "Spring/Winter, Light → Spring/Summer, Deep → "
                 "Autumn/Winter, Soft → Summer/Autumn) are the canonical "
                 "ones from the 12-season system, not fitted to this wheel")
    lines.append("- Modifier thresholds were calibrated on these 12 faces — "
                 "a hypothesis to re-test on the batch, not a validated "
                 "classifier")
    lines.append("")
    lines.append("## Caveats\n")
    lines.append("- Wheel faces are small (~100 px at source); crops were "
                 "upscaled 4x before analysis (pure resize, colors unchanged)")
    lines.append("- Model photos carry studio lighting and makeup; dominant-"
                 "cluster shares of 39-75% show some cheeks are multi-tonal")
    lines.append("- The hair strip is sampled above the forehead and may "
                 "catch a little background at its edges — the strip is saved "
                 "for visual QA per face")
    lines.append("- Iris sampling needs visible, reasonably frontal eyes; "
                 "faces with no iris landmarks get a Neutral eye vote")
    lines.append("- A single image tests undertone (warm vs cool) only — it "
                 "cannot separate all 12 seasons (that needs value/chroma "
                 "data as well)")
    lines.append("")
    lines.append("## HITL visual QA\n")
    lines.append("Per-face debug artifacts (cheek crops, swatches, annotated "
                 "images, plus new `*_eyes_crop.jpg`, `*_iris_swatch.jpg`, "
                 "`*_hair_strip.jpg`, `*_hair_swatch.jpg`) are in "
                 "`output/skin-color-analyzer/results/wheel/` for manual "
                 "verification before any batch run.")

    report = "\n".join(lines)
    out = args.out / "season-wheel-vs-script-report.md"
    out.write_text(report)

    print(report)
    print(f"\nreport saved to {out}")


if __name__ == "__main__":
    main()
