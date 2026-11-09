"""Visual Harmony & ROI Analyzer — portfolio dashboard.

Streamlit app: the project's validation and business results, plus a live
demo that runs the production color-analysis pipeline (OpenCV + MediaPipe +
scikit-learn) on an uploaded photo.

Run locally:
    streamlit run app.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from service import CONFIG, run_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WHEEL = RESULTS / "wheel"
ARMOCROMIA = RESULTS / "armocromia"
SAMPLES = ROOT / "data" / "wheel_faces"

# ---------------------------------------------------------------------------
# Palette (validated: scripts/validate_palette.js, light mode, surface #fcfcfb)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"       # categorical slot 1 / emphasis / negative
ORANGE = "#eb6834"     # categorical slot 2
AQUA = "#1baf7a"       # categorical slot 3 (relief rule: direct labels)
RED = "#e34948"        # diverging positive pole
GRAY_FILL = "#c3c2b7"  # de-emphasis fill
GRID = "#e1e0d9"       # gridline hairline
INK = "#0b0b0b"        # primary ink
INK2 = "#52514e"       # secondary ink
MUTED = "#898781"      # axis ticks / muted
SURFACE = "#fcfcfb"    # chart surface
BLUE_RAMP = [(0.0, "#cde2fb"), (0.16, "#86b6ef"), (0.33, "#5598e7"),
             (0.5, "#2a78d6"), (0.66, "#256abf"), (0.83, "#184f95"),
             (1.0, "#0d366b")]

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

st.set_page_config(page_title="Visual Harmony & ROI Analyzer",
                   page_icon="🎨", layout="wide")


def base_layout(fig, height=380, legend=True):
    """Shared chart chrome: surface, hairline grid, ink, muted ticks."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=13),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=GRID, tickfont=dict(color=MUTED),
                     linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, tickfont=dict(color=MUTED),
                     linecolor=GRID)
    return fig


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@st.cache_data
def model_rows():
    """Best-config rows from sklearn_summary.csv (hsv-lab logreg)."""
    rows = {}
    for r in load_csv(ARMOCROMIA / "sklearn_summary.csv"):
        if r["features"] == "hsv-lab" and r["model"] == "logreg":
            rows[r["target"]] = r
    return rows


@st.cache_data
def confusion_matrix():
    rows = load_csv(ARMOCROMIA / "season_confusion.csv")
    labels = list(rows[0].keys())[1:]
    z = [[int(r[c]) for c in labels] for r in rows]
    return labels, z


@st.cache_data
def feature_set_rows():
    """Family accuracy by feature set (official test)."""
    ks = {(r["model"], r["target"]): r
          for r in load_csv(ARMOCROMIA / "kfold_summary.csv")}
    sk = {(r["model"], r["features"], r["target"]): r
          for r in load_csv(ARMOCROMIA / "sklearn_summary.csv")}
    a = ks[("logreg[skin-hsv]", "family")]
    b = ks[("logreg[all-signals]", "family")]
    c = sk[("logreg", "hsv-lab", "family")]
    return [
        ("Skin hue only", float(a["acc"]) * 100),
        ("Skin + iris + hair (HSV)", float(b["acc"]) * 100),
        ("HSV + LAB, all signals", float(c["test_acc"]) * 100),
    ]


@st.cache_data
def wheel_json(stem):
    import json
    return json.loads((WHEEL / f"{stem}_results.json").read_text())


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def chip(hex_color, size=24):
    return (f'<span style="display:inline-block;width:{size}px;height:{size}px;'
            f'background:{hex_color};border:1px solid rgba(11,11,11,.25);'
            f'border-radius:4px;vertical-align:middle"></span>')


def proba_bars(proba, winner, accent=BLUE):
    html = []
    for label, p in list(proba.items())[:4]:
        pct = round(p * 100)
        color = accent if label == winner else GRAY_FILL
        html.append(
            f'<div style="font-size:13px;margin-top:6px">{label} '
            f'<b>{pct}%</b></div>'
            f'<div style="background:{GRID};height:8px;border-radius:4px">'
            f'<div style="background:{color};width:{pct}%;height:8px;'
            f'border-radius:4px"></div></div>')
    return "".join(html)


def metric_card(label, value, delta, help_text):
    st.metric(label, value, delta, help=help_text)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎨 Visual Harmony & ROI Analyzer")
    st.markdown("Does matching a model's skin undertone to the garment's "
                "color drive higher sales? A computer-vision + ML pipeline "
                "measured it on 10,000 Amazon products.")
    st.markdown("---")
    st.markdown("**Data sources**")
    st.markdown("- Deep Armocromia — 4,920 professionally labeled face "
                "images with an official train/test split (color-analysis "
                "benchmark, ECCV Workshops 2024)")
    st.markdown("- DeepFashion2 — garment benchmark (human-drawn masks)")
    st.markdown("- McAuley Amazon-Reviews-2023 — 10k dress products")
    st.markdown("---")
    st.markdown("**Honesty notes**")
    st.markdown("- Face + garment colors are measured from digital photos; "
                "studio lighting shifts them (a candidate explanation for "
                "the null business result)")
    st.markdown("- The live demo runs the exact production pipeline and "
                "the exact tuned models — same code, same files")
    st.markdown("- Educational demo, not a style advisory")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs(["🎨 Live Demo", "📊 Validation", "🧪 Instrument Ladder",
                "💼 Business Results", "🖼️ Wheel Gallery", "🔬 How It Works"])

# ===========================================================================
# LIVE DEMO
# ===========================================================================
with tabs[0]:
    st.markdown("### Upload a photo — the production pipeline classifies it")
    st.markdown("MediaPipe finds the face, K-Means clusters the cheek "
                "pixels, the midtone rule picks the real skin, iris + hair "
                "add two more signals, and the tuned scikit-learn models "
                "(trained on 4,001 Deep Armocromia faces) predict the "
                "season. Every debug artifact is the same file the batch "
                "pipeline saves.")

    c1, c2 = st.columns([1, 1])
    with c1:
        if (SAMPLES / "deep_winter.jpg").is_file() and st.button(
                "Try a sample: DEEP WINTER wheel face"):
            st.session_state["demo"] = (
                "deep_winter.jpg",
                (SAMPLES / "deep_winter.jpg").read_bytes())
        if (SAMPLES / "deep_autumn.jpg").is_file() and st.button(
                "Try a sample: DEEP AUTUMN wheel face"):
            st.session_state["demo"] = (
                "deep_autumn.jpg",
                (SAMPLES / "deep_autumn.jpg").read_bytes())
        upload = st.file_uploader("or upload your own", type=["jpg", "jpeg",
                                                              "png"])
        if upload is not None:
            st.session_state["demo"] = (upload.name, upload.getvalue())

    @st.cache_data
    def cached_run(name, image_bytes):
        return run_pipeline(image_bytes)

    result = None
    if "demo" in st.session_state:
        name, image_bytes = st.session_state["demo"]
        with st.spinner(f"Running the pipeline on {name}…"):
            result = cached_run(name, image_bytes)
        if result.get("error"):
            st.error(result["error"])
            result = None

    if result is not None:
        pred = result["predictions"]
        art = result["artifacts"]
        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.image(art["annotated"], width="stretch",
                     caption="Production pipeline output — cheek boxes, "
                             "swatch inset, pick info")
            with st.expander("Debug artifacts (human-in-the-loop QA)"):
                st.image(art["cheek_crop"], width=320,
                         caption="Cheek K-Means sample crops")
                if Path(art["eyes_crop"]).is_file():
                    st.image(art["eyes_crop"], width=320,
                             caption="Iris annulus samples (pupil excluded)")
                if Path(art["hair_strip"]).is_file():
                    st.image(art["hair_strip"], width=320,
                             caption="Hair strip sample")
                if Path(art["pick_compare"]).is_file():
                    st.image(art["pick_compare"], width=320,
                             caption="Largest vs midtone pick (flash case)")

        with col_res:
            st.markdown("#### Extracted signals")
            rows_html = []
            rows_html.append(
                f'{chip(result["skin_hex"])} **Skin** {result["skin_hex"]} '
                f'· H={result["skin_hsv"][0] * 2}° · share '
                f'{result["share"]:.0%} · pick: {result["pick_method"]}')
            if result["iris_hex"]:
                rows_html.append(f'{chip(result["iris_hex"])} **Iris** '
                                 f'{result["iris_hex"]}')
            if result["hair_hex"]:
                rows_html.append(f'{chip(result["hair_hex"])} **Hair** '
                                 f'{result["hair_hex"]}')
            for h in rows_html:
                st.markdown(h, unsafe_allow_html=True)
            if result["needs_review"]:
                st.warning("No midtone candidate — the largest cluster was "
                           "kept. Treat the result as low-confidence.")
            if result["midtone_v_delta"] and result["midtone_v_delta"] > 0:
                st.caption("The midtone rule recovered a darker cluster: "
                           "the largest cluster was likely flash washout.")

            st.markdown("#### K-Means clusters (cheek)")
            for cl in result["clusters"]:
                st.markdown(
                    f'{chip(cl["hex"], 18)} {cl["hex"]} · share '
                    f'{cl["share"]:.0%} · V={cl["v"]}',
                    unsafe_allow_html=True)

            st.markdown("#### Predictions (tuned models)")
            m = CONFIG["models"]
            for key, title, acc_key in [
                    ("family", "Warm / Cool family", "family"),
                    ("modifier", "Modifier (sub-season axis)", "modifier"),
                    ("season", "12-season", "season"),
                    ("4-season", "4-season", "4-season")]:
                p = pred[key]
                st.markdown(f"**{title}: {p['label']}** "
                            f'<span style="color:{INK2};font-size:13px">'
                            f'(test-set accuracy {m[acc_key]["test_acc"]:.0%} '
                            f'on Deep Armocromia)</span>',
                            unsafe_allow_html=True)
                st.markdown(proba_bars(p["proba"], p["label"]),
                            unsafe_allow_html=True)

            st.caption(f'{result["sample_pixels"]:,} cheek pixels sampled · '
                       f'face {result["face_dims"][0]}×'
                       f'{result["face_dims"][1]}px')

# ===========================================================================
# VALIDATION
# ===========================================================================
with tabs[1]:
    st.markdown("### Deep Armocromia benchmark — out-of-sample accuracy")
    st.markdown("4,920 professionally labeled faces run through the "
                "production pipeline (99.8% face-detection yield). "
                "Hyperparameters tuned by 5-fold inner CV **on the train "
                "set only**; the official test set (910 faces) was scored "
                "once. Same split the dataset authors use.")

    m = model_rows()
    fam = int(m["family"]["test_ok"]) / int(m["family"]["test_n"])
    four = int(m["4-season"]["test_ok"]) / int(m["4-season"]["test_n"])
    twelve = int(m["season"]["test_ok"]) / int(m["season"]["test_n"])
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Warm/cool family", f"{fam:.0%}",
                    "+18 pts vs chance", "Chance baseline 50%. Logreg on "
                    "HSV+LAB signals.")
    with c2:
        metric_card("4-season", f"{four:.0%}", "+29 pts vs chance",
                    "Chance baseline 25%. The dataset authors' deep CNN "
                    "scores ~55% — within a point of ours.")
    with c3:
        metric_card("12-season", f"{twelve:.0%}", "+22 pts vs chance",
                    "Chance baseline 8%. Authors' CNN ~32%; ours 30% — "
                    "the problem is hard for everyone.")

    # Grouped bar: ours vs chance vs published CNN
    labels = ["Warm/cool family", "4-season", "12-season"]
    ours = [fam * 100, four * 100, twelve * 100]
    chance = [50, 25, 100 / 12]
    cnn = [None, 55, 32]
    import plotly.graph_objects as go
    f = go.Figure()
    f.add_trace(go.Bar(x=labels, y=ours, name="Ours (HSV+LAB logreg)",
                       marker_color=BLUE,
                       text=[f"{v:.0f}%" for v in ours],
                       textposition="outside",
                       textfont=dict(color=INK2),
                       hovertemplate="Ours: %{y:.1f}%<extra></extra>"))
    f.add_trace(go.Bar(x=labels, y=chance, name="Chance baseline",
                       marker_color=ORANGE,
                       text=[f"{v:.0f}%" for v in chance],
                       textposition="outside",
                       textfont=dict(color=INK2),
                       hovertemplate="Chance: %{y:.1f}%<extra></extra>"))
    f.add_trace(go.Bar(x=labels, y=cnn, name="Published deep CNN",
                       marker_color=AQUA,
                       text=["", "55%", "32%"], textposition="outside",
                       textfont=dict(color=INK2),
                       hovertemplate="CNN: %{y:.0f}%<extra></extra>"))
    f.update_layout(barmode="group", bargap=0.06, bargroupgap=0.06,
                    yaxis=dict(title="accuracy", range=[0, 78]),
                    title=dict(text="Classifier accuracy vs baselines "
                                    "(official test split)",
                               font=dict(color=INK2, size=14)))
    base_layout(f)
    st.plotly_chart(f)
    st.caption("CNN numbers are the Deep Armocromia authors' published "
               "baselines (no family-level CNN baseline is published, so "
               "that bar is empty).")

    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        st.markdown("#### 12-season confusion matrix (official test)")
        labels_c, z = confusion_matrix()
        f2 = go.Figure(go.Heatmap(
            z=z, x=labels_c, y=labels_c, colorscale=BLUE_RAMP,
            colorbar=dict(title=dict(text="count", font=dict(color=INK2)),
                          tickfont=dict(color=MUTED)),
            hovertemplate="true %{y}<br>predicted %{x}<br>%{z} faces"
                          "<extra></extra>"))
        f2.update_layout(title=dict(text="logreg, HSV+LAB — 276/910 "
                                         "(30%) exact",
                                    font=dict(color=INK2, size=14)),
                         xaxis=dict(tickangle=-40),
                         height=430)
        base_layout(f2, height=430)
        st.plotly_chart(f2)
        with st.expander("Table view (all 144 cells)"):
            st.dataframe([{"true": t, **dict(zip(labels_c, row))}
                          for t, row in zip(labels_c, z)],
                         width="stretch")

    with col_b:
        st.markdown("#### Which features carry the signal?")
        fs_rows = feature_set_rows()
        names = [r[0] for r in fs_rows]
        vals = [r[1] for r in fs_rows]
        colors = [GRAY_FILL, GRAY_FILL, BLUE]
        f3 = go.Figure(go.Bar(
            x=vals, y=names, orientation="h", marker_color=colors,
            text=[f"{v:.1f}%" for v in vals], textposition="outside",
            textfont=dict(color=INK2),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        f3.add_vline(x=50, line_color=GRAY_FILL, line_dash="dash")
        f3.add_annotation(x=50, y=2.45, text="chance (50%)",
                          showarrow=False, font=dict(color=MUTED, size=11),
                          xshift=-38)
        f3.update_layout(xaxis=dict(range=[0, 80], title="family accuracy"),
                         title=dict(text="Warm/cool family accuracy by "
                                         "feature set (official test)",
                                    font=dict(color=INK2, size=14)))
        base_layout(f3, legend=False)
        st.plotly_chart(f3)
        st.markdown("- Skin hue alone already beats chance (+13 pts)")
        st.markdown("- Iris + hair add ~4 pts")
        st.markdown("- HSV **+** LAB is the best configuration — LAB alone "
                    "did not beat HSV (tested 2026-11-05)")
        st.markdown("- The hand-written rules scored 52% family out-of-"
                    "sample — trained models won that comparison too")

# ===========================================================================
# INSTRUMENT LADDER
# ===========================================================================
with tabs[2]:
    st.markdown("### Instrument validation ladder")
    st.markdown("Each step verifies one link of the chain before its "
                "output is trusted downstream — from the color math to the "
                "garment mapping.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Palette self-check", "132/132", "100%",
                    "Every canonical 12-season palette swatch classifies "
                    "as its own season — the palette matcher is exact.")
    with c2:
        metric_card("Synthetic torso recovery", "20/20", "100%",
                    "Painted-torso test images: the garment extractor "
                    "recovers the known painted color every time.")
    with c3:
        metric_card("Title color-word agreement", "74%", "on 300 listings",
                    "Product title says 'red' and the extracted garment "
                    "color is red: 74% of the time.")
    with c4:
        metric_card("Multi-photo consistency", "77%", "same season",
                    "The same product photographed across images gets the "
                    "same palette season 77% of the time.")
    with c5:
        metric_card("DeepFashion2 benchmark", "85% / 57%", "fam / season",
                    "925 images vs human-drawn garment masks: 85% family "
                    "agreement, 57% season agreement (chance ~8%).")

    st.markdown("---")
    st.markdown("**Tested and shelved, with evidence:** per-image white-"
                "balance correction using the sclera (eye white) as a "
                "built-in gray card made the family classifier **3-4 "
                "points worse** on the same 843-image subset — the "
                "lighting confound remains open, and the paper says so.")

# ===========================================================================
# BUSINESS RESULTS
# ===========================================================================
with tabs[3]:
    st.markdown("### Does harmony drive sales? No detectable effect")
    st.markdown("10,000 Amazon dress products → 1,418 judgeable products "
                "with a model face and a classifiable garment. Regression "
                "of log(rating count) on the harmony flag, controlling for "
                "average rating. All three harmony levels are null:")

    import plotly.graph_objects as go
    rows = [
        ("Family (warm/cool)", -3.3, -12.8, 7.3, "n = 1,418", "p = 0.53"),
        ("4-season", 7.3, -5.4, 21.7, "n = 1,418", "p = 0.27"),
        ("12-season", -1.8, -17.3, 16.6, "n = 1,418", "p = 0.84"),
    ]
    ynames = [r[0] for r in rows][::-1]
    xs = [r[1] for r in rows][::-1]
    his = [r[3] - r[1] for r in rows][::-1]
    los = [r[1] - r[2] for r in rows][::-1]
    cols = [RED if x > 0 else BLUE for x in xs]
    texts = [f"{x:+.1f}% (CI {lo:+.1f}…{hi:+.1f})" for x, lo, hi in
             zip(xs, [r[2] for r in rows][::-1], [r[3] for r in rows][::-1])]

    f4 = go.Figure(go.Scatter(
        x=xs, y=ynames, mode="markers", marker=dict(size=11, color=cols),
        error_x=dict(type="data", symmetric=False, array=his,
                     arrayminus=los, color=MUTED, width=2),
        text=texts, textposition="middle right",
        textfont=dict(color=INK2),
        hovertemplate="%{y}<br>effect %{x:+.1f}%<extra></extra>"))
    f4.add_vline(x=0, line_color=GRAY_FILL, line_dash="dash")
    f4.update_layout(
        xaxis=dict(title="effect on rating count (log scale)", range=[-30, 30]),
        title=dict(text="Harmony effect on rating count, 95% CI — every "
                        "interval crosses zero",
                   font=dict(color=INK2, size=14)),
        height=340)
    base_layout(f4, height=340, legend=False)
    st.plotly_chart(f4)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### The +29% hint that did not replicate")
        f5 = go.Figure(go.Scatter(
            x=[29, -1.8], y=[1, 1], mode="lines+markers",
            line=dict(color=MUTED, width=2),
            marker=dict(size=11, color=[MUTED, BLUE])))
        f5.add_annotation(x=29, y=1.12, text="early hint, n = 508, "
                                             "p = 0.074",
                          showarrow=False, font=dict(color=MUTED, size=12))
        f5.add_annotation(x=-1.8, y=1.12, text="final, n = 1,418, "
                                              "p = 0.84",
                          showarrow=False, font=dict(color=INK2, size=12))
        f5.add_vline(x=0, line_color=GRAY_FILL, line_dash="dash")
        f5.update_layout(xaxis=dict(range=[-12, 42], title="12-season "
                                    "harmony effect"),
                         yaxis=dict(visible=False), height=240)
        base_layout(f5, height=240, legend=False)
        st.plotly_chart(f5)
        st.caption("The first batch's dose-response gradient vanished at "
                   "3× scale — small-sample noise. That is why the hint "
                   "was reported as a hint.")

    with c2:
        st.markdown("#### One caveated curiosity")
        st.info("A strict two-photo ensemble flag (both photos must agree) "
                "shows **+147%** on rating count (p = 0.008, Cohen's "
                "d = 1.0) — but **n = 8** harmonious products. Too fragile "
                "to trust; consistent with 'only high-confidence exact-"
                "palette matches might matter.' A lead for a future "
                "study, not a finding.")
        st.markdown("**Why a null is still an answer:**")
        st.markdown("- Classifier error attenuates measured effects: "
                    "family 68% → 4-season 54% → 12-season 30% accuracy "
                    "means the finer the harmony flag, the more signal is "
                    "lost")
        st.markdown("- Camera white-balance and studio lighting shift "
                    "recorded colors — measurement error on both sides of "
                    "the flag")
        st.markdown("- One image per product; no price control at scale; "
                    "single category")

# ===========================================================================
# WHEEL GALLERY
# ===========================================================================
with tabs[4]:
    st.markdown("### The 12-season wheel — where the pipeline started")
    st.markdown("The hand-written v2 rule analyzed the 12 faces of the "
                "canonical season wheel. These numbers are **in-sample** — "
                "thresholds were calibrated on these same faces, and the "
                "later Deep Armocromia test showed the rule scores 52% "
                "out-of-sample. The wheel belongs in the paper as the "
                "exploration chapter, not as validation.")

    # Votes table from output/season-wheel-vs-script-report.md (2026-08-27)
    wheel_rows = [
        ("BRIGHT SPRING", "Warm", 22, "Warm", "Neutral", "Warm", "Warm", "✓"),
        ("DEEP AUTUMN", "Warm", 22, "Warm", "Warm", "Warm", "Warm", "✓"),
        ("LIGHT SPRING", "Warm", 30, "Warm", "Neutral", "Warm", "Warm", "✓"),
        ("SOFT AUTUMN", "Warm", 24, "Warm", "Warm", "Warm", "Warm", "✓"),
        ("WARM AUTUMN", "Warm", 20, "Warm", "Warm", "Warm", "Warm", "✓"),
        ("WARM SPRING", "Warm", 22, "Warm", "Warm", "Warm", "Warm", "✓"),
        ("BRIGHT WINTER", "Cool", 22, "Warm", "Cool", "Warm", "Warm", "✗"),
        ("COOL SUMMER", "Cool", 0, "Cool", "Cool", "Cool", "Cool", "✓"),
        ("COOL WINTER", "Cool", 16, "Cool", "Neutral", "Neutral", "Cool",
         "✓"),
        ("DEEP WINTER", "Cool", 14, "Cool", "Warm", "Warm", "Warm", "✗"),
        ("LIGHT SUMMER", "Cool", 342, "Cool", "Neutral", "Warm", "Neutral",
         "—"),
        ("SOFT SUMMER", "Cool", 6, "Cool", "Neutral", "Neutral", "Cool", "✓"),
    ]
    st.dataframe(
        [{"Season": r[0], "Family": r[1], "Skin H (°)": r[2],
          "Skin vote": r[3], "Iris vote": r[4], "Hair vote": r[5],
          "Combined": r[6], "Match": r[7]} for r in wheel_rows],
        width="stretch", hide_index=True)

    st.markdown("#### Per-face debug artifacts")
    stems = [r[0].lower().replace(" ", "_") for r in wheel_rows]
    face = st.selectbox("Pick a wheel face", [r[0] for r in wheel_rows])
    stem = stems[[r[0] for r in wheel_rows].index(face)]
    j = wheel_json(stem)
    cols = st.columns(3)
    cols[0].image(str(WHEEL / f"{stem}_annotated.jpg"),
                  width="stretch",
                  caption="Annotated — cheek boxes + swatch")
    cols[1].image(str(WHEEL / f"{stem}_cheek_crop.jpg"),
                  width="stretch", caption="Cheek K-Means crops")
    cols[2].image(str(WHEEL / f"{stem}_dominant_swatch.jpg"),
                  width="stretch", caption="Dominant skin color")
    cols2 = st.columns(3)
    cols2[0].image(str(WHEEL / f"{stem}_eyes_crop.jpg"),
                   width="stretch", caption="Iris annuli")
    cols2[1].image(str(WHEEL / f"{stem}_hair_strip.jpg"),
                   width="stretch", caption="Hair strip")
    cols2[2].image(str(WHEEL / f"{stem}_hair_swatch.jpg"),
                   width="stretch", caption="Hair color")
    jhsv = j["hsv_opencv"]
    st.markdown(
        f'{chip(j["dominant_hex"])} skin {j["dominant_hex"]} · H='
        f'{jhsv["h"] * 2}° · share {j["dominant_share"]:.0%} — '
        f'{chip(j.get("iris_hex") or "#ffffff")} iris '
        f'{j.get("iris_hex", "n/a")} — '
        f'{chip(j.get("hair_hex") or "#ffffff")} hair '
        f'{j.get("hair_hex", "n/a")}', unsafe_allow_html=True)
    st.caption("Artifacts produced by the v2 analyzer on 2026-08-27, "
               "before the midtone rule and LAB features were added.")

# ===========================================================================
# HOW IT WORKS
# ===========================================================================
with tabs[5]:
    st.markdown("### The pipeline, step by step")
    steps = [
        ("👤", "Face detection", "MediaPipe Face Landmarker (478 landmarks, "
         "CPU delegate) locates the face and the cheek, iris, and forehead "
         "regions."),
        ("🎯", "Cheek K-Means", "Cheek pixels are clustered into K=3 color "
         "populations (shadow / midtone / highlight) — not averaged."),
        ("⚖️", "Midtone rule", "When studio flash blows out ≥50% of the "
         "cheek, the largest cluster IS the flash. The rule picks the "
         "darkest cluster with ≥15% share inside the plausible-skin "
         "brightness band."),
        ("👁️", "Iris + hair", "Iris color comes from an annulus around "
         "each iris (dark pupil excluded); hair color from the darkest "
         "K-Means cluster of the forehead strip."),
        ("🧠", "Classify", "The 21 HSV+LAB features feed scikit-learn "
         "pipelines (logreg, tuned on 4,001 faces) predicting family, "
         "modifier, 4-season, and 12-season labels."),
    ]
    cols = st.columns(5)
    for col, (icon, title, body) in zip(cols, steps):
        col.markdown(f"**{icon} {title}**")
        col.markdown(f'<span style="color:{INK2};font-size:13px">{body}'
                     f"</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Why K-Means instead of a simple average?")
    st.markdown("A plain average assumes the cheek is one homogeneous "
                "color. It is not — studio flash, shadow, and skin are "
                "three different populations. Averaging them produces a "
                "color that exists in zero pixels. K-Means separates the "
                "populations first, then the rule picks a real one. "
                "Drag the slider:")
    flash_share = st.slider("Share of cheek pixels washed out by studio "
                            "flash", 0, 70, 15, 5) / 100
    import cv2
    from scipy import stats
    base = {"shadow": (80, 10, 0.20), "midtone": (160, 14, 0.55),
            "flash": (243, 6, 0.25)}
    shares = {"shadow": 0.20,
              "midtone": 0.55 * (1 - flash_share),
              "flash": 0.25 + 0.55 * flash_share}
    bins = np.arange(0, 257, 16)
    centers = (bins[:-1] + bins[1:]) / 2
    total = 1000
    band_colors = {"shadow": GRAY_FILL, "midtone": BLUE,
                   "flash": ORANGE}
    f6 = go.Figure()
    for name in ("shadow", "midtone", "flash"):
        mu, sd, _ = base[name]
        counts = (stats.norm.cdf(bins[1:], mu, sd)
                  - stats.norm.cdf(bins[:-1], mu, sd)) * total * shares[name]
        f6.add_trace(go.Bar(x=centers, y=counts, name=f"{name} cluster "
                            f"({shares[name]:.0%})",
                            marker=dict(color=band_colors[name],
                                        line=dict(color=SURFACE,
                                                  width=1.5))))
    mean_v = sum(shares[n] * base[n][0] for n in shares)
    cands = {n: v for n, v in shares.items()
             if v >= 0.15 and 60 <= base[n][0] <= 240}
    pick_v = base[min(cands, key=lambda n: base[n][0])][0] if cands \
        else base["midtone"][0]
    f6.add_vline(x=mean_v, line_color=INK, line_dash="dash", line_width=2)
    f6.add_vline(x=pick_v, line_color=BLUE, line_width=2)
    f6.add_annotation(x=mean_v, y=total * 0.075, text="simple average",
                      showarrow=False, font=dict(color=INK, size=11))
    f6.add_annotation(x=pick_v, y=total * 0.055, text="cluster pick",
                      showarrow=False, font=dict(color=BLUE, size=11))
    f6.update_layout(barmode="stack", xaxis=dict(title="brightness (HSV V)"),
                     yaxis=dict(title="cheek pixels"),
                     title=dict(text="Cheek pixel populations and the two "
                                     "answers",
                                font=dict(color=INK2, size=14)))
    base_layout(f6, height=360)
    st.plotly_chart(f6)

    def v_to_hex(v):
        rgb = cv2.cvtColor(np.uint8([[[10, 115, int(v)]]]),
                           cv2.COLOR_HSV2RGB)[0, 0]
        return "#{:02X}{:02X}{:02X}".format(*rgb)

    avg_hex, pick_hex = v_to_hex(mean_v), v_to_hex(pick_v)
    c1, c2 = st.columns(2)
    c1.markdown(f'{chip(avg_hex, 40)} **Simple average → {avg_hex}** '
                f'<span style="color:{INK2}">(washed-out blend of all '
                f'three populations)</span>', unsafe_allow_html=True)
    c2.markdown(f'{chip(pick_hex, 40)} **K-Means + midtone rule → '
                f'{pick_hex}** <span style="color:{INK2}">(a real skin '
                f'cluster, flash ignored)</span>', unsafe_allow_html=True)
    st.caption(f"With {flash_share:.0%} flash washout the average lands at "
               f"brightness V={mean_v:.0f}; the cluster pick stays at "
               f"V={pick_v:.0f}. At ≥80% washout even the largest cluster "
               f"is the flash — which is exactly the bug the midtone rule "
               f"was built to fix.")

    st.markdown("---")
    st.markdown("#### The models behind the predictions")
    mcols = st.columns(4)
    names = {"family": "Warm/cool family", "modifier": "Modifier",
             "season": "12-season", "4-season": "4-season"}
    for col, key in zip(mcols, names):
        info = CONFIG["models"][key]
        col.markdown(f"**{names[key]}**")
        col.markdown(f'{info["test_ok"]}/{info["test_n"]} test faces '
                     f'({info["test_acc"]:.0%}, CI '
                     f'{info["ci_lo"]:.0%}–{info["ci_hi"]:.0%})')
        col.markdown(f'logreg C={info["best_C"]} · 21 features · '
                     f'5-fold inner CV')
    st.caption("Models exported by src/export_models.py — same protocol "
               "as the sklearn-evaluation report; numbers reproduced "
               "exactly (616/910 family MATCH).")
