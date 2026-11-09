"""Backend service for the dashboard's live demo.

Runs the EXACT production pipeline (skin_color_analyzer functions: MediaPipe
face detection -> cheek K-Means -> midtone pick -> iris annulus -> hair
strip) on an uploaded photo, then classifies it with the exported Deep
Armocromia models (the same pipelines whose test-set numbers appear in the
Validation tab).

No Streamlit imports here, so this module is testable from the CLI:
    python service.py path/to/photo.jpg
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_color_analyzer import (  # noqa: E402
    DEFAULT_MODEL, crop_cheeks, create_landmarker, detect_face,
    dominant_color_kmeans, hair_strip, iris_rings, midtone_pick,
    rgb_to_hex, rgb_to_hsv_opencv, save_annotated, save_cheek_crop,
    save_eyes_crop, save_hair_strip, save_pick_compare, save_swatch)
from loocv_validation import extract_features  # noqa: E402
from lab_feature_test import lab_features_of  # noqa: E402

MODEL_DIR = ROOT / "models"
CONFIG = json.loads((MODEL_DIR / "feature_config.json").read_text())

_landmarker = None
_pipes = None


def get_landmarker():
    """MediaPipe Face Landmarker (CPU delegate), created once per process."""
    global _landmarker
    if _landmarker is None:
        _landmarker = create_landmarker(DEFAULT_MODEL)
    return _landmarker


def get_models():
    """The exported sklearn pipelines, loaded once per process."""
    global _pipes
    if _pipes is None:
        _pipes = {name: joblib.load(MODEL_DIR / f"{name}_pipeline.joblib")
                  for name in CONFIG["models"]}
    return _pipes


def predict_all(X_row):
    """One feature row (21 cols) -> per-target label + class probabilities."""
    out = {}
    for name, pipe in get_models().items():
        proba = pipe.predict_proba(X_row)[0]
        classes = [str(c) for c in pipe.classes_]
        ranked = sorted(zip(classes, proba), key=lambda t: -t[1])
        out[name] = {
            "label": str(pipe.predict(X_row)[0]),
            "proba": {c: round(float(p), 4) for c, p in ranked},
        }
    return out


def run_pipeline(image_bytes):
    """Analyze an uploaded photo and classify it. Returns a dict with
    extraction results, debug-artifact paths, and model predictions, or
    {"error": reason}."""
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error": "Could not decode the image — upload a JPG or PNG."}
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    landmarks, problem = detect_face(image_rgb, get_landmarker())
    if landmarks is None:
        return {"error": f"{problem} — try a clearer, frontal, well-lit "
                         f"photo."}

    # ---- skin: cheek K-Means + midtone rule (production pipeline) ----
    crops, boxes = crop_cheeks(image_rgb, landmarks)
    samples = np.vstack([c.reshape(-1, 3) for c in crops])
    largest, largest_share, rows = dominant_color_kmeans(samples, k=3)
    mid = midtone_pick(rows)
    if mid is not None:
        dominant, share, pick_method = mid["rgb"], mid["share"], "midtone"
    else:
        dominant, share, pick_method = largest, largest_share, \
            "fallback_largest"
    hsv = rgb_to_hsv_opencv(dominant)
    v_delta = rgb_to_hsv_opencv(largest)[2] - hsv[2] if mid is not None else 0

    # ---- eyes: iris annuli ----
    rings = iris_rings(image_rgb, landmarks)
    iris_rgb = iris_hsv = None
    if rings:
        iris_samples = np.vstack([p for p, *_ in rings])
        iris_rgb, _, _ = dominant_color_kmeans(iris_samples, k=2)
        iris_hsv = rgb_to_hsv_opencv(iris_rgb)

    # ---- hair: strip above the forehead ----
    hair_rgb = hair_hsv = None
    strip = hair_strip(image_rgb, landmarks)
    if strip is not None:
        hair_samples, _ = strip
        hair_rgb, _, _ = dominant_color_kmeans(hair_samples, k=3,
                                               pick="darkest")
        hair_hsv = rgb_to_hsv_opencv(hair_rgb)

    # ---- same raw dict + feature math the batch extraction used ----
    d = {
        "hsv_opencv": {"h": hsv[0], "s": hsv[1], "v": hsv[2]},
        "dominant_rgb": list(dominant),
        "dominant_hex": rgb_to_hex(dominant),
        "iris_hsv_opencv": ({"h": iris_hsv[0], "s": iris_hsv[1],
                             "v": iris_hsv[2]} if iris_hsv else None),
        "hair_hsv_opencv": ({"h": hair_hsv[0], "s": hair_hsv[1],
                             "v": hair_hsv[2]} if hair_hsv else None),
    }
    f = extract_features(d)
    row = {**f, "dominant_hex": d["dominant_hex"],
           "has_iris": f["has_iris"], "has_hair": f["has_hair"]}
    feats = lab_features_of(row)
    feats["contrast"] = f["contrast"]  # part of the HSV block, not LAB
    X_row = np.array([[feats[c] for c in CONFIG["columns"]]])
    predictions = predict_all(X_row)

    # ---- production debug artifacts (the same files the CLI saves) ----
    tmp = Path(tempfile.mkdtemp(prefix="harmony_demo_"))
    artifacts = {
        "cheek_crop": tmp / "cheek_crop.jpg",
        "swatch": tmp / "dominant_swatch.jpg",
        "pick_compare": tmp / "pick_compare.jpg",
        "annotated": tmp / "annotated.jpg",
        "eyes_crop": tmp / "eyes_crop.jpg",
        "iris_swatch": tmp / "iris_swatch.jpg",
        "hair_strip": tmp / "hair_strip.jpg",
        "hair_swatch": tmp / "hair_swatch.jpg",
    }
    save_cheek_crop(crops, artifacts["cheek_crop"])
    save_swatch(dominant, artifacts["swatch"])
    pick_info = (f"Pick: {pick_method} (dV={v_delta})" if mid is not None
                 else "Pick: fallback_largest (no midtone candidate)")
    if mid is not None:
        save_pick_compare(crops, largest, dominant, pick_method,
                          artifacts["pick_compare"])
    save_annotated(image_rgb, boxes, dominant, hsv, share,
                   artifacts["annotated"], pick_info=pick_info)
    if rings:
        save_eyes_crop(image_rgb, rings, artifacts["eyes_crop"])
        save_swatch(iris_rgb, artifacts["iris_swatch"])
    if hair_rgb is not None:
        save_hair_strip(image_rgb, strip[1], artifacts["hair_strip"])
        save_swatch(hair_rgb, artifacts["hair_swatch"])

    return {
        "error": None,
        "face_dims": (image_rgb.shape[1], image_rgb.shape[0]),
        "sample_pixels": int(samples.shape[0]),
        "skin_hex": rgb_to_hex(dominant),
        "skin_hsv": hsv,
        "share": round(share, 4),
        "pick_method": pick_method,
        "needs_review": pick_method == "fallback_largest",
        "largest_hex": rgb_to_hex(largest),
        "largest_share": round(largest_share, 4),
        "midtone_v_delta": v_delta,
        "clusters": [{"hex": rgb_to_hex(r["rgb"]),
                      "count": r["count"], "share": r["share"],
                      "v": r["hsv"][2]} for r in rows],
        "iris_hex": rgb_to_hex(iris_rgb) if iris_rgb else None,
        "hair_hex": rgb_to_hex(hair_rgb) if hair_rgb else None,
        "predictions": predictions,
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }


if __name__ == "__main__":
    # CLI smoke test: python service.py path/to/photo.jpg
    img_path = Path(sys.argv[1])
    result = run_pipeline(img_path.read_bytes())
    if result.get("error"):
        print(f"ERROR: {result['error']}")
        sys.exit(1)
    for k, v in result.items():
        if k not in ("clusters", "artifacts"):
            print(f"{k}: {v}")
    print("predictions:")
    for name, p in result["predictions"].items():
        top = ", ".join(f"{c} {w:.0%}" for c, w in list(p["proba"].items())[:3])
        print(f"  {name:9s} -> {p['label']}  ({top})")
