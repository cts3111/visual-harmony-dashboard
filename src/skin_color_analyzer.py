#!/usr/bin/env python3
"""
Skin Color Analyzer — Script 1 of the E-commerce Visual Harmony & ROI Analyzer.

Pipeline:
    1. Load a local image (OpenCV) and convert BGR -> RGB.
    2. Detect the face with MediaPipe Face Landmarker (Tasks API, 478
       landmarks) using the bundled face_landmarker.task model.
    3. Crop square cheek samples around the cheek landmarks, scaled to eye
       width so the sample size tracks the face, not the image dimensions.
    4. Cluster the cheek pixels with K-Means and pick the dominant skin
       color with the midtone rule: the darkest cluster that is a real
       population (share >= MIDTONE_SHARE_MIN) inside the plausible-skin
       brightness band (V MIDTONE_V_LO..MIDTONE_V_HI). Studio flash can
       blow out >=50% of the cheek pixels, which makes the largest cluster
       the flash itself, not skin; the midtone rule recovers the true
       skin. When no candidate qualifies, the pick falls back to the
       largest cluster and the results JSON sets needs_review (README).
    5. Convert the dominant color from RGB to HSV (OpenCV ranges).
    6. Also extract iris color (annulus around the iris landmarks, so the
       dark pupil stays out of the sample) and hair color (strip above the
       forehead, darkest K-Means cluster — hair is the darkest element in
       the head region).
    7. Save visual debug artifacts for human-in-the-loop QA:
         {stem}_cheek_crop.jpg       the two cheek samples, side by side
         {stem}_dominant_swatch.jpg  a solid block of the dominant skin color
         {stem}_pick_compare.jpg     largest vs midtone swatches (only when
                                     the midtone rule changed the pick)
         {stem}_annotated.jpg        original + cheek boxes + swatch + values
         {stem}_eyes_crop.jpg        iris regions with the sampled rings drawn
         {stem}_iris_swatch.jpg      solid block of the dominant iris color
         {stem}_hair_strip.jpg       the hair strip that was sampled
         {stem}_hair_swatch.jpg      solid block of the dominant hair color
         {stem}_results.json         numeric results (RGB, HSV, clusters)

Usage:
    python3 skin_color_analyzer.py path/to/image.jpg [--k 3] [--out-dir debug]

Requires the MediaPipe model file at ../models/face_landmarker.task
(already downloaded; override with --model if you move it).

Exit codes:
    0  success
    1  usage or file error
    2  no face detected in the image
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Tunable constants — deterministic pipeline, documented per project README.
# ---------------------------------------------------------------------------
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
CHEEK_LANDMARKS = (205, 425)   # Face Landmarker cheek points (same topology
EYE_CORNERS = (33, 263)        # as legacy Face Mesh; iris pts 468-477 unused)
CROP_SCALE = 0.38              # Cheek square side as a fraction of eye width
MIDLINE_BIAS = 0.15            # Nudge each crop toward the face midline, as a
                               # fraction of the crop side (keeps samples off
                               # hair/background edges)
DEFAULT_K = 3                  # K-Means clusters: shadow / midtone / highlight
KMEANS_SEED = 42               # Reseeds the RNG before every cv2.kmeans call
                               # (KMeans++ seeding is randomized): the same
                               # image yields identical clusters whether run
                               # alone or at any position inside a batch loop
MIDTONE_SHARE_MIN = 0.15       # Midtone pick: a candidate needs a real
                               # population share, not K-Means noise
MIDTONE_V_LO, MIDTONE_V_HI = 60, 240  # Midtone pick: plausible-skin
                               # brightness band (HSV V 0-255) — below is
                               # shadow, above is blown-out flash
SWATCH_SIZE = 400              # Swatch image edge length in pixels
CROP_LABELS = {205: "cheek L", 425: "cheek R"}

# Eye color: iris landmarks 468-477. The 5 points outline each iris; the
# sample ring skips the dark pupil in the middle and the sclera outside.
IRIS_GROUPS = {"iris R": (468, 469, 470, 471, 472),
               "iris L": (473, 474, 475, 476, 477)}
IRIS_RADIUS_SCALE = 1.25       # landmark points under-report the iris edge
IRIS_RING_IN = 0.35            # annulus inner edge, fraction of iris radius
IRIS_RING_OUT = 0.95           # annulus outer edge, fraction of iris radius

# Hair color: strip above the forehead (face top = highest landmark).
# The strip is sized relative to the face so it tracks small/large faces.
HAIR_STRIP_ABOVE = 0.38        # strip height as a fraction of face height
HAIR_STRIP_SIDE = 0.18         # horizontal padding as a fraction of face width


def load_image_rgb(path):
    """Load an image with OpenCV (BGR) and return it as RGB uint8."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def create_landmarker(model_path):
    """Create a MediaPipe FaceLandmarker (Tasks API) for still images.

    Delegate is pinned to CPU: the Metal/GPU delegate crashes on macOS
    ("DrishtiMetalHelper ... Service is unavailable"), and CPU is
    deterministic for a batch pipeline anyway.
    """
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_face(image_rgb, landmarker):
    """
    Run MediaPipe Face Landmarker on an RGB image.

    Returns (landmarks, None) where landmarks is an (N, 2) int array in
    pixel coordinates, or (None, problem) when no face was found.
    """
    h, w = image_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, "No face detected"
    points = result.face_landmarks[0]  # first (and only) face, normalized coords
    landmarks = np.array([[int(p.x * w), int(p.y * h)] for p in points], dtype=int)
    return landmarks, None


def crop_cheeks(image_rgb, landmarks):
    """
    Crop a square sample around each CHEEK_LANDMARK.

    The square side is CROP_SCALE times the distance between the outer eye
    corners (so the sample scales with the face, not the image size), and
    each square is nudged MIDLINE_BIAS of its side toward the midpoint
    between the cheeks — that keeps the sample on the cheek apple instead of
    catching hair or background at the face edge.
    Returns (crops, boxes) where boxes are (x0, y0, side) tuples.
    """
    h, w = image_rgb.shape[:2]
    eye_width = float(
        np.linalg.norm(landmarks[EYE_CORNERS[0]] - landmarks[EYE_CORNERS[1]])
    )
    side = max(int(eye_width * CROP_SCALE), 8)
    side = min(side, w, h)  # never larger than the image itself

    cheek_pts = [landmarks[i] for i in CHEEK_LANDMARKS]
    midline = (cheek_pts[0] + cheek_pts[1]) / 2.0

    crops, boxes = [], []
    for pt in cheek_pts:
        to_mid = midline - pt
        length = float(np.linalg.norm(to_mid))
        center = pt if length == 0 else pt + to_mid / length * (MIDLINE_BIAS * side)
        x0 = int(np.clip(round(center[0] - side // 2), 0, w - side))
        y0 = int(np.clip(round(center[1] - side // 2), 0, h - side))
        boxes.append((x0, y0, side))
        crops.append(image_rgb[y0:y0 + side, x0:x0 + side])
    return crops, boxes


def iris_rings(image_rgb, landmarks):
    """
    Sample the iris color as an annulus around each iris landmark group.

    The ring (IRIS_RING_IN..IRIS_RING_OUT of the iris radius) skips the dark
    pupil in the middle and any sclera/specular highlights at the edge.
    Returns a list of (pixels, box, center, radius); empty if the landmarker
    did not provide iris landmarks (landmarks < 478 entries).
    """
    if landmarks.shape[0] < 478:
        return []
    h, w = image_rgb.shape[:2]
    rings = []
    for idxs in IRIS_GROUPS.values():
        pts = landmarks[list(idxs)]
        center = pts.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(pts - center, axis=1))) \
            * IRIS_RADIUS_SCALE
        r = max(round(radius), 4)
        x0, y0 = int(np.clip(round(center[0] - r), 0, w - 1)), \
                 int(np.clip(round(center[1] - r), 0, h - 1))
        x1, y1 = int(np.clip(round(center[0] + r), 0, w - 1)), \
                 int(np.clip(round(center[1] + r), 0, h - 1))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        crop = image_rgb[y0:y1 + 1, x0:x1 + 1]
        yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
        dist = np.sqrt((xx - (center[0] - x0)) ** 2 + (yy - (center[1] - y0)) ** 2)
        mask = (dist >= IRIS_RING_IN * radius) & (dist <= IRIS_RING_OUT * radius)
        pixels = crop[mask]
        if len(pixels) >= 8:
            rings.append((pixels, (x0, y0, x1, y1), center, radius))
    return rings


def hair_strip(image_rgb, landmarks):
    """
    Sample a strip of pixels above the forehead for hair color.

    The strip spans the face width plus a small margin on each side (hair
    frames the forehead there), and extends HAIR_STRIP_ABOVE of the face
    height upward from the highest landmark.
    Returns (pixels, (x0, y0, x1, y1)) or None when the strip would be empty.
    """
    h, w = image_rgb.shape[:2]
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    left, right, top = xs.min(), xs.max(), ys.min()
    face_w = max(right - left, 1)
    face_h = max(ys.max() - ys.min(), 1)
    x0 = int(np.clip(left - HAIR_STRIP_SIDE * face_w, 0, w - 1))
    x1 = int(np.clip(right + HAIR_STRIP_SIDE * face_w, 0, w - 1))
    y1 = int(np.clip(top, 1, h))
    y0 = int(np.clip(top - HAIR_STRIP_ABOVE * face_h, 0, y1 - 1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return image_rgb[y0:y1, x0:x1], (x0, y0, x1, y1)


def midtone_pick(rows):
    """Darkest cluster among plausible-skin candidates, or None.

    When studio flash blows out >=50% of the cheek pixels, the largest
    K-Means cluster IS the flash — the true skin is the darkest significant
    cluster that is not shadow-black. Candidates need share >=
    MIDTONE_SHARE_MIN (a real population, not K-Means noise) and V within
    MIDTONE_V_LO..MIDTONE_V_HI. Returns None when no candidate qualifies,
    so the caller keeps the largest-cluster pick and flags the image
    needs_review. (V1 picked the median-V candidate; that mis-picked the
    washed cluster when only two candidates existed — see project README.)
    """
    cands = [r for r in rows
             if r["share"] >= MIDTONE_SHARE_MIN
             and MIDTONE_V_LO <= r["hsv"][2] <= MIDTONE_V_HI]
    if not cands:
        return None
    cands.sort(key=lambda r: r["hsv"][2])
    return cands[0]


def dominant_color_kmeans(samples, k=DEFAULT_K, pick="largest"):
    """
    K-Means over an (N, 3) float32 RGB pixel array.

    pick="largest" returns the biggest cluster (skin, iris); pick="darkest"
    returns the cluster with the lowest V (hair — the darkest element in the
    head region). Returns (color_rgb, share, rows) where rows is a list of
    dicts sorted by population, each {"rgb": [...], "hsv": [...], ...}.

    The RNG is reseeded to KMEANS_SEED before each call so KMeans++
    seeding is reproducible regardless of batch position (see constant).
    """
    pixels = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    if pixels.shape[0] == 0:
        raise ValueError("Empty pixel sample")
    k = min(k, len(np.unique(pixels, axis=0)))  # k cannot exceed unique colors

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(KMEANS_SEED)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)
    rows = []
    for i in range(k):
        rgb = [int(v) for v in np.round(centers[i])]
        hsv = rgb_to_hsv_opencv(rgb)
        rows.append({
            "rgb": rgb,
            "hsv": hsv,
            "count": int(counts[i]),
            "share": round(float(counts[i]) / len(pixels), 4),
        })
    rows.sort(key=lambda r: r["count"], reverse=True)
    if pick == "darkest":
        chosen = min(rows, key=lambda r: r["hsv"][2])
    else:
        chosen = rows[0]
    return chosen["rgb"], chosen["share"], rows


def rgb_to_hsv_opencv(rgb):
    """Convert an RGB [r, g, b] triplet to OpenCV HSV (H 0-179, S/V 0-255)."""
    h, s, v = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
    return int(h), int(s), int(v)


def rgb_to_lab_opencv(rgb):
    """Convert an RGB [r, g, b] triplet to OpenCV LAB (L 0-255, a/b centered
    at 128). Stored for auditability and as trained-model features: the Deep
    Armocromia test showed HSV+LAB combined is the best warm/cool feature
    set, while LAB alone does not beat HSV (see project README)."""
    return [int(v) for v in
            cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0]]


def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def save_cheek_crop(crops, path):
    """Stack both cheek crops side by side with labels and save as JPG."""
    panels = []
    for crop, idx in zip(crops, CHEEK_LANDMARKS):
        panel = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1),
                      (0, 255, 0), 2)
        cv2.putText(panel, f"{CROP_LABELS[idx]} (LM {idx})", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(panel)
    cv2.imwrite(str(path), np.hstack(panels))


def save_swatch(rgb, path, size=SWATCH_SIZE):
    """Save a solid block of the dominant color."""
    bgr = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2BGR)[0, 0]
    swatch = np.full((size, size, 3), bgr, dtype=np.uint8)
    cv2.imwrite(str(path), swatch)


def save_pick_compare(crops, largest_rgb, chosen_rgb, pick_method, path):
    """Cheek crops above, largest-vs-midtone swatches below, labels included.

    Written only when the midtone rule changed the pick, so the human in
    the loop can verify the darker pick is real skin, not shadow.
    """
    panels = [cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for c in crops]
    crops_img = np.hstack(panels)
    sw = crops_img.shape[1] // 2
    swatch = np.full((60, sw * 2, 3), 255, dtype=np.uint8)
    for i, (rgb, label) in enumerate([
            (largest_rgb, f"largest {rgb_to_hex(largest_rgb)}"),
            (chosen_rgb, f"{pick_method} {rgb_to_hex(chosen_rgb)}")]):
        bgr = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2BGR)[0, 0]
        swatch[10:50, i * sw:(i + 1) * sw] = bgr
        cv2.putText(swatch, label, (i * sw + 6, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([crops_img, swatch]))


def save_eyes_crop(image_rgb, rings, path):
    """Stack the iris boxes with the sampled rings drawn on top."""
    panels = []
    for pixels, (x0, y0, x1, y1), center, radius in rings:
        panel = cv2.cvtColor(image_rgb[y0:y1 + 1, x0:x1 + 1],
                             cv2.COLOR_RGB2BGR).copy()
        c = (round(center[0] - x0), round(center[1] - y0))
        cv2.circle(panel, c, round(IRIS_RING_OUT * radius), (0, 255, 0), 1)
        cv2.circle(panel, c, round(IRIS_RING_IN * radius), (0, 255, 0), 1)
        cv2.putText(panel, "iris ring", (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 255, 0), 1, cv2.LINE_AA)
        # Resize to a common panel size — iris boxes differ per eye, and
        # hstack needs matching heights.
        panels.append(cv2.resize(panel, (96, 96),
                                 interpolation=cv2.INTER_CUBIC))
    cv2.imwrite(str(path), np.hstack(panels))


def save_hair_strip(image_rgb, box, path):
    """Save the sampled hair strip with a green border."""
    (x0, y0, x1, y1) = box
    panel = cv2.cvtColor(image_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR).copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1),
                  (0, 255, 0), 2)
    cv2.putText(panel, "hair strip", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), panel)


def save_annotated(image_rgb, boxes, dominant_rgb, hsv, share, path,
                   pick_info=None):
    """Original image with cheek boxes, an inset swatch, and the results."""
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    for (x0, y0, side) in boxes:
        cv2.rectangle(canvas, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)

    # Swatch inset in the top-right corner.
    inset = int(min(canvas.shape[:2]) * 0.18)
    bgr = [int(v) for v in cv2.cvtColor(np.uint8([[dominant_rgb]]),
                                        cv2.COLOR_RGB2BGR)[0, 0]]
    cv2.rectangle(canvas, (canvas.shape[1] - inset, 0),
                  (canvas.shape[1], inset), bgr, -1)
    cv2.rectangle(canvas, (canvas.shape[1] - inset, 0),
                  (canvas.shape[1], inset), (255, 255, 255), 2)

    # Results text in the top-left corner.
    lines = [
        f"RGB {tuple(dominant_rgb)}  {rgb_to_hex(dominant_rgb)}",
        f"HSV H={hsv[0]} S={hsv[1]} V={hsv[2]} (OpenCV ranges)",
        f"Dominant cluster share: {share:.1%}",
    ]
    if pick_info:
        lines.append(pick_info)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (6, 18 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Detect a face, K-Means the cheek pixels, and report the "
                    "dominant skin color in RGB + HSV, plus iris and hair color."
    )
    parser.add_argument("image", type=Path,
                        help="Path to a local image of a human model (.jpg etc.)")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"Number of K-Means clusters (default {DEFAULT_K})")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to the MediaPipe face_landmarker.task model "
                             f"(default: {DEFAULT_MODEL})")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "debug",
                        help="Directory for debug artifacts "
                             "(default: ../debug next to the src/ folder)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        image_rgb = load_image_rgb(args.image)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        print("Download it with:\n"
              "  curl -sL -o ../models/face_landmarker.task "
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task", file=sys.stderr)
        return 1

    with create_landmarker(args.model) as landmarker:
        landmarks, problem = detect_face(image_rgb, landmarker)
    if landmarks is None:
        print(f"ERROR: {problem} in {args.image}", file=sys.stderr)
        print("No artifacts written. Try a clearer, frontal photo.", file=sys.stderr)
        return 2

    # ---- skin: cheeks ----
    crops, boxes = crop_cheeks(image_rgb, landmarks)
    samples = np.vstack([c.reshape(-1, 3) for c in crops])
    largest, largest_share, rows = dominant_color_kmeans(samples, k=args.k)
    mid = midtone_pick(rows)
    if mid is not None:
        dominant, share, pick_method = mid["rgb"], mid["share"], "midtone"
    else:
        dominant, share, pick_method = largest, largest_share, \
            "fallback_largest"
    hsv = rgb_to_hsv_opencv(dominant)
    lab = rgb_to_lab_opencv(dominant)
    v_delta = rgb_to_hsv_opencv(largest)[2] - hsv[2] if mid is not None else 0
    pick_info = (f"Pick: {pick_method} (dV={v_delta})" if mid is not None
                 else "Pick: fallback_largest (no midtone candidate)")

    # ---- eyes: iris annuli ----
    rings = iris_rings(image_rgb, landmarks)
    iris_rgb = iris_share = iris_rows = None
    if rings:
        iris_samples = np.vstack([p for p, _, _, _ in rings])
        iris_rgb, iris_share, iris_rows = dominant_color_kmeans(iris_samples, k=2)

    # ---- hair: strip above the forehead ----
    hair_rgb = hair_share = hair_rows = hair_box = None
    strip = hair_strip(image_rgb, landmarks)
    if strip is not None:
        hair_samples, hair_box = strip
        hair_rgb, hair_share, hair_rows = dominant_color_kmeans(
            hair_samples, k=3, pick="darkest")

    iris_lab = rgb_to_lab_opencv(iris_rgb) if iris_rgb else None
    hair_lab = rgb_to_lab_opencv(hair_rgb) if hair_rgb else None

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    paths = {
        "cheek_crop": out_dir / f"{stem}_cheek_crop.jpg",
        "swatch": out_dir / f"{stem}_dominant_swatch.jpg",
        "pick_compare": out_dir / f"{stem}_pick_compare.jpg",
        "annotated": out_dir / f"{stem}_annotated.jpg",
        "eyes_crop": out_dir / f"{stem}_eyes_crop.jpg",
        "iris_swatch": out_dir / f"{stem}_iris_swatch.jpg",
        "hair_strip": out_dir / f"{stem}_hair_strip.jpg",
        "hair_swatch": out_dir / f"{stem}_hair_swatch.jpg",
        "json": out_dir / f"{stem}_results.json",
    }
    save_cheek_crop(crops, paths["cheek_crop"])
    save_swatch(dominant, paths["swatch"])
    if mid is not None:
        save_pick_compare(crops, largest, dominant, pick_method,
                          paths["pick_compare"])
    save_annotated(image_rgb, boxes, dominant, hsv, share, paths["annotated"],
                   pick_info=pick_info)
    if rings:
        save_eyes_crop(image_rgb, rings, paths["eyes_crop"])
        save_swatch(iris_rgb, paths["iris_swatch"])
    if hair_rgb is not None:
        save_hair_strip(image_rgb, hair_box, paths["hair_strip"])
        save_swatch(hair_rgb, paths["hair_swatch"])

    results = {
        "image": str(args.image),
        "cheek_boxes": [list(b) for b in boxes],
        "kmeans_k": args.k,
        "sample_pixels": int(samples.shape[0]),
        "dominant_rgb": [int(v) for v in dominant],
        "dominant_hex": rgb_to_hex(dominant),
        "dominant_share": round(share, 4),
        "hsv_opencv": {"h": hsv[0], "s": hsv[1], "v": hsv[2]},
        "lab_opencv": {"l": lab[0], "a": lab[1], "b": lab[2]},
        "pick_method": pick_method,
        "largest_rgb": [int(v) for v in largest],
        "largest_hex": rgb_to_hex(largest),
        "largest_share": round(largest_share, 4),
        "midtone_v_delta": v_delta,
        "needs_review": pick_method == "fallback_largest",
        "clusters": rows,
        "iris_rgb": [int(v) for v in iris_rgb] if iris_rgb else None,
        "iris_hex": rgb_to_hex(iris_rgb) if iris_rgb else None,
        "iris_share": round(iris_share, 4) if iris_share else None,
        "iris_lab_opencv": {"l": iris_lab[0], "a": iris_lab[1],
                            "b": iris_lab[2]} if iris_rgb else None,
        "iris_hsv_opencv": {"h": rgb_to_hsv_opencv(iris_rgb)[0],
                            "s": rgb_to_hsv_opencv(iris_rgb)[1],
                            "v": rgb_to_hsv_opencv(iris_rgb)[2]} if iris_rgb else None,
        "iris_clusters": iris_rows,
        "hair_rgb": [int(v) for v in hair_rgb] if hair_rgb else None,
        "hair_hex": rgb_to_hex(hair_rgb) if hair_rgb else None,
        "hair_share": round(hair_share, 4) if hair_share else None,
        "hair_lab_opencv": {"l": hair_lab[0], "a": hair_lab[1],
                            "b": hair_lab[2]} if hair_rgb else None,
        "hair_hsv_opencv": {"h": rgb_to_hsv_opencv(hair_rgb)[0],
                            "s": rgb_to_hsv_opencv(hair_rgb)[1],
                            "v": rgb_to_hsv_opencv(hair_rgb)[2]} if hair_rgb else None,
        "hair_clusters": hair_rows,
    }
    with open(paths["json"], "w") as fh:
        json.dump(results, fh, indent=2)

    # Console summary for the human in the loop.
    print(f"image:        {args.image} ({image_rgb.shape[1]}x{image_rgb.shape[0]})")
    print(f"cheek boxes:  {boxes} ({samples.shape[0]:,} pixels sampled)")
    print(f"dominant RGB: {tuple(dominant)}  hex {rgb_to_hex(dominant)}  share {share:.1%}")
    print(f"pick:         {pick_method}"
          + (f"  dV={v_delta}" if mid is not None
             else "  (no midtone candidate; kept largest)"))
    print(f"HSV (OpenCV): H={hsv[0]}  S={hsv[1]}  V={hsv[2]}")
    print(f"HSV (human):  H={hsv[0] * 2:.0f} deg  S={hsv[1] / 255:.1%}  V={hsv[2] / 255:.1%}")
    print("clusters (largest first):")
    for row in rows:
        print(f"  {row['rgb']}  count={row['count']:>7,}  share={row['share']:.1%}")
    if iris_rgb:
        ih, is_, iv = rgb_to_hsv_opencv(iris_rgb)
        print(f"iris RGB:     {tuple(iris_rgb)}  hex {rgb_to_hex(iris_rgb)}  "
              f"share {iris_share:.1%}  (H={ih * 2:.0f} deg S={is_ / 255:.1%} "
              f"V={iv / 255:.1%})")
    else:
        print("iris RGB:     not available (no iris landmarks detected)")
    if hair_rgb:
        hh, hs, hv = rgb_to_hsv_opencv(hair_rgb)
        print(f"hair RGB:     {tuple(hair_rgb)}  hex {rgb_to_hex(hair_rgb)}  "
              f"share {hair_share:.1%}  (H={hh * 2:.0f} deg S={hs / 255:.1%} "
              f"V={hv / 255:.1%})")
    else:
        print("hair RGB:     not available (no hair visible above the forehead)")
    for name, p in paths.items():
        if p.exists():
            print(f"saved:        {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
