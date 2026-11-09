# Visual Harmony & ROI Analyzer — Dashboard

Streamlit app with the project's results and a live demo that runs the
production color-analysis pipeline on an uploaded photo. This is a
standalone project — everything it needs is in this folder (~8 MB).

## Run locally

```bash
# from this folder, using the workspace venv
python -m streamlit run app.py
# (venv lives at ../harmony-analyzer/.venv — e.g.
#  ../harmony-analyzer/.venv/bin/python -m streamlit run app.py)
```

## Deploy to Streamlit Community Cloud (free public link)

1. Push this folder as a GitHub repo (it is git-initialized and
   self-contained).
2. On share.streamlit.io → New app → pick the repo, main file path
   `app.py`.
3. Cloud installs `requirements.txt` automatically; `runtime.txt` pins
   Python 3.11 (mediapipe needs it).
4. Copy the public URL.

## Putting it on your website

- **Today:** link it from your site nav ("Live demo") or embed with an
  iframe — Streamlit apps embed fine.
- **Later upgrade path:** for a native in-page embed, rewrite the UI as
  FastAPI + static frontend. `service.py` is the entire backend logic and
  would be reused unchanged.

## Structure

- `app.py` — Streamlit UI: 6 tabs (Live Demo, Validation, Instrument
  Ladder, Business Results, Wheel Gallery, How It Works)
- `service.py` — backend: production pipeline (skin_color_analyzer
  functions) + exported-model inference; testable from the CLI
  (`python service.py path/to/photo.jpg`)
- `src/` — the production pipeline modules (copied from the research
  project, `harmony-analyzer/`)
- `models/` — face_landmarker.task + the four exported joblib pipelines
- `results/` — the CSVs the charts read, the 12-face wheel artifacts, and
  `features.csv` (lets `src/export_models.py` retrain the models — same
  protocol, numbers reproduce the reports: family 616/910 etc.)
- `data/wheel_faces/` — two sample photos for the demo buttons

## Honesty notes (the app says these too)

- The demo runs the real code and the real models — and shows each
  model's test-set accuracy next to its prediction.
- The wheel-gallery numbers are in-sample (thresholds were calibrated on
  those faces); the business result is a null — the app leads with it.
