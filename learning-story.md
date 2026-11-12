# Learning Story — visual-harmony-dashboard

> Running story of how this project was built — actions, why, and key decisions.
> README.md stays the employer-facing summary.

## 2026-11-09 — Story file created

- The workspace adopted a rule (2026-11-09): every project keeps a story md updated as the build happens, so the build can be retraced later. This file is the story file for this project.
- Pre-story history lives in git history and README.md; nothing has been backfilled yet.

## 2026-11-09 — Streamlit Cloud deploy: Python 3.14 wheel fix

- Deployed on Streamlit Community Cloud. First build stalled: the cloud used **Python 3.14.10** despite `runtime.txt` saying 3.11, and the pinned numpy 2.0.2 / scikit-learn 1.6.1 have no cp314 wheels — uv fell back to compiling both from source (~46 min per attempt, repeated retries).
- Fix: bumped to the oldest versions with cp314 wheels (checked PyPI release files): **numpy 2.3.2, scikit-learn 1.7.2**. Trained joblib models load across sklearn minor versions; numbers reproduce. `runtime.txt` updated to `python-3.14` to match what the cloud actually uses.
- Kept mediapipe==0.10.35 (its wheel downloaded fine; the macOS Metal caveat doesn't apply on Linux) and opencv-contrib-python==5.0.0.93.

## 2026-11-12 — Public app: runtime ImportError on import cv2

- Made the app public on Streamlit Cloud (was private; private apps bounce visitors to the login wall and can't be embedded). Public URL: https://visual-harmony-dashboard.streamlit.app.
- Public app then errored at startup: `ImportError` at `service.py:18` (`import cv2`), original message redacted in the UI.
- Diagnosis: opencv-contrib-python 5.0.0.93 ships abi3 wheels, so the install succeeds on Python 3.14; the failure is at runtime. The Cloud image lacks the system graphics libs the non-headless build links against (libGL.so.1, libglib). Same classic failure seen in Streamlit forums for any opencv app.
- Fix: added `packages.txt` (apt deps `libgl1`, `libglib2.0-0`), which Streamlit Cloud installs before pip. Kept the non-headless opencv pin because mediapipe depends on opencv-contrib-python; a headless swap would create a conflicting second cv2 install.
- Gotcha found on the first rebuild: the Cloud installer pipes `packages.txt` through `xargs apt install -y` verbatim, so comment lines break it ("unmatched single quote", comment words treated as package names, apt aborts and installs nothing). Rule for this repo: `packages.txt` must contain bare package names only, one per line, no comments.

## 2026-11-12 — Repo visibility: private → public

- Flipped github.com/cts3111/visual-harmony-dashboard to PUBLIC. Reason (user): Streamlit Cloud's free tier allows only one private repo, which the ticket-profit-optimizer now uses; the dashboard goes public so both keep streaming.
- Pre-flight: cloned and scanned the repo for PII (names, orgs, handles) before flipping: clean.

## 2026-11-12 — Shared style conventions with the tickets app (site theme)

- User asked both Streamlit apps to follow the same style conventions. Adopted the portfolio site's restyle tokens in a new `.streamlit/config.toml` (this repo previously had no theme config, so it ran on Streamlit defaults): green #1f7a3d primary, #fafafa base, #f2f1ed secondary, slate #1e252c text, sans serif.
- Chart palette in app.py recolored to the site set: green/teal/sage categorical slots, diverging positive pole = teal, green heatmap ramp (replaces the blue ramp), slate ink/muted, #fafafa surface. `FONT` already system sans.
