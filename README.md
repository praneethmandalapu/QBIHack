# QBIHack — OncoPulse

Multi-cancer tumor simulation and visualization: breast (TCGA-BRCA) and brain
(glioma, UCSF-LPTDG). Python work uses the shared venv under
`breast-cancer-sim/.venv/` (see `breast-cancer-sim/requirements.txt`).

**Public website:** the OncoPulse viewer is published from **`main`** —
`docs/index.html` is a copy of the built site (`build_site.py` →
`breast-cancer-sim/visualization-jasim/site/index.html`). After viewer changes,
rebuild and commit `docs/index.html` on `main` to update what GitHub Pages serves.

| Area | Path |
|------|------|
| Breast imaging + cohort | `breast-cancer-sim/simulation-vinesh-philip-chandan/philip-chandan/` |
| Brain imaging + cohort | `brain-cancer-sim/simulation-vinesh-philip-chandan/philip-chandan/` |
| PDE solver (Vinesh) | `*/simulation-vinesh-philip-chandan/vinesh/` |
| Genomic risk models | [`RISK_MODELS.md`](RISK_MODELS.md) |
| Visualization (Jasim) | `breast-cancer-sim/visualization-jasim/` |

---

## Run the OncoPulse website (breast + brain)

Jasim's **tabbed static site** (brain growth animation + breast PDE growth player)
lives in `breast-cancer-sim/visualization-jasim/`. Build once, then serve the
`site/` folder locally. Copy the output to `docs/index.html` on **`main`** to
update the public site (GitHub Pages serves from that path on `main`, not
`gh-pages`).

**Prerequisites:** venv installed (`breast-cancer-sim/.venv`), local processed
data (brain PDE baselines/follow-ups for the UCSF cohort, breast PDE volumes for
the primary TCGA pair). Large `.npy` frame stacks are gitignored; regenerate them
locally via `build_site.py`.

```bash
cd breast-cancer-sim/visualization-jasim

# 1) Bake breast + brain into a single self-contained page
#    (auto-regenerates brain frame stacks via make_brain_frames if missing)
../.venv/bin/python build_site.py
cp site/index.html ../../docs/index.html

# 2) Serve locally
../.venv/bin/python -m http.server 8080 --directory site
```

Open **http://localhost:8080**

**Windows (Vinesh):** replace `../.venv/bin/python` with
`..\.venv\Scripts\python.exe`.

After changing imaging data, risk scores, or viewer code, re-run `build_site.py`,
copy to `docs/index.html`, and commit on **`main`** before refreshing the browser
or expecting the public URL to change. Risk handoff CSVs for the imaging cohort
live in `*/visualization-jasim/risk/` (regenerate with each folder's `export_risk.py`).

### Other visualization entry points

| Surface | Command | Notes |
|---------|---------|-------|
| Streamlit demo | `../.venv/bin/python -m streamlit run demo_app.py` | Synthetic treatment scenario; not the two-cancer tabbed site |
| Vinesh/Philip app | `../.venv/bin/python -m streamlit run app/app.py` | Predict / Explain / Simulate tabs (work in progress) |

More detail: `breast-cancer-sim/visualization-jasim/HANDOFF_VINESH.md`.
