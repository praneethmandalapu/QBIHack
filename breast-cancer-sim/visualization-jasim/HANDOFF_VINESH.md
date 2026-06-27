# OncoPulse Visualization — Handoff to Vinesh (Person 2)

**From:** Jasim (Person 3, visualization) · **To:** Vinesh (Person 2, PDE engine)
**TL;DR:** Your `solve_growth()` output plugs straight into my renderers. The data
contract already matches — this doc shows you exactly where to wire it in, how to
run the two front-ends, and the few things I need from your side.

---

## 1. What this layer is

Two surfaces, both fed by the **same** sequence of 3D density frames:

| Surface | File | What it is | Run with |
|---|---|---|---|
| **OncoPulse site** | `build_site.py` → `site/index.html` | A polished, self-contained showcase webpage (the demo/pitch site). Embeds the real Plotly 3D figures + a time scrubber. | build, then serve `site/` |
| **Streamlit dashboard** | `demo_app.py` | Interactive tool: scenario controls, render-mode toggle, live KPIs. | `streamlit run demo_app.py` |

Both call the rendering + analytics engine in **`render_3d.py`**. Colors live in
`color_maps.py`. Right now both run on synthetic data (`make_*_sequence()`); the
job below is swapping in **your** `solve_growth()`.

---

## 2. The data contract (already aligned ✅)

Your `solve_growth()` returns exactly what I consume — no glue needed:

```
each frame : np.float32, shape (Z, Y, X) = (D, H, W)   # slices-first
values     : tumor density in [0, 1]
return     : list[np.ndarray]  (you return timesteps + 1 frames — fine, any length)
spacing    : (z, y, x) mm  -> pass to my analytics/slices for true-scale mm³ & mm
```

Two things I rely on — please keep them true:
1. **Values stay in `[0, 1]`.** My isosurface threshold is `0.5`, and the tissue
   shells (rim/viable/hypoxic/core) are density bands at `0.5 / 0.62 / 0.7 / 0.85`.
   If the field blows past 1.0 (CFL instability → NaN), the 3D render goes blank.
   Your `_check_cfl()` guard already protects this — keep it on.
2. **A dense core should emerge** for the Layers/Cutaway views to look good. Your
   `delta`/`apply_drug()` death term that carves a necrotic centre is exactly what
   drives those — no extra work, it just shows up.

> Note: I removed the **necrotic-fraction %** *metric* from the UI, but the 3D
> still renders the dense core as a tissue layer. Your model doesn't change.

---

## 3. Wiring your engine in (the only real task)

### A. The site — `build_site.py`
Find this block near the top (the synthetic scenario):

```python
SEQ = r.make_treatment_sequence(28, (64, 64, 64), therapy_start=12, response=0.6)
A = r.growth_analytics(SEQ, spacing=(1.0, 1.0, 1.0), days_per_step=7)
```

Replace with your solver:

```python
from tumor_pde_solver import solve_growth, dummy_volume   # add at top of file
# later, real data: from prepare_pde_input import prepare_pde_input
#                    (or philip-chandan/tcia_extractor.extract_volume())

initial = dummy_volume((64, 64, 64))                # later: real TCIA volume
params  = {"rho": 0.25, "risk_multiplier": 1.3,     # Praneeth's value
           "spacing": (1.0, 1.0, 1.0)}              # Philip's DICOM spacing
SEQ = solve_growth(initial, timesteps=27, dt=0.5, params=params)
A   = r.growth_analytics(SEQ, spacing=params["spacing"], days_per_step=7)
```

Everything downstream (3D frames, scrubber, KPIs, curve) is derived from `SEQ`
and `A`, so nothing else changes. Then rebuild:

```bash
../.venv/Scripts/python.exe build_site.py
```

### B. The dashboard — `demo_app.py`
Same idea, inside the cached `build()` function:

```python
@st.cache_data(show_spinner="Simulating tumor evolution…")
def build(...):
    seq = solve_growth(dummy_volume((grid, grid, grid)),
                       timesteps=steps-1, dt=0.5,
                       params={"risk_multiplier": 1.3, "spacing": (1,1,1)})
    disp = [r.downsample(f, factor) for f in seq]
    return seq, disp
```

### The "on therapy" story
The site shows a **grow-then-regress** arc (RECIST *Partial Response*). To
reproduce that with your engine, drive regression after a therapy step — e.g.
raise `params["delta"]` (or apply `growth_interventions.apply_drug()`) partway
through. If you just run plain growth, the site will read *Progressive Disease* —
which is also correct, just a different scenario. Either works; the visualization
adapts to whatever `SEQ` you hand it.

---

## 4. The API you can call (all in `render_3d.py`)

| Function | Returns | Use |
|---|---|---|
| `growth_analytics(frames, spacing, days_per_step)` | dict | **Start here.** Per-frame series + doubling time + RECIST + peak index. |
| `tumor_metrics(frame, spacing)` | dict | One-frame: `total_mm3`, `max_diameter_mm`, `longest_axis_mm`. |
| `render_volumetric(frame)` | Plotly fig | Hero volumetric render. |
| `render_layers(frame)` | Plotly fig | Rim → viable → hypoxic → dense core shells. |
| `render_cutaway(frame, plane="y")` | Plotly fig | Clipped to expose the core. |
| `render_sequence(frames)` | Plotly fig | Animated (Plotly's own play/slider). |
| `render_slices(frame, spacing)` | Plotly fig | Axial/coronal/sagittal triptych. |
| `render_growth_curve(analytics)` | Plotly fig | Volume + max-diameter telemetry. |
| `downsample(frame, factor)` | ndarray | Decimate for display (see perf note). |

`growth_analytics(...)` keys: `days`, `volume_mm3`, `diameter_mm`, `peak_index`,
`doubling_time_days`, `diameter_change_pct`, `recist`, `series`.

---

## 5. Running it

```bash
# from breast-cancer-sim/visualization-jasim
cd breast-cancer-sim/visualization-jasim

# --- the showcase site ---
../.venv/Scripts/python.exe build_site.py
../.venv/Scripts/python.exe -m http.server 8080 --directory site
# open http://localhost:8080

# --- the Streamlit dashboard ---
../.venv/Scripts/python.exe -m streamlit run demo_app.py
```

Deps are `numpy`, `plotly`, `streamlit` (already in `requirements.txt` + the
`.venv` one level up). No scikit-image needed.

---

## 6. Performance note (important for real data)

A full 64³ × ~28-frame animation is heavy. The rule I follow and you should too:

- **Full resolution → metrics** (`growth_analytics` on `SEQ`).
- **`downsample(frame, 2)` → display** (3D + scrubber). 64³ → 32³ keeps the page
  light and the WebGL smooth.
- Wrap any frame-building in `@st.cache_data` in Streamlit so heavy arrays aren't
  rebuilt every rerun.

If your real volumes are bigger than 64³, downsample harder (factor 3–4) for the
3D view — the analytics can still run on full res.

---

## 7. What I need from you

1. **`solve_growth()` output in `[0, 1]`**, `(Z,Y,X)` float32 — you already do this.
2. **Real `spacing` (mm)** in `params` once Philip's DICOM extractor lands, so my
   mm³ / diameter numbers are true-scale (pass it through to `growth_analytics`).
3. **Sensible `dt`/`params`** that satisfy your CFL guard so frames don't NaN.
4. A heads-up if your frame **count** or **typical grid size** changes a lot, so I
   can retune the downsample factor for the scrubber.

Ping me when your engine is calibrated and I'll do the swap + rebuild — it's a
~5-line change in each front-end. Contract's already aligned, so it should "just
work."

— Jasim
