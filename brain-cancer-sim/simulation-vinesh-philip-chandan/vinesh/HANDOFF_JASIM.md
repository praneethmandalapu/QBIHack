# PDE → Visualization Handoff (Vinesh → Jasim)

**From:** Vinesh (Person 2, PDE engine) · **To:** Jasim (Person 3, visualization)
**TL;DR:** Real calibrated glioma growth sequences are ready as frame stacks that
plug straight into your `render_sequence()` and `growth_analytics()`. Confirmed
end-to-end (writes a working Plotly HTML). No glue code needed.

---

## 1. Where the data is

Frame stacks live on the shared drive (they're `data/` — gitignored, ~27 MB each):

```
D:/pde-input-vinesh/frames-for-jasim/
├── manifest.json                              # index of all cases (load this first)
├── glioma_<pid>_idh<status>_grade<g>_frames.npy   # the sequence  (T, Z, Y, X)
└── glioma_<pid>_idh<status>_grade<g>_frames.json  # per-case metadata
```

The generator code is in the repo (`make_jasim_sequences.py`, `export_frames.py`) so
anyone can regenerate — see §6.

## 2. Frame format (matches your render_3d contract ✅)

| | |
|---|---|
| `*_frames.npy` | `float32`, shape `(T, Z, Y, X)` — a stack of T density frames |
| each frame | `(Z, Y, X) = (64, 64, 64)`, slices-first |
| values | tumor density in **[0, 1]** (your isosurface `0.5` works directly) |
| T | 26 frames per case |

## 3. Load + render (one call)

```python
import sys
sys.path.insert(0, ".../simulation-vinesh-philip-chandan/vinesh")
sys.path.insert(0, ".../visualization-jasim")

from export_frames import load_sequence        # helper I provide
import render_3d as r

frames, meta = load_sequence("D:/pde-input-vinesh/frames-for-jasim/glioma_100118_idhWT_grade4")
#   frames : list of (64,64,64) float32 in [0,1]   <-- drop-in for make_dummy_sequence()
#   meta   : dict (see §4)

fig = r.render_sequence(frames)                 # animated 3D player (Play/Pause)
A   = r.growth_analytics(frames,
                         spacing=meta["spacing_mm"],
                         days_per_step=meta["days_per_step"])   # true-scale mm³ + real days
```

`load_sequence` accepts the stem (no extension), `..._frames`, or the full `.npy` path.

### Wiring into your two front-ends
- **`build_site.py`**: replace the `SEQ = r.make_*_sequence(...)` line with
  `SEQ, meta = load_sequence(<case>)`, then
  `A = r.growth_analytics(SEQ, spacing=meta["spacing_mm"], days_per_step=meta["days_per_step"])`.
  Everything downstream (frames, scrubber, KPIs, curve) derives from `SEQ`/`A`.
- **`demo_app.py`**: build the case dropdown from `manifest.json` (`cases[].slug`),
  `load_sequence` the selected slug, render as above.

## 4. Per-case metadata schema (`*_frames.json`)

```jsonc
{
  "slug": "glioma_100118_idhWT_grade4",
  "n_frames": 26,
  "shape_per_frame": [64, 64, 64],
  "value_range": [0.0, 1.0],
  "spacing_mm": [1.0, 1.0, 1.0],   // -> growth_analytics(spacing=...)
  "days_per_step": 2.2,            // -> growth_analytics(days_per_step=...)
  "interval_days": 56,             // real baseline->followup gap
  "risk_multiplier": 4.87,         // calibrated growth knob
  "sim_params": {"risk_multiplier": 4.87, "delta": 0.0},
  "volume_mm3_per_frame": [...],   // convenience curve (see caveat §7)
  "patient_id": "100118", "disease": "glioma",
  "idh_status": "WT", "grade": "4.0", "regime": "growth",
  "real_growth_pct": 609.1,        // measured from expert masks
  "baseline_mm3": 10905, "followup_mm3": 77327
}
```

## 5. The case set (`manifest.json`) — 7 real patients

| slug | IDH / grade | regime | real growth |
|---|---|---|---|
| glioma_100118_idhWT_grade4 | WT / 4 | growth | +609% |
| glioma_100260_idhWT_grade4 | WT / 4 | growth | +426% |
| glioma_100220_idhWT_grade4 | WT / 4 | growth | +182% |
| glioma_100130_idhmut_grade3 | mut / 3 | regression | +14% |
| glioma_100134_idhmut_grade3 | mut / 3 | regression | +9% |
| glioma_100002_idhmut_grade2 | mut / 2 | regression | +3% |
| glioma_100192_idhmut_grade2 | mut / 2 | regression | −58% |

Aggressive IDH-WT GBMs grow fast; indolent IDH-mutant gliomas creep or regress —
good contrast for the demo (a growing case and a shrinking case).

## 6. Regenerate / make more cases

```bash
cd brain-cancer-sim/simulation-vinesh-philip-chandan/vinesh
# whole labeled cohort + manifest:
python make_jasim_sequences.py --data-dir D:/pde-input-vinesh --out D:/pde-input-vinesh/frames-for-jasim
# a single custom case:
python export_frames.py --baseline <baseline.npy> --slug mycase --risk 2.0 --interval-days 90
```

## 7. Honesty notes (please keep on the site)
- **Render the continuous density** — it grows smoothly and looks right.
- `volume_mm3_per_frame` uses a 0.5 threshold and is **sensitive at the start**;
  do **not** quote its raw % on screen. For headline numbers use `real_growth_pct`
  (measured) or `growth_analytics` outputs.
- These are **calibrated** to reproduce the real followup (a fit), not out-of-sample
  predictions. Label them "calibrated to real follow-up," not "predicted."

## 8. What you need installed
`numpy`, `plotly` (already in your viz deps). `export_frames`/`load_sequence` need
only numpy. Verified end-to-end on `render_sequence` + `growth_analytics`.
