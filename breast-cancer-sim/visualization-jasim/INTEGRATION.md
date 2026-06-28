# OncoPulse visualization handoff (Person 3 / jasim → Person 4 / Vinesh/Philip)

A clinical "digital twin" of tumor evolution: volumetric rendering, biologically
layered tissue, radiology slices, and quantitative burden tracking (volume,
doubling time, RECIST). Runs on synthetic data today — swap one function for
Person 2's `solve_growth()` and nothing else changes.

## See it now (no setup)
Open any of these in a browser:
`preview_volumetric.html`, `preview_layers.html`, `preview_growth_animation.html`.

Or run the full dashboard:
```bash
cd breast-cancer-sim/visualization-jasim
../.venv/Scripts/python.exe -m streamlit run demo_app.py
```

## API you can call (all return Plotly figures)
| Function | Use |
|---|---|
| `render_volumetric(frame)` | Hero shot — true volumetric transfer function. |
| `render_layers(frame)` | Nested isosurfaces: proliferating rim → viable → hypoxic → necrotic core. |
| `render_cutaway(frame, plane="y")` | Clipped volume exposing the necrotic core. |
| `render_sequence(frames)` | Animated growth (Play + timestep slider baked in). |
| `render_slices(frame, spacing)` | Axial/coronal/sagittal radiology triptych. |
| `render_growth_curve(analytics)` | Tumor-burden-over-time chart. |
| `tumor_metrics(frame, spacing)` | One-frame dict: mm³, necrotic %, max diameter. |
| `growth_analytics(frames, spacing, days_per_step)` | Series + doubling time + RECIST. |
| `downsample(frame, factor)` | Decimate heavy real arrays before display. |

## Drop into `app/tabs/simulate_tab.py`
The viz folder name has a dash, so add it to the path:

```python
import sys, pathlib
import streamlit as st

_VIZ = pathlib.Path(__file__).resolve().parents[2] / "visualization-jasim"
sys.path.append(str(_VIZ))
from render_3d import (render_volumetric, render_growth_curve,
                       growth_analytics, downsample, make_treatment_sequence)

@st.cache_data(show_spinner="Rendering tumor…")
def _frames():
    # TODO: replace with Person 2's engine:
    #   from tumor_pde_solver import solve_growth
    #   seq = solve_growth(initial_volume, timesteps, dt, params)
    seq = make_treatment_sequence()
    return seq, [downsample(f, 2) for f in seq]   # full-res for metrics, light for display

def render() -> None:
    full, disp = _frames()
    a = growth_analytics(full, days_per_step=7)
    st.metric("RECIST", a["recist"], f"{a['diameter_change_pct']:+.0f}% diameter")
    st.plotly_chart(render_volumetric(disp[a["peak_index"]]), use_container_width=True)
    st.plotly_chart(render_growth_curve(a), use_container_width=True)
```

## Performance contract (avoids the memory crash)
- Always `downsample(f, 2)` for **display**; keep full-res only for **metrics**.
- Wrap frame-building in `@st.cache_data` so heavy arrays aren't rebuilt per rerun.
- Animate with `render_sequence` (Isosurface), not `render_volumetric` (Volume is heavy per frame).

## Data contract (from Person 2 / vinesh)
Each frame: `np.float32`, shape `(D,H,W)=(Z,Y,X)`, density ∈ `[0,1]`. Isosurface at 0.5.
`solve_growth()` returns a `list` of these. Real DICOM spacing → pass `spacing=(z,y,x)` mm
into the analytics/slice functions for true-scale mm³ and diameters.
