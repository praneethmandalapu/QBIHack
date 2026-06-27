# Vinesh spike checklist (steps 5–7)

Parent plan: [`../HANDOFF_SPIKE.md`](../HANDOFF_SPIKE.md)

**Case:** `TCGA-AR-A1AX` · Luminal A · `2002-09-12`

**Wait for:** Philip-Chandan files in `data/processed/raw-extract-philip-chandan/`

---

## Your outputs

| Artifact | Path |
|----------|------|
| PDE-ready volume | `data/processed/pde-input-vinesh/luminal_a_TCGA-AR-A1AX_baseline.npy` |
| PDE metadata | `data/processed/pde-input-vinesh/luminal_a_TCGA-AR-A1AX_baseline.json` |
| Solver test dumps | `data/qc/solver-runs-vinesh/` (optional PNG/npy per timestep) |

---

## Step 5 — Load raw + resample / crop / normalize

Input: `raw-extract-philip-chandan/{slug}.npy` + `.json`

```bash
cd breast-cancer-sim
python simulation-vinesh-philip-chandan/spike_paths.py
python simulation-vinesh-philip-chandan/vinesh/prepare_pde_input.py
```

Implement in `prepare_pde_input.py` (you own this):

1. Load raw array + `spacing_mm` from JSON.
2. Resample toward isotropic **1 mm** (`scipy.ndimage.zoom` or equivalent).
3. Crop or downsample to agreed max shape (default **64³** — confirm with Philip-Chandan).
4. Normalize to **`[0, 1]`**; map initial tumor burden to > 0 (threshold/Otsu on contrast is fine).
5. Write `pde-input-vinesh/{slug}.npy` + `.json`.

**Do not** re-download or re-parse DICOM for the spike.

---

## Step 6 — PDE input manifest

Sidecar JSON should include at minimum:

```json
{
  "slug": "luminal_a_TCGA-AR-A1AX_baseline",
  "source_raw_extract": "data/processed/raw-extract-philip-chandan/luminal_a_TCGA-AR-A1AX_baseline.npy",
  "shape": [64, 64, 64],
  "dtype": "float32",
  "spacing_mm": [1.0, 1.0, 1.0],
  "value_semantics": {"0": "background/healthy", ">0": "initial tumor burden"}
}
```

Adjust shape/spacing to what you actually use.

---

## Step 7 — Integration with `solve_growth`

```python
import numpy as np
from tumor_pde_solver import solve_growth

vol = np.load(
    "data/processed/pde-input-vinesh/luminal_a_TCGA-AR-A1AX_baseline.npy"
)
frames = solve_growth(vol, timesteps=50, dt=0.1, params={"risk_multiplier": 1.2})
```

**Done when:** solver runs without reformatting the array on Philip-Chandan's side.

If it fails, report: expected vs actual **shape, dtype, axis order, value range, spacing assumption**.

---

## Parallel work before raw extract lands

You can still:

- Implement `solve_growth()` against a **dummy sphere** in `pde-input-vinesh/` shape.
- Stub `prepare_pde_input.py` and test on synthetic raw `.npy` you create locally.
- Swap in Philip-Chandan's real raw extract when notified — no DICOM work required.
