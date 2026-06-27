# Vinesh spike checklist (steps 5–7)

Parent plan: [`../HANDOFF_SPIKE.md`](../HANDOFF_SPIKE.md)

**Contract:** [`../handoff_contract.json`](../handoff_contract.json) (`version` **1.0.0**)

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

Implement in `prepare_pde_input.py` (you own this). Read targets from `handoff_contract.json` via `handoff_contract.pde_input_spec()` — do not hardcode 64 or 1.0 mm:

1. Load raw array + `spacing_mm` from JSON.
2. Resample toward contract `target_spacing_mm` (`scipy.ndimage.zoom` or equivalent).
3. Crop or downsample to contract `max_shape`.
4. Normalize to contract `value_range`; tumor voxels follow `tumor_burden_rule`.
5. Write `pde-input-vinesh/{slug}.npy` + `.json` with matching `contract_version`.

**Do not** re-download or re-parse DICOM for the spike.

---

## Step 6 — PDE input manifest

Sidecar JSON is written by `save_pde_input()` using `handoff_contract.json`. Expected fields include `contract_version`, `shape`, `spacing_mm`, and `value_semantics`.

---

## Step 7 — Integration with `solve_growth`

```python
import numpy as np
from handoff_contract import solver_spec, spike_patient
from tumor_pde_solver import solve_growth

spec = solver_spec()
slug = spike_patient()["slug"]
vol = np.load(f"data/processed/pde-input-vinesh/{slug}.npy")
frames = solve_growth(
    vol,
    timesteps=spec["timesteps"],
    dt=spec["dt"],
    params=spec["default_params"],
)
```

**Done when:** solver runs without reformatting the array on Philip-Chandan's side.

If it fails, report: expected vs actual **shape, dtype, axis order, value range, spacing assumption**.

---

## Parallel work before raw extract lands

You can still:

- Implement `solve_growth()` against a **dummy sphere** in `pde-input-vinesh/` shape.
- Stub `prepare_pde_input.py` and test on synthetic raw `.npy` you create locally.
- Swap in Philip-Chandan's real raw extract when notified — no DICOM work required.
