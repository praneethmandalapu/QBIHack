# Vinesh spike checklist (steps 5–7)

Parent plan: [`../HANDOFF_SPIKE.md`](../HANDOFF_SPIKE.md)

**Contract:** [`../handoff_contract.json`](../handoff_contract.json) (`version` **1.0.0**)

**Case:** `TCGA-AR-A1AX` · Luminal A · `2002-09-12`

**Wait for:** Philip-Chandan files in `data/processed/raw-extract-philip-chandan/TCGA-AR-A1AX/`

---

## Step 0 — Get raw extract on Windows

Philip-Chandan export is done. `data/` is not in git — pick **Option A** or **Option B**.

### Option A — Download + export (recommended)

```powershell
cd path\to\QBIHack
git pull

cd breast-cancer-sim

# One-time if scripts are blocked
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

.\simulation-vinesh-philip-chandan\download_spike_data.ps1 -ExportRaw
```

Check:

```powershell
Test-Path data\processed\raw-extract-philip-chandan\TCGA-AR-A1AX\baseline.npy
Test-Path data\processed\raw-extract-philip-chandan\TCGA-AR-A1AX\baseline.json
```

**Migrating from flat slug layout:** if you still have `luminal_a_TCGA-AR-A1AX_baseline.npy` at the stage root, run once:

```powershell
.\.venv\Scripts\python.exe simulation-vinesh-philip-chandan\philip-chandan\migrate_patient_volume_layout.py
```

### Option B — Copy files Philip-Chandan sent

```powershell
cd path\to\QBIHack\breast-cancer-sim
git pull

New-Item -ItemType Directory -Force -Path data\processed\raw-extract-philip-chandan\TCGA-AR-A1AX
Copy-Item path\to\baseline.npy data\processed\raw-extract-philip-chandan\TCGA-AR-A1AX\
Copy-Item path\to\baseline.json data\processed\raw-extract-philip-chandan\TCGA-AR-A1AX\
```

**Contract check:** open the `.json` and confirm `"contract_version": "1.0.0"`, `"shape": [352, 256, 256]`, `"spacing_mm": [3.0, 0.8594, 0.8594]`.

---

## Your outputs

| Artifact | Path |
|----------|------|
| PDE-ready volume | `data/processed/pde-input-vinesh/TCGA-AR-A1AX/g64/baseline.npy` |
| PDE metadata | `data/processed/pde-input-vinesh/TCGA-AR-A1AX/g64/baseline.json` |
| Solver test dumps | `data/qc/solver-runs-vinesh/` (optional PNG/npy per timestep) |

Use `spike_paths.resolve_pde_input_npy(slug)` in Python — it finds nested or legacy flat paths.

---

## Step 5 — Load raw + resample / crop / normalize

Input: `raw-extract-philip-chandan\TCGA-AR-A1AX\baseline.npy` + `.json` (or legacy flat `{slug}.npy` — readers fall back automatically)

```powershell
cd path\to\QBIHack\breast-cancer-sim

.\.venv\Scripts\python.exe simulation-vinesh-philip-chandan\spike_paths.py
.\.venv\Scripts\python.exe simulation-vinesh-philip-chandan\vinesh\prepare_pde_input.py
```

Implement in `prepare_pde_input.py` (you own this). Read targets from `handoff_contract.json` via `handoff_contract.pde_input_spec()` — do not hardcode 64 or 1.0 mm:

1. Load raw array + `spacing_mm` from JSON.
2. Resample toward contract `target_spacing_mm` (`scipy.ndimage.zoom` or equivalent).
3. Crop or downsample to contract `max_shape`.
4. Normalize to contract `value_range`; tumor voxels follow `tumor_burden_rule`.
5. Write `pde-input-vinesh\{tcga_id}\g64\{timepoint}.npy` + `.json` with matching `contract_version`.

You do **not** need DICOM for the spike if you have the raw extract. Use `download_spike_data.ps1 -ExportRaw` only if Philip-Chandan did not share the `.npy`/`.json`.

---

## Step 6 — PDE input manifest

Sidecar JSON is written by `save_pde_input()` using `handoff_contract.json`. Expected fields include `contract_version`, `grid_size`, `shape`, `spacing_mm`, and `value_semantics`.

---

## Step 7 — Integration with `solve_growth`

```python
import numpy as np
from handoff_contract import solver_spec, spike_patient
from spike_paths import resolve_pde_input_npy
from tumor_pde_solver import solve_growth

spec = solver_spec()
slug = spike_patient()["slug"]
vol = np.load(resolve_pde_input_npy(slug))
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
