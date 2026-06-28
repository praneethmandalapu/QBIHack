# Philip-Chandan stretch — PyRadiomics

Isolated Phase 2+ work: full-resolution raw MR → tumor mask → (later) PyRadiomics features.

**Does not modify** sprint handoff code (`tcia_extractor.py`, `vinesh/`, etc.).

## Quickstart

```bash
cd breast-cancer-sim

# Prep mask for one slug (default: spike baseline)
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/stretch/prep_volume.py

# QC overlay PNG
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/stretch/qc_mask_overlay.py \
  --slug luminal_a_TCGA-AR-A1AX_baseline

# Tests
.venv/bin/python -m pytest simulation-vinesh-philip-chandan/philip-chandan/stretch/tests/

# Extract features (PyRadiomics default)
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/stretch/extract_radiomics.py \
  --slug luminal_a_TCGA-AR-A1AX_baseline

# Batch → CSV
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/stretch/run_all_radiomics.py

# Longitudinal deltas
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/stretch/compare_longitudinal.py
```

## Inputs (read-only)

- `data/processed/raw-extract-philip-chandan/{slug}.npy` + `.json`
- `data/processed/raw-extract-philip-chandan/manifest.json`

## Outputs (gitignored)

- `data/processed/radiomics-philip-chandan/{slug}_mask.npy`
- `data/processed/radiomics-philip-chandan/features_all.csv`
- `data/processed/radiomics-philip-chandan/features_longitudinal.csv`
- `data/qc/radiomics-philip-chandan/{slug}_mask_overlay_mid-z.png`

## Mask algorithm

Mirrors `vinesh/calibrate.py::isolate_tumor` (Otsu on nonzero voxels + largest connected component) on percentile-normalized **raw** volumes at native spacing — not on 64³ PDE inputs.

See [STRETCH_PLAN.md](STRETCH_PLAN.md) for Phases 3–6 (PyRadiomics extraction, CSV handoff to Praneeth).
