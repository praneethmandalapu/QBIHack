# Breast tumor segmentation (Philip-Chandan)

Automated **focal lesion segmentation** on TCGA-Breast-Radiogenomics DCE-MRI, with benchmark evaluation against radiologist `.les` ground truth.

## Quick start

```bash
cd breast-cancer-sim

# Write .les reference masks + QC for rev2 baselines
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/segmentation/run_benchmark.py --all-primary

# One slug — cuboid_enhancement (baseline spike)
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/segmentation/segment.py \
  --slug luminal_a_TCGA-AR-A1AX_baseline --method cuboid_enhancement
```

When `{slug}_cuboid_enhancement_mask.npy`, `{slug}_nnunet_mask.npy` (or `_medsam_`) exists, re-run `run_benchmark.py` to append Dice rows to `segmentation_comparison.csv`.

## Docs

| File | Purpose |
|------|---------|
| [`PLAN.md`](PLAN.md) | Phased plan — benchmark harness **done**; nnU-Net next |

## Layout

| Module | Role |
|--------|------|
| [`run_benchmark.py`](run_benchmark.py) | `.les` reference + evaluate on-disk method masks |
| [`ground_truth.py`](ground_truth.py) | Embed TCIA `.les` → `{slug}_les_mask.npy` |
| [`evaluate.py`](evaluate.py) | Dice / volume vs reference |
| [`qc_overlay.py`](qc_overlay.py) | Mid-z overlay PNGs |
| [`segment.py`](segment.py) | CLI for single method |
| [`methods/cuboid_enhancement.py`](methods/cuboid_enhancement.py) | `.les` cuboid + local DCE enhancement (baseline spike) |

## Outputs

```
data/processed/segmentation-philip-chandan/{slug}_les_mask.npy
data/processed/segmentation-philip-chandan/{slug}_les_mask.json
data/processed/segmentation-philip-chandan/segmentation_comparison.csv
data/qc/segmentation-philip-chandan/{slug}_{method}_overlay_mid-z.png
```

**Otsu** is not benchmarked here — validated as failed (whole-breast bright regions) in [`../validation/VALIDATION.md`](../validation/VALIDATION.md).

## Status

| Phase | Status |
|-------|--------|
| **1** Benchmark harness (`.les` reference, CSV, QC) | **done** |
| **1b** `cuboid_enhancement` classical spike | **done** — rev2 Dice 0.21–0.23 vs `.les` (Otsu was 0.0) |
| **2** MAMA-MIA nnU-Net | **next** |
| Longitudinal follow-up masks | pending |
