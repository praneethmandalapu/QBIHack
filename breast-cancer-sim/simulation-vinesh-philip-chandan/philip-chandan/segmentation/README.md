# Breast tumor segmentation (Philip-Chandan)

Automated **focal lesion segmentation** on TCGA-Breast-Radiogenomics DCE-MRI, with benchmark evaluation against radiologist `.les` ground truth.

## Quick start

```bash
cd breast-cancer-sim

# Write .les reference masks + QC for rev2 baselines
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/segmentation/run_benchmark.py --all-primary

# One slug
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/segmentation/segment.py \
  --slug basal_TCGA-AR-A1AQ_baseline --method les
```

When `{slug}_nnunet_mask.npy` (or `_medsam_`) exists, re-run `run_benchmark.py` to append Dice rows to `segmentation_comparison.csv`.

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
| Benchmark harness | **done** |
| MAMA-MIA nnU-Net | pending |
| Longitudinal follow-up masks | pending |
