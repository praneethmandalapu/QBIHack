# Validation — tumor masks and radiomics

Living instructions for validating Philip-Chandan segmentation and radiomics against ground truth on **TCGA-Breast-Radiogenomics** (same cohort as our primaries), without switching datasets unless necessary.

## Current state

| What | Status |
|------|--------|
| Longitudinal MR DICOM (4 volumes) | Downloaded and exported to raw `.npy` |
| Genomic labels (PAM50, ER/PR, survival) | Praneeth / GDC — separate from imaging masks |
| **Radiologist tumor masks (`.les`)** | Available on TCIA for **~91** TCGA-BRCA MRI patients — **not downloaded or used yet** |
| Our tumor masks | Otsu + largest connected component ([`stretch/prep_volume.py`](stretch/prep_volume.py), mirrored in [`vinesh/calibrate.py`](../vinesh/calibrate.py)) |
| Segmentation “validation” today | Visual QC overlays only — not Dice vs expert |

[`cohort.json`](cohort/cohort.json) sets `"use_les_mask": true` and `"subset": "TCGA-Breast-Radiogenomics"` as **intent**; [`tcia_extractor.py`](tcia_extractor.py) does not load `.les` files yet.

**TCIA reference:** [TCGA-Breast-Radiogenomics — Segmented Lesions (`*.les`)](https://www.cancerimagingarchive.net/analysis-result/tcga-breast-radiogenomics/)

---

## Do you need another dataset?

**Not necessarily** — you may already have ground truth on the same cohort:

1. **Download `.les` masks** from TCGA-Breast-Radiogenomics for your primaries (if present).
2. **Parse `.les` → 3D mask** aligned to the DCE series used for annotation.
3. **Compare Otsu mask vs `.les`** (Dice, volume error) on baseline — and on follow-up **only if** a `.les` exists for that timepoint.

### `.les` file format (TCIA)

Each file is binary:

1. Six `uint16` values: inclusive cuboid bounds `y_start, y_end, x_start, x_end, z_start, z_end` (relative to the annotated DCE volume).
2. Remaining bytes: `int8` voxels (0 = background, 1 = lesion) for that cuboid, row-major over the box.

Naming convention: `*Sn-m.les` — `n` = DCE sequence index, `m` = lesion index (e.g. `S2-1.les` = sequence 2, lesion 1). Use the sequence that matches the series radiologists annotated.

### Caveats

- Masks are usually **one lesion / one annotated timepoint**, not guaranteed for both baseline and follow-up.
- Your primaries might be in the 91, but **~48** TCGA-BRCA MRI patients on TCIA **do not** have `.les` masks — confirm per barcode before planning validation.
- **Axis/spacing alignment** between `.les` cuboid coordinates and our SimpleITK `(Z, Y, X)` volumes needs care (index order, series choice, resampling).

### If `TCGA-AR-A1AX` / `TCGA-AR-A1AQ` lack `.les`

- Pick a **backup** from the 91 with both MRI + mask — see [`cohort/COHORT.md`](cohort/COHORT.md) backups.
- Use an **external benchmark** (e.g. Duke Breast Cancer MRI, ISPY) only if you need broader segmentation benchmarking beyond TCGA.

---

## Practical recommendations

| Goal | Path |
|------|------|
| **Ship demo / Praneeth handoff** | Keep Otsu; review QC overlays in `data/qc/radiomics-philip-chandan/`; run [`stretch/run_all_radiomics.py`](stretch/run_all_radiomics.py) → `features_all.csv` |
| **Validate segmentation** | Wire `.les` loader + Dice/volume-error QC (same TCIA collection); optional script under `stretch/` |
| **Validate radiomics pipeline** | Compare selected features to TCIA’s **Quantitative Radiomic Features** XLS on masked cases |

---

## Suggested workflow

### Step 0 — Confirm `.les` availability per primary

For each barcode (`TCGA-AR-A1AX`, `TCGA-AR-A1AQ`):

- Query or browse [TCGA-Breast-Radiogenomics segmentations](https://www.cancerimagingarchive.net/analysis-result/tcga-breast-radiogenomics/) for matching `*.les` files.
- Note which DCE sequence (`Sn`) and timepoint they correspond to.
- If missing, pick a backup patient from [`COHORT.md`](cohort/COHORT.md) that has both MRI and `.les`.

### Step 1 — Download segmentations

- Download **Segmented Lesions (`*.les`)** ZIP from TCIA (Radiogenomics collection).
- Store under a gitignored path, e.g. `data/raw/tcia-radiogenomics/lesions/`.
- Do **not** commit `.les` or large ZIPs to git.

### Step 2 — Parse and align

- Implement a small `.les` reader (cuboid header + voxel payload → dense 3D mask in annotation frame).
- Map cuboid indices onto the **same DCE series** used for our raw extract (match series UID / study date from DICOM).
- Document axis convention: TCIA docs use `y, x, z` bounds; our volumes are `(Z, Y, X)` from SimpleITK — verify with one known case before batch metrics.

### Step 3 — Compare masks

On each slug with a matching `.les`:

| Metric | Compare |
|--------|---------|
| **Dice** | Otsu mask ([`stretch/prep_volume.py`](stretch/prep_volume.py)) vs radiologist mask |
| **Volume error** | Relative difference in mm³ (use `spacing_mm` from raw extract sidecar) |
| **Visual QC** | Overlay both contours on mid-Z slice (similar to [`stretch/qc_mask_overlay.py`](stretch/qc_mask_overlay.py)) |

Acceptance is project-defined; document results in QC PNGs or a small CSV (`validation_metrics.csv`).

### Step 4 — Optional radiomics cross-check

For patients in the 91 with TCIA’s pre-computed feature spreadsheets:

- Extract the same feature classes (firstorder, shape, GLCM) with our pipeline on **radiologist** mask vs **Otsu** mask.
- Compare a handful of features to TCIA XLS values (allow tolerance; bin width and normalization must match their protocol or comparison is qualitative only).

---

## What we are *not* validating with `.les`

- **PDE growth calibration** ([`vinesh/calibrate.py`](../vinesh/calibrate.py)) uses follow-up **burden**, not expert segmentation — longitudinal “validation” there is consistency with observed change, not Dice.
- **Genomic risk models** (Praneeth) use GDC/clinical data, not imaging masks.

---

## Next implementation (optional)

If validation becomes a priority:

1. `stretch/load_les_mask.py` — parse `.les` → `(Z, Y, X)` numpy mask.
2. `stretch/validate_segmentation.py` — Dice + volume error vs Otsu; write QC overlays and metrics CSV.
3. Unit test on a synthetic `.les`-sized cuboid (round-trip header + voxels).

Until then, treat Otsu masks as **heuristic ROIs** suitable for demo and relative longitudinal comparison, not as clinician-equivalent segmentations.

---

## Summary

Annotations **exist on TCIA** for a subset of our collection, but we **are not using them yet**. You do **not** need a new dataset first — check whether your two primaries have `.les` files, then either use those or pick a radiogenomics case that does.
