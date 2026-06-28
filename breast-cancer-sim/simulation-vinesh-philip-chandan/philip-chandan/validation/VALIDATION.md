# Validation — tumor masks and radiomics

Living instructions for validating Philip-Chandan segmentation and radiomics against ground truth on **TCGA-Breast-Radiogenomics** (same cohort as our primaries), without switching datasets unless necessary.

## Current state

| What | Status |
|------|--------|
| Longitudinal MR DICOM (4 volumes) | Downloaded and exported to raw `.npy` |
| Genomic labels (PAM50, ER/PR, survival) | Praneeth / GDC — separate from imaging masks |
| **Radiologist tumor masks (`.les`)** | Downloaded (~103 KB ZIP, 91 patients) → `data/raw/tcia-radiogenomics/lesions/` |
| **`.les` loader** | **done** — [`../stretch/load_les_mask.py`](../stretch/load_les_mask.py) |
| **Segmentation validation** | **done** — [`../stretch/validate_segmentation.py`](../stretch/validate_segmentation.py) |
| Our tumor masks | Otsu + largest connected component ([`../stretch/prep_volume.py`](../stretch/prep_volume.py), mirrored in [`../../vinesh/calibrate.py`](../../vinesh/calibrate.py)) |
| Metrics on disk | `data/processed/validation-philip-chandan/validation_metrics.csv` |
| QC overlays | `data/qc/validation-philip-chandan/*_{les,otsu}_mid-z.png` |
| Pipeline report | Section 7 + Figure 6 in [`../PIPELINE_REPORT.pdf`](../PIPELINE_REPORT.pdf) |

[`../cohort/cohort.json`](../cohort/cohort.json) sets `"use_les_mask": true` as **intent** for future radiomics; sprint/PDE handoff still uses Otsu today. [`../tcia_extractor.py`](../tcia_extractor.py) does not load `.les` in the main export path yet.

**TCIA reference:** [TCGA-Breast-Radiogenomics — Segmented Lesions (`*.les`)](https://www.cancerimagingarchive.net/analysis-result/tcga-breast-radiogenomics/)

---

## Findings — Otsu vs expert `.les` (rev2 baselines)

**Conclusion: our Otsu segmentation scores poorly against radiologist `.les` masks.** Treat Otsu ROIs as demo heuristics for PDE/radiomics prototyping, not clinician-equivalent tumor boundaries.

Both rev2 primaries have baseline `.les` files. Annotations are on **DCE sequence S2 = VIBRANT** (same series as our raw extract when slice counts match). Neither primary has a follow-up `.les`.

| Slug | TCGA ID | `.les` file | Dice | Expert vol (mm³) | Otsu vol (mm³) | Area (Otsu / `.les`) |
|------|---------|-------------|------|------------------|----------------|----------------------|
| `luminal_a_TCGA-AR-A1AX_baseline` | TCGA-AR-A1AX | `TCGA-AR-A1AX-S2-1.les` | **0.000** | ~2,927 | ~7,756,222 | **~2,650×** |
| `basal_TCGA-AR-A1AQ_baseline` | TCGA-AR-A1AQ | `TCGA-AR-A1AQ-S2-1.les` | **0.000** | ~6,075 | ~3,762,799 | **~620×** |

Expert masks are small, localized lesions (~1.3k–2.7k voxels). Otsu + largest connected component captures **orders of magnitude more tissue** (~1.7M–3.5M voxels) with **no voxel overlap** (Dice = 0).

### Why Otsu fails here

1. **Wrong region, not just wrong size** — on VIBRANT (352–464 slices), Otsu’s largest bright component often lies on **different z indices** than the radiologist ROI (e.g. A1AX: `.les` peak z ≈ 22, Otsu peak z ≈ 299).
2. **Dynamic series geometry** — full VIBRANT stacks are not a simple 3D slab; SimpleITK stacks hundreds of slices with repeated patient position metadata. Global Otsu + largest-CC is a brittle default.
3. **Thresholding whole FOV** — percentile-normalized Otsu on the entire volume favors large enhancing regions (breast parenchyma / vasculature), not the annotated lesion cuboid.

### Implications

| Use case | Recommendation |
|----------|----------------|
| **PDE demo / relative longitudinal change** | Otsu may still be usable if the team accepts heuristic ROIs (document limitation). |
| **Radiomics vs literature / TCIA features** | Prefer **`.les` masks** (or tightened ROI logic) — see [`../PLAN.md`](../PLAN.md) known issue on Luminal A follow-up. |
| **Claiming segmentation quality** | Do **not** — validation shows expert disagreement, not minor tuning error. |

Re-run validation after local `.les` download:

```bash
cd breast-cancer-sim
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/stretch/validate_segmentation.py --all-primary
```

Regenerate report Section 7 / Figure 6:

```bash
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/generate_pipeline_report.py
```

---

## Step 0 — `.les` availability (done)

| Patient | Subtype | `.les`? | DCE index | Annotated series | Follow-up `.les`? |
|---------|---------|---------|-----------|------------------|-------------------|
| **TCGA-AR-A1AX** | Luminal A | Yes | S2 | VIBRANT (baseline 2002-09-12) | No |
| **TCGA-AR-A1AQ** | Basal-like | Yes | S2 | VIBRANT (baseline 2001-11-21) | No |

Download URL (full archive, ~103 KB): [TCGA_Segmented_Lesions_UofC.zip](https://www.cancerimagingarchive.net/wp-content/uploads/TCGA_Segmented_Lesions_UofC.zip)

Local path: `data/raw/tcia-radiogenomics/lesions/TCGA_Segmented_Lesions_UofC/`

Backup patients with `.les` (if pivoting): `TCGA-BH-A0DK`, `TCGA-BH-A0BQ`, `TCGA-AR-A24S` — see [`../cohort/COHORT.md`](../cohort/COHORT.md).

---

## `.les` file format (TCIA)

Each file is binary:

1. Six `uint16` values in **column-major 3×2 order**: `y_start, x_start, z_start, y_end, x_end, z_end` (inclusive bounds relative to the annotated DCE volume).
2. Remaining bytes: `int8` voxels (0 = background, 1 = lesion) for that cuboid, row-major with shape `(Y, X, Z)`.

Embed into our `(Z, Y, X)` volumes via [`load_les_mask.py`](../stretch/load_les_mask.py) (transpose cuboid → dense mask).

Naming: `*Sn-m.les` — `n` = DCE-MRI sequence index, `m` = lesion index (e.g. `S2-1.les` = sequence 2, lesion 1). DCE order: ax T1 → VIBRANT → BRAVA (see `pick_dce_series` in [`validate_segmentation.py`](../stretch/validate_segmentation.py)).

### Caveats

- Masks are usually **one lesion / one annotated timepoint**, not guaranteed for both baseline and follow-up.
- ~48 TCGA-BRCA MRI patients on TCIA **do not** have `.les` masks — confirm per barcode before planning validation.
- Compare on the **annotated DCE series**, not an arbitrary contrast pick — slice counts must match (validation uses VIBRANT for both primaries).

---

## Implementation (done)

| Component | Path |
|-----------|------|
| `.les` parser | [`../stretch/load_les_mask.py`](../stretch/load_les_mask.py) |
| Dice + volume + area fraction + QC PNGs | [`../stretch/validate_segmentation.py`](../stretch/validate_segmentation.py) |
| 3D napari viewer | [`view_les_napari.py`](view_les_napari.py) |
| Unit tests | [`../stretch/tests/test_load_les_mask.py`](../stretch/tests/test_load_les_mask.py) |
| Paths | [`../stretch/paths.py`](../stretch/paths.py) (`QC_VALIDATION_DIR`, `validation_metrics_csv`, …) |

Metrics written to `data/processed/validation-philip-chandan/validation_metrics.csv`.

---

## 3D inspection (napari)

Load baseline MR from `raw-extract-philip-chandan/` with the matching radiologist `.les` overlay. Only **baseline** slugs have `.les` for rev2 primaries.

The viewer splits stacked **VIBRANT** volumes into temporal DCE phases (4 × ~88 slices for rev2 primaries), loads optional **pre-contrast S1 (Ax T1)**, and can show a **subtraction** layer (active phase − resampled pre-contrast). Use the **Expert mask** dock button to toggle the `.les` overlay (same pattern as brain-cancer-sim `view_volume_napari.py`).

```bash
cd breast-cancer-sim

# List slugs that have a local .les file
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py --list

# Luminal A baseline (VIBRANT S2) — auto-jumps to expert lesion slice
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \
  --slug luminal_a_TCGA-AR-A1AX_baseline

# Basal-like baseline
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \
  --slug basal_TCGA-AR-A1AQ_baseline

# Also overlay Otsu for comparison (active phase only)
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \
  --slug luminal_a_TCGA-AR-A1AX_baseline --otsu

# Cuboid shell only (see MR inside annotation box; no filled lesion blocking view)
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \
  --slug luminal_a_TCGA-AR-A1AX_baseline --cuboid

# Skip pre-contrast download / subtraction
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \
  --slug luminal_a_TCGA-AR-A1AX_baseline --no-precontrast
```

**Dock widgets:** `DCE controls` (phase picker, subtraction, pre-contrast, MIP row, CAD markers, jump-to-lesion), `Expert mask` (show/hide `.les` on detail + hanging panes).

**Layout:** Left = detail viewer with overlays. Right = **hanging protocol** — four linked subtraction (or DCE) phases side-by-side, MIP row beneath, yellow **CAD-style** enhancement peaks (lime = expert centroid). TCIA does not ship vendor CAD; markers are computed from local maxima on subtraction volumes (QC only).

Alternative: export NIfTI via SimpleITK for **ITK-SNAP** / **3D Slicer** / Fiji.

---

## Optional next steps

### Radiomics cross-check (not done)

For patients in the 91 with TCIA’s pre-computed feature spreadsheets:

- Extract the same feature classes (firstorder, shape, GLCM) with our pipeline on **radiologist** mask vs **Otsu** mask.
- Compare a handful of features to TCIA XLS values (allow tolerance; bin width and normalization must match their protocol or comparison is qualitative only).

### Use `.les` in stretch radiomics (not done)

Wire `.les` as ROI in [`../stretch/prep_volume.py`](../stretch/prep_volume.py) when `cohort.json` `"use_les_mask": true` and a matching file exists — would address oversized Otsu masks (especially Luminal A follow-up; see [`../PLAN.md`](../PLAN.md)).

---

## What we are *not* validating with `.les`

- **PDE growth calibration** ([`../../vinesh/calibrate.py`](../../vinesh/calibrate.py)) uses follow-up **burden**, not expert segmentation — longitudinal “validation” there is consistency with observed change, not Dice.
- **Genomic risk models** (Praneeth) use GDC/clinical data, not imaging masks.

---

## Practical recommendations

| Goal | Path |
|------|------|
| **Ship demo / PDE handoff** | Keep Otsu; document poor `.les` agreement in report and this file |
| **Improve segmentation** | Use `.les` where available, or tighten Otsu / connected-component logic; do not expect Dice improvement without ROI change |
| **Validate radiomics pipeline** | Compare features on `.les` vs Otsu ROI; optional TCIA XLS cross-check |
| **Re-run metrics** | `stretch/validate_segmentation.py --all-primary` |
| **Inspect in 3D** | `validation/view_les_napari.py --slug …` |

---

## Summary

Expert `.les` annotations **exist for both rev2 primaries** and are **downloaded and validated**. Our Otsu + largest-component masks **agree poorly** with radiologist ROIs (Dice 0, Otsu area **620–2,650×** expert). No new dataset is required for this conclusion — the gap is algorithm/ROI choice, not missing ground truth.
