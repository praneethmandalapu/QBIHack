# Philip-Chandan Imaging Pipeline — Report

QBIHack breast-cancer-sim · TCGA-BRCA longitudinal MRI from cohort discovery through raw export, napari tumor ROI, and expert-mask PDE handoff to Vinesh.

**PDF:** [`PIPELINE_REPORT.pdf`](PIPELINE_REPORT.pdf) (regenerate with `generate_pipeline_report.py`)  
**Handoff contract:** [`../handoff_contract.json`](../handoff_contract.json) v1.1.0  
**Cohort:** [`cohort/cohort.json`](cohort/cohort.json) rev2

---

## Scope

Philip-Chandan owns **Person 5: Radiomics Pipeline** in this folder. For the sprint we deliver:

1. **Cohort selection** — longitudinal Luminal A vs Basal-like TCGA-BRCA MRI on TCIA, aligned with Praneeth genomics  
2. **Download** — DICOM into `data/raw/tcia/`  
3. **Validate** — structural DICOM checks + visual slice QC  
4. **Export** — raw float32 `.npy` + JSON sidecars for Vinesh (Option B: no normalize/resample here)  
5. **Tumor ROI** — napari aligned-bbox workflow inside TCIA expert bounding boxes (center annotation + manual threshold)  
6. **PDE handoff** — publish expert masks → Vinesh `prepare_pde_input.py` (resample, normalize, crop to 64³)

**Out of scope (sprint):** PyRadiomics feature extraction (`stretch/`), PDE solve, 3D render, Streamlit UI.

---

## 1. Patient selection criteria

Each candidate had to pass:

| Criterion | Why it matters |
|-----------|----------------|
| MRI on TCIA | Many TCGA patients have genomics but no public imaging |
| Longitudinal MR (≥2 study dates) | Baseline + follow-up for growth comparison |
| PAM50 subtype match | Luminal A and Basal-like labels align with cBioPortal |
| Genomics for Praneeth | ER/PR status and survival on GDC/cBioPortal |
| Usable contrast series | Post-contrast T1 (e.g. VIBRANT) for tumor visibility |

Rev1 primaries (`TCGA-BH-A0BR`, `TCGA-A2-A04P`) failed — no MRI on TCIA. Only **19 / 139** TCGA-BRCA patients with MRI on TCIA have multiple MR studies.

### Rev2 primary pair

| Subtype | TCGA ID | Baseline | Follow-up |
|---------|---------|----------|-----------|
| Luminal A | `TCGA-AR-A1AX` | 2002-09-12 | 2003-09-24 (~12 mo) |
| Basal-like | `TCGA-AR-A1AQ` | 2001-11-21 | 2003-05-07 (~17 mo) |

---

## 2. Finding patients — `cohort/cohort_discovery.py`

Automates discovery against TCIA NBIA REST API and cBioPortal (PAM50, ER/PR, survival).

```bash
cd breast-cancer-sim
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py audit
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py find-longitudinal --subtype "Luminal A"
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py recommend-pair
```

Human-readable rationale: [`cohort/COHORT.md`](cohort/COHORT.md).

---

## 3. Download — `download_tcia.py`

DICOM layout:

```
data/raw/tcia/
├── luminal_a/TCGA-AR-A1AX/
│   ├── 2002-09-12/
│   └── 2003-09-24/
└── basal/TCGA-AR-A1AQ/
    ├── 2001-11-21/
    └── 2003-05-07/
```

- Collection `TCGA-BRCA`; prefers post-contrast series (+C, VIBRANT, T1)  
- **idc-index** primary download; **tcia-utils** NBIA fallback  

```bash
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/download_tcia.py --all-primary --longitudinal
```

---

## 4. Validate — `validate_series()` in `tcia_extractor.py`

| Check | What we catch |
|-------|----------------|
| DICOM slices present | Empty or non-image folders |
| Consistent Rows/Columns | Mixed slice dimensions |
| Unique InstanceNumber | Duplicate or corrupted ordering |
| SimpleITK read succeeds | Metadata OK but ITK cannot stack |
| Slice count, shape, spacing_mm | Recorded in sidecar JSON |

Extraction uses **SimpleITK** `ImageSeriesReader`; axis order **(Z, Y, X)** float32.

### Visual QC — `qc_slice_plot.py`

| Output | Path |
|--------|------|
| Plain mid-Z slice | `data/qc/slice-plots-philip-chandan/{slug}_mid-z.png` |
| Intensity overlay (90th pct, QC only) | `data/qc/slice-plots-philip-chandan/{slug}_mid-z-overlay.png` |

Lime contour on raw MR is **QC visualization only** — not tumor segmentation.

**Figure 1 (PDF):** Luminal A baseline raw slice + intensity overlay.

---

## 5. Export — raw volumes for Vinesh

Philip-Chandan writes **raw** extracts; Vinesh owns PDE prep per [`handoff_contract.json`](../handoff_contract.json).

```bash
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py --all-primary
```

Per slug `{subtype_slug}_{tcga_id}_{timepoint}`:

| Output | Path |
|--------|------|
| Raw volume | `data/processed/raw-extract-philip-chandan/{slug}.npy` |
| Metadata | `data/processed/raw-extract-philip-chandan/{slug}.json` |
| Manifest | `data/processed/raw-extract-philip-chandan/manifest.json` v1.1.0 |

### Exported volume shapes (rev2)

| Slug | Shape (Z,Y,X) | Spacing (mm) |
|------|---------------|--------------|
| `luminal_a_TCGA-AR-A1AX_baseline` | [352, 256, 256] | [3.0, 0.8594, 0.8594] |
| `basal_TCGA-AR-A1AQ_baseline` | [464, 256, 256] | [3.0, 0.859375, 0.859375] |
| `luminal_a_TCGA-AR-A1AX_followup` | [552, 512, 512] | [2.2, 0.5273, 0.5273] |
| `basal_TCGA-AR-A1AQ_followup` | [448, 256, 256] | [3.0, 0.9375, 0.9375] |

**Figures 2–3 (PDF):** Baseline subtype comparison and longitudinal LumA pair (raw overlays).

---

## 6. Tumor ROI — TCIA center annotation and napari aligned-bbox workflow

TCGA-Breast-Radiogenomics `.les` files are **not** hand-traced 3D contours. Radiologists marked an approximate **tumor center**; fuzzy c-means auto-segmentation produced sparse FCM voxels inside a small **bounding cuboid** (~31–34% fill for rev2 baselines). The useful prior is the **center + bounding box**, not the sparse dots alone.

Global Otsu on the full breast fails badly against these masks (Section 8). Production ROI uses the **aligned-bbox napari workflow**:

1. Split stacked VIBRANT into DCE phases; rigidly align P2–P3 z-band slabs to phase 1  
2. Inside the tight `.les` Y/X bbox on aligned P2–P3: sweep threshold; keep **center-connected** bright region from expert center  
3. In napari (`view_aligned_cuboid_napari.py`): adjust threshold, jump to elbow, export mask → local `.npy`  
4. `publish_expert_mask.py` → `data/processed/segmentations/{slug}_mask.nii.gz` + `segmentation_path` on raw JSON sidecar  

| Rev2 baseline | Phase @ threshold | Mask voxels | Notes |
|---------------|-------------------|-------------|-------|
| `TCGA-AR-A1AX` (Luminal A) | P2 @ 0.35 | ~1,798 | Center-connected napari export |
| `TCGA-AR-A1AQ` (Basal-like) | P2 @ 0.412 | ~4,683 | Rim lesion; necrotic-core fill (+387 vox) |

```bash
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/validation/view_aligned_cuboid_napari.py \
  --slug luminal_a_TCGA-AR-A1AX_baseline
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/publish_expert_mask.py \
  --slug luminal_a_TCGA-AR-A1AX_baseline
```

**Figures 4–5 (PDF):** Aligned-bbox threshold curve; napari mask overlays on baseline primaries.

---

## 7. PDE simulation input — `vinesh/prepare_pde_input.py` (Vinesh)

After expert mask publish, Vinesh runs:

1. Load raw MR + expert mask from `segmentations/{slug}_mask.nii.gz`  
2. Resample to isotropic **1 mm** (`scipy.ndimage.zoom`)  
3. Min-max normalize to **[0, 1]**  
4. Keep **continuous** intensity inside expert mask; set background to **0**  
5. Crop/pad to **max 64³** centered on tumor center of mass  

Continuous values (not binary) are required so the PDE logistic term `rho*u*(1-u)` can grow the tumor.

### PDE prep QC — `qc_otsu_plot.py`

Documents the same pipeline as `prepare_pde_input.py` (via `prepare_pde_stages()`):

| Output | Path |
|--------|------|
| Normalized slice + expert mask contour (pre-crop) | `data/qc/otsu-segmentation-vinesh/{slug}_otsu-norm-overlay.png` |
| PDE input slice (post-crop 64³) | `data/qc/otsu-segmentation-vinesh/{slug}_pde-input-mid-z.png` |

```bash
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/qc_otsu_plot.py --slug luminal_a_TCGA-AR-A1AX_baseline
.venv/bin/python simulation-vinesh-philip-chandan/vinesh/prepare_pde_input.py --slug luminal_a_TCGA-AR-A1AX_baseline
```

PDE outputs: `data/processed/pde-input-vinesh/{tcga_id}/g64/{timepoint}.npy` + `.json`.

**Figures 6–7 (PDF):** Expert mask on normalized volume; baseline PDE inputs LumA vs Basal.

---

## 8. Why global Otsu was retired — validation vs `.les`

Early sprint used global Otsu + largest connected component. Benchmarking against TCIA radiologist `.les` masks (baseline VIBRANT only) showed Dice ≈ 0 and Otsu area **620–2,650×** expert volume — wrong region, not just wrong size. This motivated the napari aligned-bbox workflow (Section 6). See [`validation/VALIDATION.md`](validation/VALIDATION.md) and PDF Figure 8.

---

## Handoff summary

| Stage | Owner | Deliverable |
|-------|-------|-------------|
| Cohort | Philip-Chandan | `cohort.json` rev2 |
| Raw extract | Philip-Chandan | `.npy` + JSON, `(Z,Y,X)` float32, not normalized |
| Tumor ROI | Philip-Chandan | Napari aligned-bbox → `segmentations/{slug}_mask.nii.gz` |
| PDE input | Vinesh | 64³, 1 mm, [0,1], continuous density inside expert mask |
| Simulation | Vinesh | `solve_growth()` frames |
| Render | Jasim | 3D view from PDE/solver output |
| UI | Vinesh/Philip | Subtype/timepoint toggle from manifest |

---

## Regenerate report artifacts

```bash
cd breast-cancer-sim
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py --all-primary
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/publish_expert_mask.py --slug luminal_a_TCGA-AR-A1AX_baseline
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/qc_otsu_plot.py --slug luminal_a_TCGA-AR-A1AX_baseline
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/generate_pipeline_report.py
```

QC PNGs live under `data/qc/` (gitignored). The committed PDF is a snapshot; regenerate locally when data is present.

---

## Repository map (this folder)

```
philip-chandan/
├── report.md                 # this document
├── PIPELINE_REPORT.pdf       # generated narrative + figures
├── generate_pipeline_report.py
├── cohort/                   # cohort.json, cohort_discovery.py
├── download_tcia.py
├── tcia_extractor.py
├── export_raw_extract.py
├── export_all_raw.py
├── qc_slice_plot.py          # raw MR slice QC + intensity overlay
├── qc_otsu_plot.py           # Otsu + PDE input documentation figures
└── tests/
```

---

## Tests

```bash
.venv/bin/python -m pytest simulation-vinesh-philip-chandan/philip-chandan/tests/ \
  simulation-vinesh-philip-chandan/philip-chandan/cohort/tests/ -q
```

41 tests cover extraction, download helpers, and cohort discovery (mocked HTTP).
