# Philip-Chandan Brain Imaging Pipeline — Report

QBIHack brain-cancer-sim · UCSF longitudinal glioma Phase 0 spike through raw export, expert-mask QC, Vinesh PDE prep, and clinical napari viewer.

**PDF:** [`PIPELINE_REPORT.pdf`](PIPELINE_REPORT.pdf) (regenerate with `generate_pipeline_report.py`)  
**Handoff contract:** [`../handoff_contract.json`](../handoff_contract.json) v1.0.0  
**Cohort:** [`cohort/cohort.json`](cohort/cohort.json) rev1-ucsf-spike

---

## Scope

Philip-Chandan owns **Person 5: Imaging Pipeline** in this folder. Phase 0 delivered:

1. **Cohort spike** — UCSF-ALPTDG patient `100002`, baseline T1ce + expert mask  
2. **NIfTI extract** — `nifti_extractor.py` → `(Z, Y, X)` float32  
3. **Export** — raw `.npy` + JSON sidecar + copied mask in `segmentations/`  
4. **Visual QC** — `qc_slice_plot.py` mid-Z overlay PNGs  
5. **Clinical napari viewer** — WW/WL, FLAIR/T2 layers, brain-masked defaults, MPR grid, optional CLAHE  
6. **Vinesh PDE prep** — `../vinesh/prepare_pde_input.py` → `pde-input-vinesh/` (expert mask, no Otsu)  
7. **Solver smoke test** — `solve_growth()` on spike slug  

**Next (Phase 1):** `manifest.json` v1.0.0, Vinesh `calibrate.py`, demo toggle (`100002` vs `100118`).

---

## Spike patient

| Field | Value |
|-------|-------|
| Dataset | UCSF Longitudinal Glioma (ALPTDG) |
| Patient | `100002` |
| Diagnosis | Oligodendroglioma, WHO grade 2, IDH-mut, MGMT+ |
| Slug | `glioma_ucsf_100002_baseline` |
| Raw shape | `(155, 240, 240)` @ 1 mm isotropic |
| PDE input | `(64, 64, 64)` @ 1 mm |

Grid size is configured in `handoff_contract.json` (`grid_size_options`: 64). Outputs land in `pde-input-vinesh/<patient_id>/g64/`.

Expert segmentations are dataset ground truth — **no Otsu fallback** in brain v1.

---

## Regenerate PDF

```bash
cd brain-cancer-sim
.venv/bin/pip install -r requirements.txt   # includes fpdf2
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/generate_pipeline_report.py
```

---

## PDE prep (Vinesh)

`prepare_pde_input.py` resamples MR + expert mask to 1 mm, normalizes to `[0, 1]`, zeros background, crops to `64³`. `qc_pde_plot.py` writes documentation PNGs under `data/qc/pde-prep-vinesh/`.

Full pipeline narrative and figures are in **Section 6** of the PDF.

---

## Napari viewer (summary)

Full glossary of T1 / T2 / FLAIR / T1ce / CLAHE / WW/WL is in **Section 4–5** of the PDF.

```bash
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/view_volume_napari.py --slug glioma_ucsf_100002_baseline --pde-input
```

See also [`VALIDATION.md`](VALIDATION.md).

---

## File map

```
philip-chandan/
├── generate_pipeline_report.py   # → PIPELINE_REPORT.pdf
├── report.md                     # this file
├── view_volume_napari.py         # clinical QC viewer
├── nifti_extractor.py
├── export_raw_extract.py
├── qc_slice_plot.py
├── qc_pde_plot.py                # PDE prep QC figures
└── cohort/

../vinesh/
├── prepare_pde_input.py          # expert mask → PDE grid
└── run_growth.py
```
