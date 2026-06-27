# Brain imaging pipeline — write fresh here (do not copy breast tcia_extractor).

This folder owns:

- NIfTI / DICOM loading for datasets in `DATASETS.md` (UCSF, MU-Glioma-Post, …)
- Expert segmentation → PDE initial condition (see `handoff_contract.json`)
- `cohort.json` with longitudinal timepoints per patient
- Longitudinal validation vs follow-up scans

Breast reference (pattern only): `breast-cancer-sim/simulation-vinesh-philip-chandan/philip-chandan/`

Suggested first spike: one UCSF Longitudinal Glioma patient, baseline MR + mask.
