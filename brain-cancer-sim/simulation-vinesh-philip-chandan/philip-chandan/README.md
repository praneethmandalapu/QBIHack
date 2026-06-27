# Brain imaging pipeline — Philip-Chandan

Write fresh here (do not copy breast `tcia_extractor.py`).

**Start:** [`PLAN.md`](PLAN.md)

This folder owns:

- NIfTI loading for datasets in [`DATASETS.md`](../../DATASETS.md) (UCSF, MU-Glioma-Post, …)
- Expert segmentation → raw extract handoff (see [`../handoff_contract.json`](../handoff_contract.json))
- `cohort/cohort.json` with longitudinal timepoints per patient
- Longitudinal validation vs follow-up scans

Breast reference (pattern only): `breast-cancer-sim/simulation-vinesh-philip-chandan/philip-chandan/`

Suggested first spike: one UCSF Longitudinal Glioma patient, baseline MR + mask.
