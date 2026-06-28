# Genomics / risk models — brain (Praneeth)

Brain analog of `breast-cancer-sim/models-praneeth/`. Dataset: **UCSF-LPTDG**
(UCSF Longitudinal Postoperative Diffuse Glioma) — 298 patients, two timepoints
each with BraTS-style segmentation masks, plus molecular markers (IDH / MGMT /
1p19q / grade) and measured tumour growth.

## What's here now

- `clean_ucsf.py` — standard data-cleaning pass over the UCSF clinical workbook.
- `data/processed/ucsf_longitudinal_master.csv` — **the master table**: one row
  per patient with t1-vs-t2 sub-region volumes, measured growth %, and the full
  molecular/clinical fields. This is the feature source for the risk model.
- `data/processed/ucsf_clinical_clean.csv`, `ucsf_imaging_long_clean.csv` — tidy
  per-patient and per-timepoint views.

Cohort (primary pair) lives in
`simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort.json`.

## The clean signal (why brain is easier than breast)

Unlike breast (PAM50 is a weak per-patient predictor), in glioma the molecular
markers dominate prognosis. In this data the contrast is clean, no inversion:

- IDH-WT glioblastoma (grade 4): whole-tumour **+170% to +609%** over ~2 months.
- IDH-mut grade 2-3 glioma: **stable or shrinking** (-20% to -42%).

## TODO — the risk model (`oncopulse`-equivalent)

Build a per-patient `risk -> growth_multiplier` the same way as breast:

- **Route B (recommended):** tabular model / transparent rule on UCSF molecular
  features (IDH, grade, MGMT, baseline volume) -> growth_multiplier. Airtight
  imaging join (every imaging patient has these fields).
- **Route A (stretch):** XGBoost on TCGA-GBM/LGG expression (reuse the breast
  `download_data.py`/`build_features.py`/`train_xgboost.py`), predict IDH/grade/
  survival, get gene-level SHAP; join to imaging via the IDH/grade label.

Expose it through a `neuropulse.py` loader mirroring `breast-cancer-sim/oncopulse.py`
(`get_patient(id) -> {risk, idh, grade}`, `growth_multiplier(id)`).

**Validation cohort:** add **CGGA** (Chinese Glioma Genome Atlas) as the METABRIC-like
holdout — train on TCGA-GBM / TCGA-LGG (GDC or cBioPortal), evaluate generalization
on CGGA. See repo-level TODO in [`../README.md`](../README.md).
