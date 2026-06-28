# Genomics / risk models — brain (Praneeth)

Brain analog of `breast-cancer-sim/models-praneeth/`. Dataset: **UCSF-LPTDG**
(UCSF Longitudinal Postoperative Diffuse Glioma) — 298 patients, two timepoints
each with BraTS-style segmentation masks, plus molecular markers (IDH / MGMT /
1p19q / grade) and measured tumour growth.

## What's here now

- `clean_ucsf.py` — standard data-cleaning pass over the UCSF clinical workbook.
- `train_brain_risk.py` — the risk-score model (below).
- `data/processed/ucsf_longitudinal_master.csv` — **the master table**: one row
  per patient with t1-vs-t2 sub-region volumes, measured growth %, and the full
  molecular/clinical fields. Feature source for the risk model.
- `data/processed/ucsf_clinical_clean.csv`, `ucsf_imaging_long_clean.csv` — tidy
  per-patient and per-timepoint views.
- `data/processed/brain_patient_features.csv` — 298 patients scored (risk + growth_multiplier).

Cohort (primary pair) lives in
`simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort.json`.

## The clean signal (why brain is easier than breast)

Unlike breast (PAM50 is a weak per-patient predictor), in glioma the molecular
markers dominate prognosis. In this data the contrast is clean, no inversion:

- IDH-WT glioblastoma (grade 4): whole-tumour **+170% to +609%** over ~2 months.
- IDH-mut grade 2-3 glioma: **stable or shrinking** (-20% to -42%).

## The risk model — BUILT (`train_brain_risk.py`)

XGBoost predicting **mortality risk** (death recorded during follow-up) from
baseline molecular markers + grade + tumour volumes + treatment status.
Out-of-fold **CV AUC 0.79**. Clean biology: mean risk IDH-WT 0.67 vs IDH-mut 0.21.

Target rationale: 2-month tumour growth is edema/steroid-confounded (AUC ~0.54,
noise); mortality is a real prognostic endpoint and tracks aggressiveness, so
higher risk -> faster simulated growth.

Artifacts:
- `saved/brain_model.pkl`, `saved/brain_metrics.json`, `saved/brain_shap_importance.csv`
- `../data/processed/brain_patient_features.csv` — 298 patients: risk + growth_multiplier

Loader: **`brain-cancer-sim/neuropulse.py`** mirrors `oncopulse.py` —
`get_patient(id)`, `growth_multiplier(id)`, `predict(features)`.

**How teammates bring this in:** see the repo-root [`RISK_MODELS.md`](../../RISK_MODELS.md).

## Stretch — expression model + external validation

Add a TCGA-GBM/LGG **expression** XGBoost + SHAP for a gene-level "why" panel
(reuse the breast `download_data.py`/`build_features.py`/`train_xgboost.py`).
Use **CGGA** (Chinese Glioma Genome Atlas) as the METABRIC-like holdout: train on
TCGA-GBM / TCGA-LGG (GDC or cBioPortal), evaluate generalization on CGGA. See the
repo-level TODO in [`../README.md`](../README.md).
