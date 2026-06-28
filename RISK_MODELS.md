# OncoPulse risk models — how to bring in Praneeth's models

Two risk models, **same interface**, one job: turn a patient into a `risk` score
and a `growth_multiplier` that drives the tumour-growth PDE.

| Cancer | Folder | Loader | Keyed by | Predicts |
|--------|--------|--------|----------|----------|
| **Breast** | `breast-cancer-sim/` | `oncopulse.py` | TCGA barcode (`TCGA-AR-A1AX`) | disease-specific survival risk from gene expression |
| **Brain (glioma)** | `brain-cancer-sim/` | `neuropulse.py` | UCSF SubjectID (`100118`) | mortality risk from molecular markers + grade + tumour volume |

Both are live and tracked on `main`. Both feed Vinesh's solver the same way:
`rho_eff = rho * risk_multiplier`, with `risk_multiplier = 0.8 + risk` (range
0.8–1.8). **Vinesh's solver needs zero changes between cancers** — only which
loader you import.

---

## 30-second quickstart

```python
# BREAST
import sys; sys.path.insert(0, "breast-cancer-sim")
from oncopulse import get_patient, growth_multiplier
get_patient("TCGA-AR-A1AX")          # {"risk":0.247,"pam50":"BRCA_LumA",...}
growth_multiplier("TCGA-AR-A1AX")    # 1.047

# BRAIN
import sys; sys.path.insert(0, "brain-cancer-sim")
from neuropulse import get_patient, growth_multiplier
get_patient(100118)                  # {"risk":0.614,"idh":"WT","grade":4.0,...}
growth_multiplier(100118)            # 1.414
```

> `get_patient` and `growth_multiplier` are **pure CSV lookups** — no xgboost, no
> macOS libomp fix. Only `score_expression` (breast) / `predict` (brain), which
> score a brand-new unscored patient, load the `.pkl` model.

---

## Breast model — `oncopulse.py`

- **Predicts:** disease-specific-survival risk ∈ [0,1] from **30 driver genes
  (z-scored)**. Trained on pooled METABRIC + TCGA-BRCA. CV AUC **0.76**.
- **API:** `get_patient(barcode)` → `{risk, pam50, expr(30), genes}`;
  `growth_multiplier(barcode)`; `score_expression(expr_dict)`; `GENE_LIST`.
- **Artifacts:** `models-praneeth/saved/model.pkl`,
  `data/processed/tcga_patient_features.csv` (1,082 patients scored),
  `gene_list.json`, SHAP files.
- **Full detail:** see [`breast-cancer-sim/models-praneeth/HOWTOUSEMODEL.md`](breast-cancer-sim/models-praneeth/HOWTOUSEMODEL.md).
- **Caveat:** `risk` is a relative rank, not a calibrated probability
  (`scale_pos_weight`). Per-patient genomics can diverge from PAM50 subtype (a
  feature, not a bug).

## Brain model — `neuropulse.py`

- **Predicts:** **mortality risk** ∈ [0,1] (death recorded during follow-up) from
  baseline molecular markers (IDH, MGMT, grade, 1p19q, ATRX), tumour sub-region
  volumes, age, and treatment status. Trained on UCSF-LPTDG (298 glioma
  patients). Out-of-fold CV AUC **0.79**.
- **API:** `get_patient(subject_id)` →
  `{risk, growth_multiplier, idh, grade, mgmt, diagnosis}`;
  `growth_multiplier(subject_id)`; `predict(features_dict)`; `list_patients()`.
- **Artifacts:** `models-praneeth/saved/brain_model.pkl`,
  `models-praneeth/saved/brain_metrics.json`, `brain_shap_importance.csv`,
  `data/processed/brain_patient_features.csv` (298 patients, out-of-fold scores).
- **Why mortality, not 2-month growth:** whole-tumour growth is edema-dominated
  and confounded by interim steroids/RT (CV AUC ~0.54, noise). Mortality is a
  real prognostic endpoint the molecular markers genuinely predict (~0.79), and
  it tracks aggressiveness, so higher risk → faster simulated growth.
- **Clean biology (no inversion):** mean risk IDH-WT **0.67** vs IDH-mut **0.21**.
  Primary pair: `100118` (IDH-WT GBM g4) risk 0.61 → mult 1.41; `100192`
  (IDH-mut astro g2) risk 0.14 → mult 0.94.
- **Caveat:** `risk` is a relative prognostic rank; "death recorded" partly
  depends on follow-up length — read it as aggressiveness ordering, not absolute
  survival probability.

---

## Bringing it into the simulation (Vinesh)

The seam is one scalar. In `tumor_pde_solver.py`, `risk_multiplier` scales `rho`.
Wire it per patient at call time — pick the loader for the cancer you're running:

```python
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]   # the *-cancer-sim/ root
sys.path.insert(0, str(ROOT))
from neuropulse import growth_multiplier            # or: from oncopulse import ...

params = {**DEFAULT_PARAMS, "risk_multiplier": growth_multiplier(patient_id)}
frames = solve_growth(baseline_volume, timesteps=50, dt=0.1, params=params)
```

For the **predict-then-validate** loop, drive growth with the genomic/molecular
multiplier (NOT calibrated to the follow-up), simulate to the real interval, and
compare against the held-out follow-up mask — see
`vinesh/calibrate.py::validate_growth`.

## Bringing it into the app (Vinesh/Philip)

- **Risk / Predict tab:** `get_patient(id)` for the score + molecular fields.
- **Explain tab:** read `brain_shap_importance.csv` / breast `shap_importance.csv`.
- **Cohort tab (brain):** `data/processed/brain_patient_features.csv` and
  `ucsf_longitudinal_master.csv` power a 298-patient browser.

## Dependency map (so you don't fight libomp)

| Call | Loads model.pkl? | Needs xgboost + libomp? |
|------|------------------|--------------------------|
| `get_patient`, `growth_multiplier`, `list_patients` | no | **no** |
| `score_expression` (breast) / `predict` (brain) | yes | yes |

xgboost on macOS needs `libomp.dylib`; both `models-praneeth/` folders ship
`_macos_omp_fix.py` (import it before xgboost). Lookups never need it. The
re-exec in that shim breaks heredoc/stdin — run training from a real file.

## Regenerating

```bash
# breast
cd breast-cancer-sim/models-praneeth
python download_data.py && python build_features.py && python train_xgboost.py
python generate_shap.py && python build_patient_table.py

# brain (extract the UCSF clinical xlsx into data/raw/ucsf_glioma/ first)
cd brain-cancer-sim/models-praneeth
python clean_ucsf.py && python train_brain_risk.py
```

Questions a doc can't answer → ping Praneeth.
