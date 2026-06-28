# Brain Cancer Simulation (QBIHack)

Longitudinal glioma / brain tumor growth simulation. Layout mirrors [`breast-cancer-sim`](../breast-cancer-sim/): team-owned folders, shared handoff contract, gitignored `data/raw/`.

## Structure

| Directory | Owner | Purpose |
|-----------|-------|---------|
| `DATASETS.md` | shared | Candidate longitudinal MRI datasets (UCSF, MU-Glioma-Post, …) |
| `data/` | shared | Raw downloads + processed arrays (`data/raw/` gitignored) |
| `simulation-vinesh-philip-chandan/` | Philip-Chandan + Vinesh | Imaging pipeline + PDE solver + handoff contract |
| `simulation-vinesh-philip-chandan/philip-chandan/` | Philip-Chandan | NIfTI extract, masks, cohort — see `PLAN.md` |
| `simulation-vinesh-philip-chandan/vinesh/` | Vinesh | PDE growth engine, `prepare_pde_input.py` *(stub)* |
| `models-praneeth/` | Praneeth | Genomics / risk models *(stub)* |
| `visualization-jasim/` | Jasim | Plotly 3D rendering |
| `app/` | Vinesh/Philip | Streamlit shell (tab stubs) |

## Setup

```bash
cd brain-cancer-sim
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick smoke test

```bash
# PDE solver
python simulation-vinesh-philip-chandan/vinesh/test_solver.py

# End-to-end via contract wrapper
python simulation-vinesh-philip-chandan/vinesh/run_growth.py

# 3D preview (writes tumor_preview.html)
python visualization-jasim/render_3d.py

# Interactive MR + segmentation (napari — demo works without data)
python simulation-vinesh-philip-chandan/philip-chandan/view_volume_napari.py --demo
```

## Run the app

```bash
streamlit run app/app.py
```

## Relationship to breast-cancer-sim

| Copied (stable) | Written fresh |
|-----------------|---------------|
| `tumor_pde_solver.py`, `growth_interventions.py` | `philip-chandan/` — NIfTI, expert masks |
| `handoff_contract.py` pattern | `handoff_contract.json` — brain segmentation spec |
| `render_3d.py`, `color_maps.py` | Cohort, genomics, longitudinal validation |
| Streamlit tab shell | Dataset download scripts |

Do **not** import from `../breast-cancer-sim/` at runtime — each project is self-contained.

## TODO

| Item | Owner | Notes |
|------|-------|-------|
| **CGGA** (Chinese Glioma Genome Atlas) | Praneeth | METABRIC-like external validation for glioma genomics — train on **TCGA-GBM / TCGA-LGG**, hold out **CGGA** for cross-cohort checks (mirrors breast METABRIC + TCGA-BRCA). No TCIA imaging link; pairs with imaging cohort picks via molecular labels (IDH, grade, MGMT), not shared patient IDs. |
| Imaging spike (UCSF-ALPTDG, MU-Glioma-Post, or LUMIERE) | Philip-Chandan | Longitudinal MRI + masks — see [`DATASETS.md`](DATASETS.md), [`philip-chandan/PLAN.md`](simulation-vinesh-philip-chandan/philip-chandan/PLAN.md) |
| Genomics risk model | Praneeth | Patient table, features, XGBoost/SHAP — see [`models-praneeth/README.md`](models-praneeth/README.md) |
| PDE input prep | Vinesh | `prepare_pde_input.py` — resample expert mask → solver grid |
