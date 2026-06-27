# Brain Cancer Simulation (QBIHack)

Longitudinal glioma / brain tumor growth simulation. Forked from the breast-cancer-sim
modular layout: **copy** disease-agnostic engine + viz; **write fresh** imaging pipeline
(expert segmentations, NIfTI loaders — see `simulation/imaging/`).

## Structure

| Directory | Purpose |
|-----------|---------|
| `DATASETS.md` | Candidate longitudinal MRI datasets (UCSF, MU-Glioma-Post, …) |
| `data/` | Raw downloads + processed arrays (gitignored under `data/raw/`) |
| `simulation/solver/` | PDE growth engine + drug interventions (ported from breast) |
| `simulation/imaging/` | **Fresh code** — volume extraction, masks, cohort (not copied) |
| `simulation/handoff_contract.json` | Versioned array contract between imaging ↔ solver |
| `visualization/` | Plotly 3D rendering (ported from breast) |
| `app/` | Streamlit shell (tab stubs; wire in Phase 3) |

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
python simulation/solver/test_solver.py

# End-to-end via contract wrapper
python simulation/run_growth.py

# 3D preview (writes tumor_preview.html)
python visualization/render_3d.py
```

## Run the app

```bash
streamlit run app/app.py
```

## Relationship to breast-cancer-sim

| Copied (stable) | Written fresh |
|-----------------|---------------|
| `tumor_pde_solver.py`, `growth_interventions.py` | `simulation/imaging/` — NIfTI, expert masks |
| `handoff_contract.py` pattern | `handoff_contract.json` — brain segmentation spec |
| `render_3d.py`, `color_maps.py` | Cohort, genomics, longitudinal validation |
| Streamlit tab shell | Dataset download scripts |

Do **not** import from `../breast-cancer-sim/` at runtime — each project is self-contained.
