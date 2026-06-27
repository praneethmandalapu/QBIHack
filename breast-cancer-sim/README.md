# Breast Cancer Simulation (QBIHack)

Multi-domain pipeline for breast cancer risk prediction, explainability, tumor growth simulation, and 3D visualization.

## Structure

| Directory | Owner | Purpose |
|-----------|-------|---------|
| `data/` | — | Raw METABRIC/TCIA inputs and processed features (gitignored) |
| `models-praneeth/` | Praneeth | Genomics & risk ML (XGBoost, SHAP) |
| `simulation-vinesh-philip-chandan/` | Vinesh, Philip, Chandan | Tumor growth simulation (see subfolders below) |
| `simulation-vinesh-philip-chandan/vinesh/` | Vinesh | PDE tumor growth engine and drug interventions |
| `simulation-vinesh-philip-chandan/philip-chandan/` | Philip, Chandan | TCIA DICOM → 3D volume extraction |
| `visualization-jasim/` | Jasim | 3D rendering and tissue color maps |
| `app-vihari/` | Vihari | Streamlit frontend and LLM narrative |

### Simulation layout

```
simulation-vinesh-philip-chandan/
├── vinesh/
│   ├── tumor_pde_solver.py      # PDE growth engine
│   └── growth_interventions.py  # Drug logic on volume
└── philip-chandan/
    └── tcia_extractor.py        # DICOM → 3D volume
```

## Setup

```bash
cd breast-cancer-sim
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place METABRIC CSVs in `data/raw/` and TCIA DICOM series in `data/raw/`. Processed arrays go in `data/processed/`.

## Run the app

```bash
streamlit run app-vihari/app.py
```
