"""Shared on-disk paths for the one-patient Option B integration spike."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Philip-Chandan: DICOM download + QC (existing layout; do not rename mid-download)
RAW_TCIA_PHILIP_CHANDAN = REPO_ROOT / "data" / "raw" / "tcia"

# Philip-Chandan deliverable → Vinesh input
RAW_EXTRACT_PHILIP_CHANDAN = REPO_ROOT / "data" / "processed" / "raw-extract-philip-chandan"

# Vinesh deliverable → solve_growth input
PDE_INPUT_VINESH = REPO_ROOT / "data" / "processed" / "pde-input-vinesh"

# QC artifacts
QC_SLICE_PLOTS_PHILIP_CHANDAN = REPO_ROOT / "data" / "qc" / "slice-plots-philip-chandan"
QC_SOLVER_RUNS_VINESH = REPO_ROOT / "data" / "qc" / "solver-runs-vinesh"

SPIKE_PATIENT = {
    "tcga_id": "TCGA-AR-A1AX",
    "subtype": "Luminal A",
    "study_date": "2002-09-12",
    "slug": "luminal_a_TCGA-AR-A1AX_baseline",
}


def ensure_spike_dirs() -> None:
    """Create gitignored data subfolders for the parallel spike."""
    for path in (
        RAW_EXTRACT_PHILIP_CHANDAN,
        PDE_INPUT_VINESH,
        QC_SLICE_PLOTS_PHILIP_CHANDAN,
        QC_SOLVER_RUNS_VINESH,
    ):
        path.mkdir(parents=True, exist_ok=True)


def raw_extract_npy(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return RAW_EXTRACT_PHILIP_CHANDAN / f"{name}.npy"


def raw_extract_metadata(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return RAW_EXTRACT_PHILIP_CHANDAN / f"{name}.json"


def pde_input_npy(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return PDE_INPUT_VINESH / f"{name}.npy"


def pde_input_metadata(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return PDE_INPUT_VINESH / f"{name}.json"


if __name__ == "__main__":
    ensure_spike_dirs()
    print("Spike data folders ready:")
    for label, path in (
        ("raw DICOM (Philip-Chandan)", RAW_TCIA_PHILIP_CHANDAN),
        ("raw extract (Philip-Chandan → Vinesh)", RAW_EXTRACT_PHILIP_CHANDAN),
        ("PDE input (Vinesh → solve_growth)", PDE_INPUT_VINESH),
        ("slice QC (Philip-Chandan)", QC_SLICE_PLOTS_PHILIP_CHANDAN),
        ("solver QC (Vinesh)", QC_SOLVER_RUNS_VINESH),
    ):
        print(f"  {label}: {path}")
