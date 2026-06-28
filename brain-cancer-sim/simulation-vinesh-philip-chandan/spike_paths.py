"""Shared on-disk paths for the brain imaging integration spike."""

from __future__ import annotations

from pathlib import Path

from handoff_contract import default_grid_size, grid_size_options, spike_patient

REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_EXTRACT_PHILIP_CHANDAN = REPO_ROOT / "data" / "processed" / "raw-extract-philip-chandan"
SEGMENTATIONS_DIR = REPO_ROOT / "data" / "processed" / "segmentations"
PDE_INPUT_VINESH = REPO_ROOT / "data" / "processed" / "pde-input-vinesh"
QC_SLICE_PLOTS_PHILIP_CHANDAN = REPO_ROOT / "data" / "qc" / "slice-plots-philip-chandan"
QC_PDE_PREP_VINESH = REPO_ROOT / "data" / "qc" / "pde-prep-vinesh"

SPIKE_PATIENT = spike_patient()


def ensure_spike_dirs() -> None:
    for path in (
        RAW_EXTRACT_PHILIP_CHANDAN,
        SEGMENTATIONS_DIR,
        PDE_INPUT_VINESH,
        QC_SLICE_PLOTS_PHILIP_CHANDAN,
        QC_PDE_PREP_VINESH,
    ):
        path.mkdir(parents=True, exist_ok=True)
    for size in grid_size_options():
        pde_input_dir(size).mkdir(parents=True, exist_ok=True)
        qc_pde_prep_dir(size).mkdir(parents=True, exist_ok=True)


def raw_extract_npy(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return RAW_EXTRACT_PHILIP_CHANDAN / f"{name}.npy"


def raw_extract_metadata(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return RAW_EXTRACT_PHILIP_CHANDAN / f"{name}.json"


def segmentation_mask_path(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return SEGMENTATIONS_DIR / f"{name}_mask.nii.gz"


def pde_input_dir(grid_size: int | None = None) -> Path:
    size = grid_size if grid_size is not None else default_grid_size()
    return PDE_INPUT_VINESH / f"g{size}"


def pde_input_npy(slug: str | None = None, *, grid_size: int | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return pde_input_dir(grid_size) / f"{name}.npy"


def pde_input_metadata(slug: str | None = None, *, grid_size: int | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return pde_input_dir(grid_size) / f"{name}.json"


def pde_input_npy_legacy(slug: str | None = None) -> Path:
    """Pre-g64 layout: flat file under pde-input-vinesh/ (read fallback only)."""
    name = slug or SPIKE_PATIENT["slug"]
    return PDE_INPUT_VINESH / f"{name}.npy"


def qc_pde_prep_dir(grid_size: int | None = None) -> Path:
    size = grid_size if grid_size is not None else default_grid_size()
    return QC_PDE_PREP_VINESH / f"g{size}"


def slice_plot_path(slug: str, *, overlay: bool = False) -> Path:
    suffix = "_mid-z-overlay.png" if overlay else "_mid-z.png"
    return QC_SLICE_PLOTS_PHILIP_CHANDAN / f"{slug}{suffix}"
