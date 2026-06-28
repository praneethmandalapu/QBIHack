"""Shared on-disk paths for the one-patient Option B integration spike."""

from __future__ import annotations

from pathlib import Path

from handoff_contract import default_grid_size, grid_size_options, spike_patient

REPO_ROOT = Path(__file__).resolve().parents[1]

# Philip-Chandan: DICOM download + QC (existing layout; do not rename mid-download)
RAW_TCIA_PHILIP_CHANDAN = REPO_ROOT / "data" / "raw" / "tcia"

# Philip-Chandan deliverable → Vinesh input
RAW_EXTRACT_PHILIP_CHANDAN = REPO_ROOT / "data" / "processed" / "raw-extract-philip-chandan"

# Expert / napari masks for PDE prep (brain-parity layout)
SEGMENTATIONS_DIR = REPO_ROOT / "data" / "processed" / "segmentations"

# Vinesh deliverable → solve_growth input
PDE_INPUT_VINESH = REPO_ROOT / "data" / "processed" / "pde-input-vinesh"

# QC artifacts
QC_SLICE_PLOTS_PHILIP_CHANDAN = REPO_ROOT / "data" / "qc" / "slice-plots-philip-chandan"
QC_OTSU_PLOTS_VINESH = REPO_ROOT / "data" / "qc" / "otsu-segmentation-vinesh"
QC_SOLVER_RUNS_VINESH = REPO_ROOT / "data" / "qc" / "solver-runs-vinesh"

SPIKE_PATIENT = spike_patient()


def slug_to_tcga_timepoint(slug: str) -> tuple[str, str]:
    """Parse cohort slug ``{subtype_slug}_{tcga_id}_{timepoint}``."""
    timepoint = slug.rsplit("_", 1)[-1]
    tcga_id = slug.rsplit("_", 1)[0].rsplit("_", 1)[-1]
    return tcga_id, timepoint


def _resolve_tcga_timepoint(
    slug: str | None = None,
    *,
    tcga_id: str | None = None,
    timepoint: str | None = None,
) -> tuple[str, str]:
    if tcga_id and timepoint:
        return tcga_id, timepoint
    name = slug or SPIKE_PATIENT["slug"]
    return slug_to_tcga_timepoint(name)


def ensure_spike_dirs() -> None:
    """Create gitignored data subfolders for the parallel spike."""
    for path in (
        RAW_EXTRACT_PHILIP_CHANDAN,
        SEGMENTATIONS_DIR,
        PDE_INPUT_VINESH,
        QC_SLICE_PLOTS_PHILIP_CHANDAN,
        QC_OTSU_PLOTS_VINESH,
        QC_SOLVER_RUNS_VINESH,
    ):
        path.mkdir(parents=True, exist_ok=True)


def raw_extract_dir(
    slug: str | None = None,
    *,
    tcga_id: str | None = None,
) -> Path:
    pid, _ = _resolve_tcga_timepoint(slug, tcga_id=tcga_id, timepoint="baseline")
    if tcga_id:
        pid = tcga_id
    return RAW_EXTRACT_PHILIP_CHANDAN / pid


def raw_extract_npy(
    slug: str | None = None,
    *,
    tcga_id: str | None = None,
    timepoint: str | None = None,
) -> Path:
    pid, tp = _resolve_tcga_timepoint(slug, tcga_id=tcga_id, timepoint=timepoint)
    return raw_extract_dir(tcga_id=pid) / f"{tp}.npy"


def raw_extract_metadata(
    slug: str | None = None,
    *,
    tcga_id: str | None = None,
    timepoint: str | None = None,
) -> Path:
    pid, tp = _resolve_tcga_timepoint(slug, tcga_id=tcga_id, timepoint=timepoint)
    return raw_extract_dir(tcga_id=pid) / f"{tp}.json"


def raw_extract_npy_legacy(slug: str | None = None) -> Path:
    """Pre-patient layout: flat slug file under raw-extract-philip-chandan/."""
    name = slug or SPIKE_PATIENT["slug"]
    return RAW_EXTRACT_PHILIP_CHANDAN / f"{name}.npy"


def raw_extract_metadata_legacy(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return RAW_EXTRACT_PHILIP_CHANDAN / f"{name}.json"


def resolve_raw_extract_npy(slug: str | None = None) -> Path:
    """Return existing raw .npy (patient layout, then legacy flat slug path)."""
    name = slug or SPIKE_PATIENT["slug"]
    primary = raw_extract_npy(name)
    if primary.is_file():
        return primary
    legacy = raw_extract_npy_legacy(name)
    if legacy.is_file():
        return legacy
    return primary


def resolve_raw_extract_metadata(slug: str | None = None) -> Path:
    """Return existing raw JSON sidecar (patient layout, then legacy flat slug path)."""
    name = slug or SPIKE_PATIENT["slug"]
    primary = raw_extract_metadata(name)
    if primary.is_file():
        return primary
    legacy = raw_extract_metadata_legacy(name)
    if legacy.is_file():
        return legacy
    return primary


def pde_input_dir(
    grid_size: int | None = None,
    *,
    tcga_id: str | None = None,
    slug: str | None = None,
) -> Path:
    size = grid_size if grid_size is not None else default_grid_size()
    pid, _ = _resolve_tcga_timepoint(slug, tcga_id=tcga_id, timepoint="baseline")
    if tcga_id:
        pid = tcga_id
    return PDE_INPUT_VINESH / pid / f"g{size}"


def pde_input_npy(
    slug: str | None = None,
    *,
    grid_size: int | None = None,
    tcga_id: str | None = None,
    timepoint: str | None = None,
) -> Path:
    pid, tp = _resolve_tcga_timepoint(slug, tcga_id=tcga_id, timepoint=timepoint)
    return pde_input_dir(grid_size, tcga_id=pid) / f"{tp}.npy"


def pde_input_metadata(
    slug: str | None = None,
    *,
    grid_size: int | None = None,
    tcga_id: str | None = None,
    timepoint: str | None = None,
) -> Path:
    pid, tp = _resolve_tcga_timepoint(slug, tcga_id=tcga_id, timepoint=timepoint)
    return pde_input_dir(grid_size, tcga_id=pid) / f"{tp}.json"


def pde_input_npy_legacy(slug: str | None = None) -> Path:
    """Pre-patient layout: flat file under pde-input-vinesh/."""
    name = slug or SPIKE_PATIENT["slug"]
    return PDE_INPUT_VINESH / f"{name}.npy"


def pde_input_metadata_legacy(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return PDE_INPUT_VINESH / f"{name}.json"


def resolve_pde_input_npy(slug: str | None = None, *, grid_size: int | None = None) -> Path:
    """Return existing PDE .npy (patient layout, then legacy flat slug path)."""
    name = slug or SPIKE_PATIENT["slug"]
    for candidate in (
        pde_input_npy(name, grid_size=grid_size),
        pde_input_npy_legacy(name),
    ):
        if candidate.is_file():
            return candidate
    return pde_input_npy(name, grid_size=grid_size)


def resolve_pde_input_metadata(slug: str | None = None, *, grid_size: int | None = None) -> Path:
    """Return existing PDE JSON sidecar (patient layout, then legacy flat slug path)."""
    name = slug or SPIKE_PATIENT["slug"]
    npy = resolve_pde_input_npy(name, grid_size=grid_size)
    return npy.with_suffix(".json")


def segmentation_mask_path(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return SEGMENTATIONS_DIR / f"{name}_mask.nii.gz"


def slice_plot_path(slug: str, *, overlay: bool = False) -> Path:
    suffix = "_mid-z-overlay.png" if overlay else "_mid-z.png"
    return QC_SLICE_PLOTS_PHILIP_CHANDAN / f"{slug}{suffix}"


def longitudinal_slice_plot_path(tcga_id: str) -> Path:
    return QC_SLICE_PLOTS_PHILIP_CHANDAN / f"{tcga_id}_longitudinal_mid-z-overlay.png"


if __name__ == "__main__":
    ensure_spike_dirs()
    for size in grid_size_options():
        pde_input_dir(size).mkdir(parents=True, exist_ok=True)
    print("Spike data folders ready:")
    for label, path in (
        ("raw DICOM (Philip-Chandan)", RAW_TCIA_PHILIP_CHANDAN),
        ("raw extract (Philip-Chandan → Vinesh)", RAW_EXTRACT_PHILIP_CHANDAN),
        ("PDE input (Vinesh → solve_growth)", PDE_INPUT_VINESH),
        ("slice QC (Philip-Chandan)", QC_SLICE_PLOTS_PHILIP_CHANDAN),
        ("Otsu segmentation QC (Vinesh)", QC_OTSU_PLOTS_VINESH),
        ("solver QC (Vinesh)", QC_SOLVER_RUNS_VINESH),
    ):
        print(f"  {label}: {path}")
