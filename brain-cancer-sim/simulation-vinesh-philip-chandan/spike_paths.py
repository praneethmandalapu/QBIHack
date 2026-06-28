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


def slug_to_patient_timepoint(slug: str) -> tuple[str, str]:
    """Parse cohort slug ``{disease}_{dataset}_{patient_id}_{timepoint}``."""
    timepoint = slug.rsplit("_", 1)[-1]
    patient_id = slug.rsplit("_", 1)[0].rsplit("_", 1)[-1]
    return patient_id, timepoint


def _resolve_patient_timepoint(
    slug: str | None = None,
    *,
    patient_id: str | None = None,
    timepoint: str | None = None,
) -> tuple[str, str]:
    if patient_id and timepoint:
        return patient_id, timepoint
    name = slug or SPIKE_PATIENT["slug"]
    return slug_to_patient_timepoint(name)


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
        qc_pde_prep_dir(size).mkdir(parents=True, exist_ok=True)


def raw_extract_dir(
    slug: str | None = None,
    *,
    patient_id: str | None = None,
) -> Path:
    pid, _ = _resolve_patient_timepoint(slug, patient_id=patient_id, timepoint="baseline")
    if patient_id:
        pid = patient_id
    return RAW_EXTRACT_PHILIP_CHANDAN / pid


def raw_extract_npy(
    slug: str | None = None,
    *,
    patient_id: str | None = None,
    timepoint: str | None = None,
) -> Path:
    pid, tp = _resolve_patient_timepoint(slug, patient_id=patient_id, timepoint=timepoint)
    return raw_extract_dir(patient_id=pid) / f"{tp}.npy"


def raw_extract_metadata(
    slug: str | None = None,
    *,
    patient_id: str | None = None,
    timepoint: str | None = None,
) -> Path:
    pid, tp = _resolve_patient_timepoint(slug, patient_id=patient_id, timepoint=timepoint)
    return raw_extract_dir(patient_id=pid) / f"{tp}.json"


def raw_extract_npy_legacy(slug: str | None = None) -> Path:
    """Pre-patient-id layout: flat slug file under raw-extract-philip-chandan/."""
    name = slug or SPIKE_PATIENT["slug"]
    return RAW_EXTRACT_PHILIP_CHANDAN / f"{name}.npy"


def raw_extract_metadata_legacy(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return RAW_EXTRACT_PHILIP_CHANDAN / f"{name}.json"


def resolve_raw_extract_npy(slug: str | None = None) -> Path:
    """Return existing raw .npy (patient-id layout, then legacy flat slug path)."""
    name = slug or SPIKE_PATIENT["slug"]
    primary = raw_extract_npy(name)
    if primary.is_file():
        return primary
    legacy = raw_extract_npy_legacy(name)
    if legacy.is_file():
        return legacy
    return primary


def resolve_raw_extract_metadata(slug: str | None = None) -> Path:
    """Return existing raw JSON sidecar (patient-id layout, then legacy flat slug path)."""
    name = slug or SPIKE_PATIENT["slug"]
    primary = raw_extract_metadata(name)
    if primary.is_file():
        return primary
    legacy = raw_extract_metadata_legacy(name)
    if legacy.is_file():
        return legacy
    return primary


def segmentation_mask_path(slug: str | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    return SEGMENTATIONS_DIR / f"{name}_mask.nii.gz"


def pde_input_dir(
    grid_size: int | None = None,
    *,
    patient_id: str | None = None,
    slug: str | None = None,
) -> Path:
    size = grid_size if grid_size is not None else default_grid_size()
    pid, _ = _resolve_patient_timepoint(slug, patient_id=patient_id, timepoint="baseline")
    if patient_id:
        pid = patient_id
    return PDE_INPUT_VINESH / pid / f"g{size}"


def pde_input_npy(
    slug: str | None = None,
    *,
    grid_size: int | None = None,
    patient_id: str | None = None,
    timepoint: str | None = None,
) -> Path:
    pid, tp = _resolve_patient_timepoint(slug, patient_id=patient_id, timepoint=timepoint)
    return pde_input_dir(grid_size, patient_id=pid) / f"{tp}.npy"


def pde_input_metadata(
    slug: str | None = None,
    *,
    grid_size: int | None = None,
    patient_id: str | None = None,
    timepoint: str | None = None,
) -> Path:
    pid, tp = _resolve_patient_timepoint(slug, patient_id=patient_id, timepoint=timepoint)
    return pde_input_dir(grid_size, patient_id=pid) / f"{tp}.json"


def pde_input_npy_legacy_grid(slug: str | None = None, *, grid_size: int | None = None) -> Path:
    """Pre-patient-id layout: ``pde-input-vinesh/g{size}/{slug}.npy``."""
    name = slug or SPIKE_PATIENT["slug"]
    size = grid_size if grid_size is not None else default_grid_size()
    return PDE_INPUT_VINESH / f"g{size}" / f"{name}.npy"


def pde_input_metadata_legacy_grid(slug: str | None = None, *, grid_size: int | None = None) -> Path:
    name = slug or SPIKE_PATIENT["slug"]
    size = grid_size if grid_size is not None else default_grid_size()
    return PDE_INPUT_VINESH / f"g{size}" / f"{name}.json"


def pde_input_npy_legacy(slug: str | None = None) -> Path:
    """Pre-g64 layout: flat file under pde-input-vinesh/ (read fallback only)."""
    name = slug or SPIKE_PATIENT["slug"]
    return PDE_INPUT_VINESH / f"{name}.npy"


def resolve_pde_input_npy(slug: str | None = None, *, grid_size: int | None = None) -> Path:
    """Return existing PDE .npy (patient-id layout, then legacy grid, then flat)."""
    name = slug or SPIKE_PATIENT["slug"]
    for candidate in (
        pde_input_npy(name, grid_size=grid_size),
        pde_input_npy_legacy_grid(name, grid_size=grid_size),
        pde_input_npy_legacy(name),
    ):
        if candidate.is_file():
            return candidate
    return pde_input_npy(name, grid_size=grid_size)


def resolve_pde_input_metadata(slug: str | None = None, *, grid_size: int | None = None) -> Path:
    """Return existing PDE JSON sidecar (patient-id layout, then legacy paths)."""
    name = slug or SPIKE_PATIENT["slug"]
    npy = resolve_pde_input_npy(name, grid_size=grid_size)
    return npy.with_suffix(".json")


def qc_pde_prep_dir(grid_size: int | None = None) -> Path:
    size = grid_size if grid_size is not None else default_grid_size()
    return QC_PDE_PREP_VINESH / f"g{size}"


def slice_plot_path(slug: str, *, overlay: bool = False) -> Path:
    suffix = "_mid-z-overlay.png" if overlay else "_mid-z.png"
    return QC_SLICE_PLOTS_PHILIP_CHANDAN / f"{slug}{suffix}"
