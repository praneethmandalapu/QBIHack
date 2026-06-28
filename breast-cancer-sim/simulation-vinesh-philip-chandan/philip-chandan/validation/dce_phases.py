"""DCE phase splitting and pre-contrast loading for breast MRI napari QC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk

from download_tcia import download_series_to_dir, group_series_by_study, list_mr_series
from paths import VALIDATION_DICOM_DIR
from tcia_extractor import extract_volume_with_spacing, iter_dicom_files
from validate_segmentation import pick_dce_series


@dataclass(frozen=True)
class DcePhase:
    """One temporal phase inside a stacked DCE series (Z indices are half-open)."""

    index: int
    z_start: int
    z_end: int
    acquisition_time: str

    @property
    def n_slices(self) -> int:
        return self.z_end - self.z_start


def _sort_dicom_paths(dicom_dir: Path) -> list[Path]:
    paths = [path for path in iter_dicom_files(dicom_dir) if path.suffix.lower() == ".dcm"]
    if not paths:
        return []

    def sort_key(path: Path) -> tuple[int, float]:
        dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        instance = int(getattr(dataset, "InstanceNumber", 0) or 0)
        position = getattr(dataset, "ImagePositionPatient", None)
        z_pos = float(position[2]) if position is not None and len(position) >= 3 else 0.0
        return instance, z_pos

    return sorted(paths, key=sort_key)


def phase_ranges_from_dicom(dicom_dir: Path) -> list[DcePhase]:
    """Group DICOM slices by AcquisitionTime in read order."""
    paths = _sort_dicom_paths(dicom_dir)
    if not paths:
        return []

    groups: list[tuple[str, int, int]] = []
    current_acq: str | None = None
    start = 0
    for index, path in enumerate(paths):
        acq = str(getattr(pydicom.dcmread(path, stop_before_pixels=True, force=True), "AcquisitionTime", "") or "")
        if current_acq is None:
            current_acq = acq
            start = 0
            continue
        if acq != current_acq:
            groups.append((current_acq, start, index))
            current_acq = acq
            start = index
    groups.append((current_acq or "", start, len(paths)))

    return [
        DcePhase(index=phase_index, z_start=z_start, z_end=z_end, acquisition_time=acq)
        for phase_index, (acq, z_start, z_end) in enumerate(groups, start=1)
    ]


def infer_equal_phase_ranges(n_slices: int, n_phases: int = 4) -> list[DcePhase]:
    """Fallback when DICOM metadata is unavailable."""
    if n_slices % n_phases != 0:
        return [DcePhase(index=1, z_start=0, z_end=n_slices, acquisition_time="")]
    phase_size = n_slices // n_phases
    return [
        DcePhase(
            index=phase_index,
            z_start=(phase_index - 1) * phase_size,
            z_end=phase_index * phase_size,
            acquisition_time="",
        )
        for phase_index in range(1, n_phases + 1)
    ]


def resolve_dce_dicom_dir_for_study(
    *,
    tcga_id: str,
    study_date: str,
    dce_index: int,
    source_dicom_dir: Path | None = None,
) -> Path | None:
    """Prefer validation cache, then exported raw DICOM study folder."""
    series_list = list_mr_series(tcga_id)
    grouped = group_series_by_study(series_list)
    if study_date not in grouped:
        if source_dicom_dir and source_dicom_dir.is_dir() and any(source_dicom_dir.rglob("*.dcm")):
            return source_dicom_dir
        return None

    target_series = pick_dce_series(grouped[study_date], dce_index)
    series_uid = str(target_series["SeriesInstanceUID"])
    cache_dir = VALIDATION_DICOM_DIR / tcga_id / study_date / series_uid
    if any(cache_dir.rglob("*")):
        return cache_dir

    if source_dicom_dir and source_dicom_dir.is_dir() and any(source_dicom_dir.rglob("*.dcm")):
        return source_dicom_dir

    return None


def resolve_phase_ranges(
    *,
    volume_shape: tuple[int, ...],
    dicom_dir: Path | None,
    n_phases: int = 4,
) -> list[DcePhase]:
    """Return temporal phase Z ranges for a stacked DCE volume."""
    if dicom_dir is not None and dicom_dir.is_dir():
        phases = phase_ranges_from_dicom(dicom_dir)
        if phases and phases[-1].z_end == volume_shape[0]:
            return phases
    return infer_equal_phase_ranges(volume_shape[0], n_phases=n_phases)


def split_dce_phases(volume: np.ndarray, phases: list[DcePhase]) -> list[np.ndarray]:
    """Split a stacked DCE volume into per-phase 3D arrays."""
    return [volume[phase.z_start : phase.z_end] for phase in phases]


def select_phases(
    phases: list[DcePhase],
    phase_volumes: list[np.ndarray],
    indices: tuple[int, ...],
) -> tuple[list[DcePhase], list[np.ndarray]]:
    """Keep only ``indices`` (e.g. P1–P3); preserve ascending phase order."""
    wanted = set(indices)
    selected_phases: list[DcePhase] = []
    selected_volumes: list[np.ndarray] = []
    for phase, volume in zip(phases, phase_volumes, strict=True):
        if phase.index in wanted:
            selected_phases.append(phase)
            selected_volumes.append(volume)
    missing = wanted - {phase.index for phase in selected_phases}
    if missing:
        have = [phase.index for phase in phases]
        raise ValueError(f"Requested DCE phases {sorted(missing)} not in series (have {have})")
    pairs = sorted(zip(selected_phases, selected_volumes), key=lambda item: item[0].index)
    return [phase for phase, _ in pairs], [volume for _, volume in pairs]


def resample_volume(
    volume: np.ndarray,
    spacing_mm: list[float],
    target_shape: tuple[int, int, int],
    target_spacing: list[float],
) -> np.ndarray:
    """Resample (Z, Y, X) float32 volume onto a target grid."""
    moving = sitk.GetImageFromArray(volume.astype(np.float32))
    dz, dy, dx = (float(s) for s in spacing_mm)
    moving.SetSpacing((dx, dy, dz))

    tdz, tdy, tdx = (float(s) for s in target_spacing)
    tz, ty, tx = (int(s) for s in target_shape)
    reference = sitk.GetImageFromArray(np.zeros(target_shape, dtype=np.float32))
    reference.SetSpacing((tdx, tdy, tdz))

    resampled = sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )
    return sitk.GetArrayFromImage(resampled).astype(np.float32)


def _validation_series_dir(tcga_id: str, study_date: str, series_uid: str) -> Path:
    return VALIDATION_DICOM_DIR / tcga_id / study_date / series_uid


def load_dce_series_volume(
    *,
    tcga_id: str,
    study_date: str,
    dce_index: int,
) -> tuple[np.ndarray, list[float], str, Path]:
    """Download (if needed) and load one DCE series by 1-based index."""
    series_list = list_mr_series(tcga_id)
    grouped = group_series_by_study(series_list)
    if study_date not in grouped:
        raise FileNotFoundError(f"No MR study on {study_date} for {tcga_id}")

    target_series = pick_dce_series(grouped[study_date], dce_index)
    series_description = str(target_series.get("SeriesDescription", ""))
    cache_dir = _validation_series_dir(
        tcga_id,
        study_date,
        str(target_series["SeriesInstanceUID"]),
    )
    if not any(cache_dir.rglob("*")):
        download_series_to_dir(target_series, cache_dir)
    volume, spacing = extract_volume_with_spacing(cache_dir)
    return volume, spacing, series_description, cache_dir


def load_precontrast_volume(
    *,
    tcga_id: str,
    study_date: str,
) -> tuple[np.ndarray, list[float], str] | None:
    """Load S1 pre-contrast DCE series (typically Ax T1)."""
    try:
        volume, spacing, series_description, _ = load_dce_series_volume(
            tcga_id=tcga_id,
            study_date=study_date,
            dce_index=1,
        )
    except (FileNotFoundError, ValueError):
        return None
    return volume, spacing, series_description


def compute_subtraction(post: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """Raw subtraction (post - pre); caller normalizes for display."""
    if post.shape != pre.shape:
        raise ValueError(f"Subtraction shape mismatch: post {post.shape} vs pre {pre.shape}")
    return (post.astype(np.float32) - pre.astype(np.float32)).astype(np.float32)


def lesion_z_in_phase(expert_mask: np.ndarray, phase: DcePhase) -> int | None:
    """Best Z index within one phase volume for navigating to the expert ROI."""
    phase_mask = expert_mask[phase.z_start : phase.z_end]
    if not phase_mask.any():
        return None
    return int(phase_mask.sum(axis=(1, 2)).argmax())


def mask_for_phase(expert_mask: np.ndarray, phase: DcePhase) -> np.ndarray:
    """Crop a full-stack expert mask to one temporal phase."""
    return expert_mask[phase.z_start : phase.z_end].astype(np.uint8)


def mip_along_z(volume: np.ndarray) -> np.ndarray:
    """Maximum intensity projection along Z → (Y, X)."""
    return volume.max(axis=0).astype(np.float32)


def mip_as_volume(volume: np.ndarray) -> np.ndarray:
    """Return MIP as (1, Y, X) for napari."""
    return mip_along_z(volume)[np.newaxis, ...]


def expert_centroid_zyx(expert_mask: np.ndarray, phase: DcePhase) -> tuple[float, float, float] | None:
    """Centroid of the expert ROI within one phase volume."""
    phase_mask = mask_for_phase(expert_mask, phase)
    coords = np.argwhere(phase_mask > 0)
    if coords.size == 0:
        return None
    centroid = coords.mean(axis=0)
    return float(centroid[0]), float(centroid[1]), float(centroid[2])


def detect_cad_markers(
    volume: np.ndarray,
    *,
    percentile: float = 92.0,
    min_distance: int = 8,
    max_markers: int = 5,
) -> np.ndarray:
    """Return CAD-style (N, 3) z,y,x coordinates at enhancement peaks (QC only)."""
    from skimage.feature import peak_local_max

    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    positive = finite[finite > 0]
    if positive.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    threshold = float(np.percentile(positive, percentile))
    coords = peak_local_max(
        volume,
        min_distance=min_distance,
        threshold_abs=threshold,
        num_peaks=max_markers,
    )
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return coords.astype(np.float32)
