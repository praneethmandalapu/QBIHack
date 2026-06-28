"""Extract 3D brain MR volumes and expert masks from NIfTI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

UCSF_TIMEPOINT_TOKENS = {
    "baseline": "time1",
    "followup": "time2",
    "tp1": "time1",
    "tp2": "time2",
    "time1": "time1",
    "time2": "time2",
}

# BraTS-style expert labels; whole tumor (WT) = 1 + 2 + 3 per cohort.json.
WT_SEGMENTATION_LABELS = (1, 2, 3)


def _load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} from {path}")
    volume = np.transpose(data, (2, 1, 0)).astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    return volume, spacing


def extract_volume(nifti_path: Path) -> np.ndarray:
    """Load NIfTI as (Z, Y, X) float32."""
    volume, _ = _load_nifti(nifti_path)
    return volume


def extract_spacing(nifti_path: Path) -> tuple[float, float, float]:
    """Return voxel spacing as (dz, dy, dx) mm matching (Z, Y, X)."""
    _, spacing = _load_nifti(nifti_path)
    return spacing


def load_expert_mask(mask_path: Path, mr_shape: tuple[int, ...]) -> np.ndarray:
    """Load binary expert mask aligned to MR grid."""
    mask, _ = _load_nifti(mask_path)
    if mask.shape != mr_shape:
        raise ValueError(f"Mask shape {mask.shape} != MR shape {mr_shape} ({mask_path})")
    return (mask > 0).astype(np.float32)


def load_segmentation_labels(seg_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load integer segmentation labels as (Z, Y, X) with spacing (dz, dy, dx) mm."""
    import nibabel as nib

    img = nib.load(str(seg_path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} from {seg_path}")
    labels = np.transpose(data, (2, 1, 0))
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    return labels, spacing


def compute_wt_volume_mm3(
    labels: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    wt_labels: tuple[int, ...] = WT_SEGMENTATION_LABELS,
) -> float:
    """Whole-tumor volume in mm^3 from expert label map (default labels 1+2+3)."""
    voxel_mm3 = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2])
    count = int(np.isin(labels, wt_labels).sum())
    return count * voxel_mm3


def wt_volume_from_segmentation(seg_path: Path) -> float:
    """Convenience: WT mm^3 directly from a segmentation NIfTI path."""
    labels, spacing = load_segmentation_labels(seg_path)
    return compute_wt_volume_mm3(labels, spacing)


def resolve_ucsf_paths(
    patient_dir: Path,
    timepoint_label: str = "baseline",
) -> tuple[Path, Path]:
    """Return (t1ce MR, expert seg) for a UCSF-ALPTDG patient folder."""
    patient_id = patient_dir.name
    token = UCSF_TIMEPOINT_TOKENS.get(timepoint_label.lower(), timepoint_label)
    mr_path = patient_dir / f"{patient_id}_{token}_t1ce.nii.gz"
    seg_path = patient_dir / f"{patient_id}_{token}_seg.nii.gz"
    if not mr_path.is_file():
        raise FileNotFoundError(f"MR not found: {mr_path}")
    if not seg_path.is_file():
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")
    return mr_path, seg_path


def resolve_ucsf_supplementary_paths(
    patient_dir: Path,
    timepoint_label: str = "baseline",
) -> dict[str, Path]:
    """Return available UCSF supplementary MR series for the same timepoint."""
    patient_id = patient_dir.name
    token = UCSF_TIMEPOINT_TOKENS.get(timepoint_label.lower(), timepoint_label)
    candidates = {
        "t1ce": patient_dir / f"{patient_id}_{token}_t1ce.nii.gz",
        "t1": patient_dir / f"{patient_id}_{token}_t1.nii.gz",
        "t2": patient_dir / f"{patient_id}_{token}_t2.nii.gz",
        "flair": patient_dir / f"{patient_id}_{token}_flair.nii.gz",
    }
    return {name: path for name, path in candidates.items() if path.is_file()}


def validate_nifti_pair(mr_path: Path, seg_path: Path) -> dict[str, Any]:
    """Validate MR + expert mask paths before export."""
    errors: list[str] = []
    if not mr_path.is_file():
        errors.append(f"MR missing: {mr_path}")
    if not seg_path.is_file():
        errors.append(f"segmentation missing: {seg_path}")
    if errors:
        return {"ok": False, "errors": errors, "shape": None, "spacing_mm": None}

    volume = extract_volume(mr_path)
    spacing = extract_spacing(mr_path)
    try:
        load_expert_mask(seg_path, volume.shape)
    except ValueError as exc:
        errors.append(str(exc))

    return {
        "ok": not errors,
        "errors": errors,
        "shape": tuple(volume.shape),
        "spacing_mm": list(spacing),
    }


def extract_volume_with_spacing(mr_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    return _load_nifti(mr_path)
