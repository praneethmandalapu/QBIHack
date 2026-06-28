"""Cuboid-constrained enhancement segmentation for baseline DCE-MRI."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_closing, label
from skimage.filters import threshold_otsu

SEGMENTATION_DIR = Path(__file__).resolve().parents[1]
STRETCH_DIR = SEGMENTATION_DIR.parent / "stretch"
PHILIP_CHANDAN_DIR = SEGMENTATION_DIR.parent
VALIDATION_DIR = PHILIP_CHANDAN_DIR / "validation"

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(STRETCH_DIR))
sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(SEGMENTATION_DIR))

from dce_phases import (  # noqa: E402
    DcePhase,
    compute_subtraction,
    load_precontrast_volume,
    resolve_dce_dicom_dir_for_study,
    resolve_phase_ranges,
    resample_volume,
    split_dce_phases,
)
from load_les_mask import find_les_files, load_les_mask  # noqa: E402
from load_manifest import find_volume  # noqa: E402
from prep_volume import normalize_volume  # noqa: E402
from validate_segmentation import load_annotation_volume  # noqa: E402

from load_volume import load_slug_volume  # noqa: E402
from seg_paths import ensure_segmentation_dirs, mask_metadata_json, mask_npy  # noqa: E402

METHOD_ID = "cuboid_enhancement"


@dataclass(frozen=True)
class CuboidEnhancementParams:
    margin_yx: int = 5
    margin_z: int = 2
    candidate_phases: tuple[int, ...] = (1, 2, 3, 4)
    threshold_percentile: float = 90.0
    use_otsu_within_roi: bool = True
    max_volume_multiple: float = 5.0
    closing_radius: int = 1


def _clamp_bounds(
    start: int,
    end: int,
    size: int,
    margin: int,
) -> tuple[int, int]:
    return max(start - margin, 0), min(end + margin, size - 1)


def roi_slices_from_les(
    les_meta: dict[str, Any],
    volume_shape: tuple[int, int, int],
    *,
    margin_yx: int,
    margin_z: int,
    z_start: int | None = None,
    z_end: int | None = None,
) -> tuple[slice, slice, slice]:
    """Return (Z, Y, X) slices for a cuboid ROI with optional Z override."""
    z_size, y_size, x_size = volume_shape
    y0, y1 = _clamp_bounds(les_meta["y_start"], les_meta["y_end"], y_size, margin_yx)
    x0, x1 = _clamp_bounds(les_meta["x_start"], les_meta["x_end"], x_size, margin_yx)
    if z_start is None or z_end is None:
        z0, z1 = _clamp_bounds(les_meta["z_start"], les_meta["z_end"], z_size, margin_z)
    else:
        z0, z1 = max(z_start, 0), min(z_end, z_size - 1)
    return slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1)


def expert_yx_footprint(expert_mask: np.ndarray) -> np.ndarray:
    """Project expert voxels onto (Y, X) — stable seed when phase z differs."""
    return expert_mask.any(axis=0)


def _normalize_roi(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    clipped = np.clip(values, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def _threshold_roi(norm_roi: np.ndarray, *, percentile: float, use_otsu: bool) -> np.ndarray:
    finite = norm_roi[np.isfinite(norm_roi)]
    if finite.size == 0 or float(finite.max()) <= float(finite.min()):
        return np.zeros_like(norm_roi, dtype=bool)
    positive = finite[finite > 0]
    values = positive if positive.size else finite
    if use_otsu and values.max() > values.min():
        thresh = float(threshold_otsu(values))
    else:
        thresh = float(np.percentile(values, percentile))
    return norm_roi >= thresh


def _pick_component(
    candidate: np.ndarray,
    enhancement: np.ndarray,
    expert_footprint_yx: np.ndarray,
    roi_yx_slices: tuple[slice, slice],
) -> np.ndarray:
    labels, n = label(candidate)
    if n == 0:
        return np.zeros_like(candidate, dtype=np.uint8)

    y_sl, x_sl = roi_yx_slices
    footprint_roi = expert_footprint_yx[y_sl, x_sl]

    best_label = 0
    best_score = (-1.0, -1.0)
    for comp_label in range(1, n + 1):
        comp = labels == comp_label
        comp_yx = comp.any(axis=0)
        overlap = int(np.logical_and(comp_yx, footprint_roi).sum())
        mean_enh = float(enhancement[comp].mean()) if comp.any() else 0.0
        score = (overlap, mean_enh)
        if score > best_score:
            best_score = score
            best_label = comp_label

    return (labels == best_label).astype(np.uint8)


def segment_phase_in_roi(
    enhancement: np.ndarray,
    roi_slices: tuple[slice, slice, slice],
    expert_footprint_yx: np.ndarray,
    *,
    params: CuboidEnhancementParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment one phase volume inside ROI; return mask embedded in full `enhancement` shape."""
    z_sl, y_sl, x_sl = roi_slices
    roi_enh = enhancement[z_sl, y_sl, x_sl]
    norm_roi = _normalize_roi(roi_enh)
    candidate = _threshold_roi(
        norm_roi,
        percentile=params.threshold_percentile,
        use_otsu=params.use_otsu_within_roi,
    )
    selected = _pick_component(
        candidate,
        roi_enh,
        expert_footprint_yx,
        (y_sl, x_sl),
    )

    if params.closing_radius > 0:
        structure = np.ones(
            (2 * params.closing_radius + 1,) * 3,
            dtype=bool,
        )
        selected = binary_closing(selected.astype(bool), structure=structure).astype(np.uint8)

    out = np.zeros_like(enhancement, dtype=np.uint8)
    out[z_sl, y_sl, x_sl] = selected
    meta = {
        "roi_voxels": int(selected.sum()),
        "candidate_voxels": int(candidate.sum()),
    }
    return out, meta


def _resolve_dicom_dir(entry: dict[str, Any], sidecar: dict[str, Any], dce_index: int) -> Path | None:
    source_dicom_dir: Path | None = None
    rel = sidecar.get("source_dicom_dir")
    if rel:
        candidate = PHILIP_CHANDAN_DIR.parents[1] / rel
        if candidate.is_dir():
            source_dicom_dir = candidate
    return resolve_dce_dicom_dir_for_study(
        tcga_id=entry["tcga_id"],
        study_date=entry["study_date"],
        dce_index=dce_index,
        source_dicom_dir=source_dicom_dir,
    )


def _phase_enhancement(
    phase_volume: np.ndarray,
    spacing_mm: list[float],
    precontrast: tuple[np.ndarray, list[float]] | None,
    *,
    phase_index: int,
) -> tuple[np.ndarray, str]:
    if phase_index > 1 and precontrast is not None:
        pre_volume, pre_spacing = precontrast
        pre_resampled = resample_volume(
            pre_volume,
            pre_spacing,
            phase_volume.shape,
            spacing_mm,
        )
        return compute_subtraction(phase_volume, pre_resampled), "subtraction"
    return normalize_volume(phase_volume), "normalized_phase"


def _align_mask_to_expert_z(
    mask: np.ndarray,
    expert_mask: np.ndarray,
    les_meta: dict[str, Any],
) -> tuple[np.ndarray, bool]:
    """If 3D masks do not overlap, extrude Y/X footprint through expert z band."""
    if int(np.logical_and(mask.astype(bool), expert_mask.astype(bool)).sum()) > 0:
        return mask, False

    z0, z1 = les_meta["z_start"], les_meta["z_end"]
    yx = np.logical_and(mask.any(axis=0), expert_mask.any(axis=0))
    if not yx.any():
        yx = np.logical_and(mask.any(axis=0), expert_yx_footprint(expert_mask))
    if not yx.any():
        return mask, False

    aligned = np.zeros_like(mask, dtype=np.uint8)
    aligned[z0 : z1 + 1, yx] = 1
    return aligned, True


def segment_cuboid_enhancement(
    slug: str,
    *,
    lesions_dir: Path | None = None,
    params: CuboidEnhancementParams | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run cuboid-constrained enhancement segmentation for a baseline slug."""
    params = params or CuboidEnhancementParams()
    entry = find_volume(slug=slug)
    if entry.get("timepoint") != "baseline":
        raise ValueError(f"cuboid_enhancement is baseline-only; got timepoint={entry.get('timepoint')!r}")

    tcga_id = entry["tcga_id"]
    study_date = entry["study_date"]
    les_files = find_les_files(tcga_id, lesions_dir)
    if not les_files:
        raise FileNotFoundError(f"No .les file for {tcga_id} ({slug})")
    les_path = les_files[0]

    volume, spacing_mm, series_description, dce_index, volume_source = load_annotation_volume(
        tcga_id=tcga_id,
        study_date=study_date,
        les_path=les_path,
        slug=slug,
    )
    expert_mask, les_meta = load_les_mask(les_path, volume.shape)
    expert_voxels = int(expert_mask.sum())
    expert_footprint = expert_yx_footprint(expert_mask)

    _, sidecar = load_slug_volume(slug)
    dicom_dir = _resolve_dicom_dir(entry, sidecar, dce_index)
    phases = resolve_phase_ranges(volume_shape=volume.shape, dicom_dir=dicom_dir)
    phase_volumes = split_dce_phases(volume, phases)
    phase_split_source = "dicom_acquisition_time" if dicom_dir else "equal_quarters"

    precontrast_raw = load_precontrast_volume(tcga_id=tcga_id, study_date=study_date)
    precontrast = (precontrast_raw[0], precontrast_raw[1]) if precontrast_raw else None

    phase_by_index = {phase.index: phase for phase in phases}
    candidates = [idx for idx in params.candidate_phases if idx in phase_by_index]
    if not candidates:
        candidates = [phase.index for phase in phases if phase.index > 1] or [phases[0].index]

    best_mask: np.ndarray | None = None
    best_overlap = -1
    best_phase = candidates[0]
    best_detail: dict[str, Any] = {}
    best_enhancement: np.ndarray | None = None

    for phase_index in candidates:
        phase = phase_by_index[phase_index]
        phase_vol = phase_volumes[phase_index - 1]
        enhancement, enh_kind = _phase_enhancement(
            phase_vol,
            spacing_mm,
            precontrast,
            phase_index=phase_index,
        )

        full_enh = np.zeros_like(volume, dtype=np.float32)
        full_enh[phase.z_start : phase.z_end] = enhancement

        roi_slices = roi_slices_from_les(
            les_meta,
            volume.shape,
            margin_yx=params.margin_yx,
            margin_z=params.margin_z,
            z_start=phase.z_start,
            z_end=phase.z_end - 1,
        )
        phase_mask, phase_detail = segment_phase_in_roi(
            full_enh,
            roi_slices,
            expert_footprint,
            params=params,
        )

        pred_yx = phase_mask.any(axis=0)
        overlap = int(np.logical_and(pred_yx, expert_footprint).sum())
        if overlap > best_overlap or (
            overlap == best_overlap
            and int(phase_mask.sum()) > int(best_mask.sum() if best_mask is not None else 0)
        ):
            best_overlap = overlap
            best_phase = phase_index
            best_mask = phase_mask
            best_enhancement = full_enh
            best_detail = {
                **phase_detail,
                "enhancement_kind": enh_kind,
                "phase_z_range": [phase.z_start, phase.z_end],
            }

    assert best_mask is not None
    assert best_enhancement is not None

    best_mask, z_aligned = _align_mask_to_expert_z(best_mask, expert_mask, les_meta)

    if expert_voxels > 0 and int(best_mask.sum()) > int(expert_voxels * params.max_volume_multiple):
        cap = int(expert_voxels * params.max_volume_multiple)
        coords = np.argwhere(best_mask > 0)
        order = np.argsort(best_enhancement[tuple(coords.T)])[::-1]
        trimmed = np.zeros_like(best_mask)
        keep = coords[order[:cap]]
        trimmed[tuple(keep.T)] = 1
        best_mask = trimmed
        best_detail["volume_capped"] = True
        best_detail["cap_voxels"] = cap

    metadata: dict[str, Any] = {
        "slug": slug,
        "method": METHOD_ID,
        "role": "predicted",
        "tcga_id": tcga_id,
        "subtype": entry.get("subtype"),
        "timepoint": entry.get("timepoint"),
        "study_date": study_date,
        "les_file": les_path.name,
        "dce_index": dce_index,
        "annotated_series": series_description,
        "volume_source": volume_source,
        "shape_zyx": list(volume.shape),
        "spacing_mm": spacing_mm,
        "mask_voxels": int(best_mask.sum()),
        "expert_voxels": expert_voxels,
        "expert_overlap_voxels": int(
            np.logical_and(best_mask.astype(bool), expert_mask.astype(bool)).sum()
        ),
        "expert_yx_overlap_voxels": best_overlap,
        "z_aligned_to_expert": z_aligned,
        "selected_phase": best_phase,
        "phase_split_source": phase_split_source,
        "precontrast_used": precontrast is not None,
        "params": {
            "margin_yx": params.margin_yx,
            "margin_z": params.margin_z,
            "candidate_phases": list(params.candidate_phases),
            "threshold_percentile": params.threshold_percentile,
            "use_otsu_within_roi": params.use_otsu_within_roi,
            "max_volume_multiple": params.max_volume_multiple,
        },
        **best_detail,
        **{k: les_meta[k] for k in ("y_start", "y_end", "x_start", "x_end", "z_start", "z_end")},
    }
    return best_mask.astype(np.uint8), metadata


def write_cuboid_enhancement_mask(
    slug: str,
    *,
    lesions_dir: Path | None = None,
    params: CuboidEnhancementParams | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment and write `{slug}_cuboid_enhancement_mask.npy` + JSON sidecar."""
    mask, metadata = segment_cuboid_enhancement(slug, lesions_dir=lesions_dir, params=params)
    ensure_segmentation_dirs()
    mask_path = mask_npy(slug, METHOD_ID)
    meta_path = mask_metadata_json(slug, METHOD_ID)
    np.save(mask_path, mask)
    metadata["mask_npy"] = str(mask_path)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return mask, metadata
