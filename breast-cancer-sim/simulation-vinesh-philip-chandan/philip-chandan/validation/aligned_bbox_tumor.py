"""Tumor mask from aligned P1 z-band slabs + bbox threshold curve."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label

from cuboid_phase_registration import ZBandAlignmentResult
from dce_phases import DcePhase
from les_cuboid_brightness import (
    bbox_yx_slices,
    bright_fraction_curves_by_phase,
    compute_aligned_bbox_brightness_table,
    extract_bbox_from_slab,
    les_fraction_in_bbox_slab,
)

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PHILIP_CHANDAN_DIR.parents[1]
SEGMENTATION_OUT_DIR = REPO_ROOT / "data" / "processed" / "segmentation-philip-chandan"
METHOD_ID = "aligned_bbox_tumor"


@dataclass(frozen=True)
class AlignedTumorSelection:
    phase_index: int
    threshold: float
    les_fraction: float
    bright_fraction_at_threshold: float


def threshold_for_target_fraction(
    thresholds: np.ndarray,
    fractions: np.ndarray,
    target: float,
) -> float:
    """Interpolate threshold where ``fractions`` crosses ``target`` (monotone decreasing)."""
    if thresholds.size == 0:
        return 0.5
    if target >= float(fractions[0]):
        return float(thresholds[0])
    if target <= float(fractions[-1]):
        return float(thresholds[-1])

    for index in range(len(thresholds) - 1):
        f0, f1 = float(fractions[index]), float(fractions[index + 1])
        t0, t1 = float(thresholds[index]), float(thresholds[index + 1])
        if f0 >= target >= f1:
            if abs(f0 - f1) <= 1e-12:
                return t0
            return t0 + (target - f0) / (f1 - f0) * (t1 - t0)
    return float(thresholds[-1])


def pick_phase_and_threshold(
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
    les_fraction: float,
    *,
    candidate_phases: tuple[int, ...] = (2, 3, 4),
) -> AlignedTumorSelection:
    """Pick post-contrast phase with strongest drop from peak to les-matched threshold."""
    best: AlignedTumorSelection | None = None
    best_enhancement = -1.0

    for phase_index in candidate_phases:
        if phase_index not in curves:
            continue
        thresholds, fractions = curves[phase_index]
        thresh = threshold_for_target_fraction(thresholds, fractions, les_fraction)
        idx = int(np.argmin(np.abs(thresholds - thresh)))
        bright_at = float(fractions[idx]) if fractions.size else les_fraction
        enhancement = float(fractions[0] - bright_at) if fractions.size else 0.0
        if enhancement > best_enhancement:
            best_enhancement = enhancement
            best = AlignedTumorSelection(
                phase_index=phase_index,
                threshold=thresh,
                les_fraction=les_fraction,
                bright_fraction_at_threshold=bright_at,
            )

    if best is not None:
        return best

    thresholds, fractions = curves.get(1, (np.array([0.5]), np.array([1.0])))
    thresh = threshold_for_target_fraction(thresholds, fractions, les_fraction)
    idx = int(np.argmin(np.abs(thresholds - thresh)))
    return AlignedTumorSelection(
        phase_index=1,
        threshold=thresh,
        les_fraction=les_fraction,
        bright_fraction_at_threshold=float(fractions[idx]) if fractions.size else les_fraction,
    )


def _expert_footprint_yx(expert_bbox: np.ndarray) -> np.ndarray:
    return expert_bbox.any(axis=0)


def _pick_component(
    candidate: np.ndarray,
    intensity: np.ndarray,
    expert_footprint_yx: np.ndarray,
) -> np.ndarray:
    labels, n = label(candidate)
    if n == 0:
        return np.zeros_like(candidate, dtype=np.uint8)

    best_label = 0
    best_score = (-1, -1.0)
    for comp_label in range(1, n + 1):
        comp = labels == comp_label
        comp_yx = comp.any(axis=0)
        overlap = int(np.logical_and(comp_yx, expert_footprint_yx).sum())
        mean_int = float(intensity[comp].mean()) if comp.any() else 0.0
        score = (overlap, mean_int)
        if score > best_score:
            best_score = score
            best_label = comp_label
    return (labels == best_label).astype(np.uint8)


def segment_tumor_in_aligned_bbox(
    alignment: ZBandAlignmentResult,
    phases: list[DcePhase],
    les_meta: dict[str, Any],
    *,
    phase_index: int | None = None,
    threshold: float | None = None,
    threshold_step: float = 0.05,
    candidate_phases: tuple[int, ...] = (2, 3, 4),
) -> tuple[np.ndarray, AlignedTumorSelection, dict[str, Any]]:
    """Binary mask inside bbox coords (z-band × y × x) on aligned slabs."""
    rows = compute_aligned_bbox_brightness_table(
        alignment.slabs_aligned,
        phases,
        les_meta,
        alignment.expert_slab,
        threshold_step=threshold_step,
    )
    curves = bright_fraction_curves_by_phase(rows)
    les_frac, les_voxels, cuboid_voxels = les_fraction_in_bbox_slab(
        alignment.expert_slab,
        les_meta,
    )

    if phase_index is None or threshold is None:
        selection = pick_phase_and_threshold(
            curves,
            les_frac,
            candidate_phases=candidate_phases,
        )
        phase_index = phase_index or selection.phase_index
        threshold = threshold if threshold is not None else selection.threshold
    else:
        _, fractions = curves.get(phase_index, (np.array([]), np.array([])))
        idx = int(np.argmin(np.abs(curves[phase_index][0] - threshold))) if fractions.size else 0
        selection = AlignedTumorSelection(
            phase_index=phase_index,
            threshold=threshold,
            les_fraction=les_frac,
            bright_fraction_at_threshold=float(fractions[idx]) if fractions.size else 0.0,
        )

    slab = alignment.slabs_aligned[phase_index]
    bbox = extract_bbox_from_slab(slab, les_meta)
    from les_cuboid_brightness import _normalize_roi

    norm_bbox = _normalize_roi(bbox.ravel()).reshape(bbox.shape)
    candidate = norm_bbox >= float(threshold)

    y_sl, x_sl = bbox_yx_slices(les_meta)
    expert_bbox = alignment.expert_slab[:, y_sl, x_sl].astype(bool)
    footprint = _expert_footprint_yx(expert_bbox)
    mask_bbox = _pick_component(candidate, bbox, footprint)

    detail = {
        "phase_index": phase_index,
        "threshold": float(threshold),
        "les_fraction": les_frac,
        "les_voxels": les_voxels,
        "bbox_voxels": cuboid_voxels,
        "mask_voxels": int(mask_bbox.sum()),
        "candidate_voxels": int(candidate.sum()),
    }
    return mask_bbox.astype(np.uint8), selection, detail


def embed_bbox_mask_in_full_volume(
    bbox_mask: np.ndarray,
    les_meta: dict[str, Any],
    full_shape: tuple[int, int, int],
) -> np.ndarray:
    """Paste bbox mask into full VIBRANT stack at global .les z/y/x indices."""
    full = np.zeros(full_shape, dtype=np.uint8)
    z_sl = slice(int(les_meta["z_start"]), int(les_meta["z_end"]) + 1)
    y_sl, x_sl = bbox_yx_slices(les_meta)
    full[z_sl, y_sl, x_sl] = bbox_mask
    return full


def write_aligned_bbox_tumor_mask(
    slug: str,
    alignment: ZBandAlignmentResult,
    phases: list[DcePhase],
    les_meta: dict[str, Any],
    full_shape: tuple[int, int, int],
    *,
    phase_index: int | None = None,
    threshold: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Write ``{slug}_aligned_bbox_tumor_mask.npy`` + JSON under segmentation outputs."""
    bbox_mask, selection, detail = segment_tumor_in_aligned_bbox(
        alignment,
        phases,
        les_meta,
        phase_index=phase_index,
        threshold=threshold,
    )
    full_mask = embed_bbox_mask_in_full_volume(bbox_mask, les_meta, full_shape)

    SEGMENTATION_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = SEGMENTATION_OUT_DIR / f"{slug}_{METHOD_ID}_mask.npy"
    meta_path = SEGMENTATION_OUT_DIR / f"{slug}_{METHOD_ID}_mask.json"
    np.save(mask_path, full_mask)

    metadata: dict[str, Any] = {
        "slug": slug,
        "method": METHOD_ID,
        "role": "predicted",
        "shape_zyx": list(full_shape),
        "z_band_local": list(alignment.z_band_local),
        "selected_phase": selection.phase_index,
        "threshold": selection.threshold,
        "les_fraction": selection.les_fraction,
        "mask_npy": str(mask_path),
        **detail,
        **(extra_metadata or {}),
        **{k: les_meta[k] for k in ("y_start", "y_end", "x_start", "x_end", "z_start", "z_end")},
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return full_mask, metadata
