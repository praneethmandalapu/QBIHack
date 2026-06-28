"""Tumor mask from aligned P1 z-band slabs + center-connected bbox threshold."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cuboid_phase_registration import ZBandAlignmentResult
from dce_phases import DcePhase
from les_cuboid_brightness import (
    POSTCONTRAST_ANALYSIS_PHASES,
    bbox_yx_slices,
    bright_fraction_curves_by_phase,
    center_connected_mask_in_bbox,
    compute_aligned_bbox_connected_table,
    elbow_threshold,
    extract_bbox_from_slab,
    les_fraction_in_bbox_slab,
    normalized_bbox_volume_in_slab,
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
    connected_fraction_at_threshold: float
    gap_voxels: int = 0


def pick_phase_and_threshold(
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    candidate_phases: tuple[int, ...] = POSTCONTRAST_ANALYSIS_PHASES,
) -> AlignedTumorSelection:
    """Pick P2/P3 with largest connected-region drop (peak − value at elbow)."""
    best: AlignedTumorSelection | None = None
    best_drop = -1.0

    for phase_index in candidate_phases:
        if phase_index not in curves:
            continue
        thresholds, fractions = curves[phase_index]
        if thresholds.size == 0:
            continue
        thresh, _ = elbow_threshold(thresholds, fractions)
        idx = int(np.argmin(np.abs(thresholds - thresh)))
        at_elbow = float(fractions[idx])
        drop = float(fractions[0] - at_elbow) if fractions.size else 0.0
        if drop > best_drop:
            best_drop = drop
            best = AlignedTumorSelection(
                phase_index=phase_index,
                threshold=thresh,
                les_fraction=0.0,
                connected_fraction_at_threshold=at_elbow,
            )

    if best is not None:
        return best

    return AlignedTumorSelection(
        phase_index=candidate_phases[0],
        threshold=0.5,
        les_fraction=0.0,
        connected_fraction_at_threshold=0.0,
    )


def segment_tumor_in_aligned_bbox(
    alignment: ZBandAlignmentResult,
    phases: list[DcePhase],
    les_meta: dict[str, Any],
    *,
    phase_index: int | None = None,
    threshold: float | None = None,
    threshold_step: float = 0.05,
    gap_voxels: int = 0,
    candidate_phases: tuple[int, ...] = POSTCONTRAST_ANALYSIS_PHASES,
) -> tuple[np.ndarray, AlignedTumorSelection, dict[str, Any]]:
    """Center-connected mask inside bbox on aligned slab (z-band × y × x)."""
    rows = compute_aligned_bbox_connected_table(
        alignment.slabs_aligned,
        phases,
        les_meta,
        alignment.expert_slab,
        threshold_step=threshold_step,
        gap_voxels=gap_voxels,
        phase_filter=candidate_phases,
    )
    curves = bright_fraction_curves_by_phase(rows)
    les_frac, les_voxels, cuboid_voxels = les_fraction_in_bbox_slab(
        alignment.expert_slab,
        les_meta,
    )

    if phase_index is None or threshold is None:
        selection = pick_phase_and_threshold(curves, candidate_phases=candidate_phases)
        phase_index = phase_index or selection.phase_index
        threshold = threshold if threshold is not None else selection.threshold
    else:
        _, fractions = curves.get(phase_index, (np.array([]), np.array([])))
        idx = int(np.argmin(np.abs(curves[phase_index][0] - threshold))) if fractions.size else 0
        selection = AlignedTumorSelection(
            phase_index=phase_index,
            threshold=threshold,
            les_fraction=les_frac,
            connected_fraction_at_threshold=float(fractions[idx]) if fractions.size else 0.0,
            gap_voxels=gap_voxels,
        )

    slab = alignment.slabs_aligned[phase_index]
    norm_bbox = normalized_bbox_volume_in_slab(slab, les_meta)
    mask_bbox = center_connected_mask_in_bbox(
        norm_bbox,
        float(threshold),
        gap_voxels=gap_voxels,
    )

    detail = {
        "phase_index": phase_index,
        "threshold": float(threshold),
        "gap_voxels": int(gap_voxels),
        "les_fraction": les_frac,
        "les_voxels": les_voxels,
        "bbox_voxels": cuboid_voxels,
        "mask_voxels": int(mask_bbox.sum()),
        "segmentation": "center_connected_from_bbox_center",
        "analysis_phases": list(candidate_phases),
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
    gap_voxels: int = 0,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Write ``{slug}_aligned_bbox_tumor_mask.npy`` + JSON under segmentation outputs."""
    bbox_mask, selection, detail = segment_tumor_in_aligned_bbox(
        alignment,
        phases,
        les_meta,
        phase_index=phase_index,
        threshold=threshold,
        gap_voxels=gap_voxels,
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
        "gap_voxels": selection.gap_voxels,
        "les_fraction": selection.les_fraction,
        "connected_fraction_at_threshold": selection.connected_fraction_at_threshold,
        "mask_npy": str(mask_path),
        **detail,
        **(extra_metadata or {}),
        **{k: les_meta[k] for k in ("y_start", "y_end", "x_start", "x_end", "z_start", "z_end")},
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    spacing = (extra_metadata or {}).get("spacing_mm")
    if spacing and len(spacing) == 3:
        import sys

        sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
        from publish_expert_mask import publish_expert_mask_from_volume  # noqa: E402

        expert_nii = publish_expert_mask_from_volume(
            slug,
            full_mask,
            spacing_mm=[float(v) for v in spacing],
        )
        metadata["segmentation_path"] = str(expert_nii.relative_to(REPO_ROOT))
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return full_mask, metadata
