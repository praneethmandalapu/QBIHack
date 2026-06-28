"""Metrics for comparing a predicted tumor mask to a reference (.les)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

SEGMENTATION_DIR = Path(__file__).resolve().parent
STRETCH_DIR = SEGMENTATION_DIR.parent / "stretch"
sys.path.insert(0, str(STRETCH_DIR))

from validate_segmentation import dice_coefficient, volume_mm3  # noqa: E402


def compare_to_reference(
    predicted: np.ndarray,
    reference: np.ndarray,
    spacing_mm: list[float],
) -> dict[str, float | int]:
    """Return Dice, volumes, and voxel counts for pred vs reference."""
    ref_vol = volume_mm3(reference, spacing_mm)
    pred_vol = volume_mm3(predicted, spacing_mm)
    ref_voxels = int(reference.astype(bool).sum())
    pred_voxels = int(predicted.astype(bool).sum())

    if ref_vol <= 0:
        rel_volume_error = float("nan")
        area_fraction = float("nan")
    else:
        rel_volume_error = (pred_vol - ref_vol) / ref_vol
        area_fraction = pred_voxels / ref_voxels if ref_voxels > 0 else float("nan")

    return {
        "dice": dice_coefficient(reference, predicted),
        "reference_volume_mm3": ref_vol,
        "predicted_volume_mm3": pred_vol,
        "relative_volume_error": rel_volume_error,
        "reference_voxels": ref_voxels,
        "predicted_voxels": pred_voxels,
        "area_fraction_pred_over_ref": area_fraction,
    }


def load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    mask = np.load(path)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask at {path}, got shape {mask.shape}")
    return (mask > 0).astype(np.uint8)
