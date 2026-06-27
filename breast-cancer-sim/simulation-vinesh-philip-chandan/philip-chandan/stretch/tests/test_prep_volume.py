"""Tests for stretch prep_volume (Phase 2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

STRETCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRETCH_DIR))

from prep_volume import (  # noqa: E402
    crop_to_mask_bbox,
    normalize_volume,
    numpy_to_sitk,
    tumor_mask_largest_component,
)


def test_normalize_volume_range(synthetic_raw_sphere):
    raw, _ = synthetic_raw_sphere
    norm = normalize_volume(raw)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    assert norm.max() > norm.min()


def test_tumor_mask_nonempty_and_localized(synthetic_raw_sphere):
    raw, _ = synthetic_raw_sphere
    norm = normalize_volume(raw)
    mask = tumor_mask_largest_component(norm)
    assert mask.dtype == np.uint8
    assert mask.any()
    # Mask should be much smaller than the full FOV (localized tumor).
    assert mask.mean() < 0.15


def test_crop_to_mask_bbox_shrinks(synthetic_raw_sphere):
    raw, _ = synthetic_raw_sphere
    norm = normalize_volume(raw)
    mask = tumor_mask_largest_component(norm)
    cropped_img, cropped_mask = crop_to_mask_bbox(norm, mask, margin=5)
    assert cropped_img.shape == cropped_mask.shape
    assert np.prod(cropped_img.shape) < np.prod(norm.shape)


def test_numpy_to_sitk_spacing(synthetic_raw_sphere):
    raw, spacing = synthetic_raw_sphere
    norm = normalize_volume(raw)
    mask = tumor_mask_largest_component(norm)
    sitk_image, sitk_mask = numpy_to_sitk(norm, mask, spacing)
    assert sitk_image.GetSpacing() == tuple(float(s) for s in spacing)
    assert sitk_mask.GetSpacing() == sitk_image.GetSpacing()
    assert sitk_image.GetSize() == sitk_mask.GetSize()


def test_empty_volume_returns_empty_mask():
    flat = np.full((16, 16, 16), 100.0, dtype=np.float32)
    norm = normalize_volume(flat)
    mask = tumor_mask_largest_component(norm)
    assert not mask.any()
