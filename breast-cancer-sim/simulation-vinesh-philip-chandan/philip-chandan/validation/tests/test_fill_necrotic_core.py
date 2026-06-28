"""Tests for necrotic-core hole fill on aligned-bbox masks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

VALIDATION_DIR = Path(__file__).resolve().parent.parent
PHILIP_CHANDAN_DIR = VALIDATION_DIR.parent

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(VALIDATION_DIR))

from fill_necrotic_core import fill_necrotic_core_in_mask  # noqa: E402


def test_2d_fill_closes_ring_in_cuboid() -> None:
    mask = np.zeros((4, 12, 12), dtype=np.uint8)
    mask[1, 2:10, 2] = 1
    mask[1, 2:10, 9] = 1
    mask[1, 2, 2:10] = 1
    mask[1, 9, 2:10] = 1
    assert mask[1, 5, 5] == 0

    filled, stats = fill_necrotic_core_in_mask(
        mask,
        z_slice=slice(0, 4),
        y_slice=slice(0, 12),
        x_slice=slice(0, 12),
        mode="2d",
    )
    assert stats["core_voxels_filled"] > 0
    assert filled[1, 5, 5] == 1


def test_3d_fill_leaves_open_z_column() -> None:
    mask = np.zeros((4, 8, 8), dtype=np.uint8)
    mask[:, 1, 1] = 1
    mask[:, 1, 6] = 1
    mask[:, 6, 1] = 1
    mask[:, 6, 6] = 1
    filled_2d, s2 = fill_necrotic_core_in_mask(
        mask,
        z_slice=slice(0, 4),
        y_slice=slice(0, 8),
        x_slice=slice(0, 8),
        mode="2d",
    )
    filled_3d, s3 = fill_necrotic_core_in_mask(
        mask,
        z_slice=slice(0, 4),
        y_slice=slice(0, 8),
        x_slice=slice(0, 8),
        mode="3d",
    )
    assert s2["core_voxels_filled"] >= s3["core_voxels_filled"]
