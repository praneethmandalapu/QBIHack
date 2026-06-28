"""Tests for cuboid-constrained enhancement segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SEGMENTATION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SEGMENTATION_DIR))

from methods.cuboid_enhancement import (  # noqa: E402
    CuboidEnhancementParams,
    expert_yx_footprint,
    roi_slices_from_les,
    segment_phase_in_roi,
)


def test_roi_slices_clips_to_volume():
    les_meta = {
        "y_start": 10,
        "y_end": 20,
        "x_start": 15,
        "x_end": 25,
        "z_start": 5,
        "z_end": 8,
    }
    z_sl, y_sl, x_sl = roi_slices_from_les(
        les_meta,
        (40, 64, 64),
        margin_yx=3,
        margin_z=1,
    )
    assert z_sl.start == 4 and z_sl.stop == 10
    assert y_sl.start == 7 and y_sl.stop == 24
    assert x_sl.start == 12 and x_sl.stop == 29


def test_roi_slices_phase_z_override():
    les_meta = {
        "y_start": 10,
        "y_end": 12,
        "x_start": 10,
        "x_end": 12,
        "z_start": 2,
        "z_end": 4,
    }
    z_sl, _, _ = roi_slices_from_les(
        les_meta,
        (32, 32, 32),
        margin_yx=2,
        margin_z=2,
        z_start=16,
        z_end=23,
    )
    assert z_sl.start == 16 and z_sl.stop == 24


def test_expert_yx_footprint_projects_all_z():
    expert = np.zeros((8, 10, 10), dtype=np.uint8)
    expert[1, 3, 4] = 1
    expert[6, 3, 4] = 1
    footprint = expert_yx_footprint(expert)
    assert footprint.shape == (10, 10)
    assert footprint[3, 4] == 1
    assert int(footprint.sum()) == 1


def test_segment_phase_in_roi_finds_bright_blob_near_expert():
    shape = (24, 32, 32)
    enhancement = np.zeros(shape, dtype=np.float32)
    expert = np.zeros(shape, dtype=np.uint8)

    # Expert seed in phase 1 (z 0-5), bright blob in phase 2 (z 6-11).
    expert[2, 14, 16] = 1
    expert[3, 15, 17] = 1
    enhancement[8:11, 13:18, 15:20] = 100.0
    enhancement[8:11, 20:22, 20:22] = 80.0  # distractor outside footprint

    les_meta = {
        "y_start": 12,
        "y_end": 18,
        "x_start": 14,
        "x_end": 20,
        "z_start": 2,
        "z_end": 4,
    }
    roi_slices = roi_slices_from_les(
        les_meta,
        shape,
        margin_yx=2,
        margin_z=0,
        z_start=6,
        z_end=11,
    )
    params = CuboidEnhancementParams(
        closing_radius=0,
        use_otsu_within_roi=True,
        threshold_percentile=50.0,
    )
    mask, meta = segment_phase_in_roi(
        enhancement,
        roi_slices,
        expert_yx_footprint(expert),
        params=params,
    )

    assert meta["roi_voxels"] > 0
    assert mask[8:11, 13:18, 15:20].sum() > 0
    assert mask[8:11, 20:22, 20:22].sum() == 0
    assert mask[2, 14, 16] == 0  # not in phase-2 z band
