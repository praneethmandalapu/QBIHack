"""Tests for post-alignment bbox bright-fraction sweeps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

VALIDATION_DIR = Path(__file__).resolve().parents[1]
STRETCH_DIR = VALIDATION_DIR.parent / "stretch"
PHILIP_CHANDAN_DIR = VALIDATION_DIR.parent

sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(STRETCH_DIR))
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))

from aligned_bbox_tumor import (  # noqa: E402
    embed_bbox_mask_in_full_volume,
    pick_phase_and_threshold,
    threshold_for_target_fraction,
)
from dce_phases import DcePhase  # noqa: E402
from les_cuboid_brightness import (  # noqa: E402
    compute_aligned_bbox_brightness_table,
    extract_bbox_from_slab,
)


def test_extract_bbox_from_slab() -> None:
    slab = np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6)
    les_meta = {"y_start": 1, "y_end": 3, "x_start": 2, "x_end": 4, "z_start": 0, "z_end": 1}
    bbox = extract_bbox_from_slab(slab, les_meta)
    assert bbox.shape == (2, 3, 3)
    assert np.array_equal(bbox, slab[:, 1:4, 2:5])


def test_aligned_bbox_brightness_table() -> None:
    les_meta = {"y_start": 0, "y_end": 3, "x_start": 0, "x_end": 3, "z_start": 1, "z_end": 2}
    slab = np.zeros((2, 8, 8), dtype=np.float32)
    slab[:, :4, :4] = 0.2
    slab[:, :4, :4:2] = 0.9

    expert = np.zeros_like(slab, dtype=np.uint8)
    expert[0, 1, 1] = 1
    expert[1, 2, 2] = 1

    phases = [DcePhase(index=1, z_start=0, z_end=8, acquisition_time="")]
    rows = compute_aligned_bbox_brightness_table(
        {1: slab},
        phases,
        les_meta,
        expert,
        thresholds=np.array([0.5, 0.8]),
        normalize=False,
    )
    assert len(rows) == 2
    assert rows[0].cuboid_voxels == 4 * 4 * 2
    assert rows[0].les_voxels == 2
    assert rows[0].bright_fraction >= rows[1].bright_fraction


def test_threshold_for_target_fraction_interpolates() -> None:
    thresholds = np.array([0.2, 0.4, 0.6])
    fractions = np.array([1.0, 0.5, 0.0])
    assert abs(threshold_for_target_fraction(thresholds, fractions, 0.5) - 0.4) < 1e-6
    assert threshold_for_target_fraction(thresholds, fractions, 1.0) == 0.2


def test_pick_phase_prefers_enhancement() -> None:
    curves = {
        2: (np.array([0.1, 0.5]), np.array([1.0, 0.2])),
        3: (np.array([0.1, 0.5]), np.array([1.0, 0.4])),
    }
    selection = pick_phase_and_threshold(curves, 0.4, candidate_phases=(2, 3))
    assert selection.phase_index == 2
    assert abs(selection.threshold - 0.4) < 1e-5


def test_embed_bbox_mask_in_full_volume() -> None:
    les_meta = {"y_start": 1, "y_end": 2, "x_start": 1, "x_end": 2, "z_start": 3, "z_end": 4}
    bbox = np.ones((2, 2, 2), dtype=np.uint8)
    full = embed_bbox_mask_in_full_volume(bbox, les_meta, (10, 6, 6))
    assert int(full.sum()) == 8
    assert full[3:5, 1:3, 1:3].all()


def test_steepest_dropout_finds_knee() -> None:
    from les_cuboid_brightness import steepest_dropout_threshold

    thresholds = np.array([0.2, 0.4, 0.6, 0.8])
    fractions = np.array([1.0, 0.9, 0.2, 0.0])
    knee_t, slope = steepest_dropout_threshold(thresholds, fractions)
    assert 0.4 <= knee_t <= 0.6
    assert slope < 0


def test_expand_les_meta_yx_clamps_to_volume() -> None:
    from les_cuboid_brightness import expand_les_meta_yx

    meta = {"y_start": 2, "y_end": 4, "x_start": 1, "x_end": 3, "z_start": 0, "z_end": 1}
    expanded = expand_les_meta_yx(meta, 5, y_size=6, x_size=5)
    assert expanded["y_start"] == 0
    assert expanded["x_end"] == 4


def test_threshold_mask_in_slab() -> None:
    from les_cuboid_brightness import threshold_mask_in_slab

    slab = np.zeros((2, 6, 6), dtype=np.float32)
    slab[:, 1:4, 2:5] = np.linspace(0, 1, 18, dtype=np.float32).reshape(2, 3, 3)
    les_meta = {"y_start": 1, "y_end": 3, "x_start": 2, "x_end": 4, "z_start": 0, "z_end": 1}
    mask = threshold_mask_in_slab(slab, les_meta, 0.5)
    assert mask.shape == slab.shape
    assert 0 < int(mask.sum()) < mask.size
