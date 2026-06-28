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
)
from dce_phases import DcePhase  # noqa: E402
from les_cuboid_brightness import (  # noqa: E402
    center_connected_mask_in_bbox,
    compute_aligned_bbox_connected_table,
    elbow_threshold,
    extract_bbox_from_slab,
)


def test_extract_bbox_from_slab() -> None:
    slab = np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6)
    les_meta = {"y_start": 1, "y_end": 3, "x_start": 2, "x_end": 4, "z_start": 0, "z_end": 1}
    bbox = extract_bbox_from_slab(slab, les_meta)
    assert bbox.shape == (2, 3, 3)
    assert np.array_equal(bbox, slab[:, 1:4, 2:5])


def test_center_connected_keeps_center_blob_only() -> None:
    norm = np.zeros((2, 5, 5), dtype=np.float32)
    norm[:, 2, 2] = 1.0
    norm[:, 0, 0] = 1.0
    mask = center_connected_mask_in_bbox(norm, 0.5, gap_voxels=0)
    assert int(mask[:, 2, 2].sum()) == 2
    assert int(mask[:, 0, 0].sum()) == 0


def test_connected_table_p2_p3_only() -> None:
    les_meta = {"y_start": 0, "y_end": 3, "x_start": 0, "x_end": 3, "z_start": 1, "z_end": 2}
    slab = np.zeros((2, 8, 8), dtype=np.float32)
    slab[:, :4, :4] = 0.9

    expert = np.zeros_like(slab, dtype=np.uint8)
    phases = [
        DcePhase(index=2, z_start=0, z_end=8, acquisition_time=""),
        DcePhase(index=3, z_start=0, z_end=8, acquisition_time=""),
        DcePhase(index=4, z_start=0, z_end=8, acquisition_time=""),
    ]
    rows = compute_aligned_bbox_connected_table(
        {2: slab, 3: slab, 4: slab},
        phases,
        les_meta,
        expert,
        thresholds=np.array([0.5, 0.8]),
    )
    phase_indices = {row.phase_index for row in rows}
    assert phase_indices == {2, 3}
    assert all(row.bright_fraction >= rows[-1].bright_fraction for row in rows if row.phase_index == 2)


def test_elbow_threshold_mid_drop() -> None:
    thresholds = np.array([0.2, 0.4, 0.6, 0.8])
    fractions = np.array([1.0, 0.95, 0.15, 0.0])
    elbow_t, dist = elbow_threshold(thresholds, fractions)
    assert 0.4 <= elbow_t <= 0.6
    assert dist > 0


def test_pick_phase_prefers_larger_elbow_drop() -> None:
    curves = {
        2: (np.array([0.2, 0.4, 0.6]), np.array([1.0, 0.5, 0.1])),
        3: (np.array([0.2, 0.4, 0.6]), np.array([1.0, 0.9, 0.8])),
    }
    selection = pick_phase_and_threshold(curves, candidate_phases=(2, 3))
    assert selection.phase_index == 2


def test_embed_bbox_mask_in_full_volume() -> None:
    les_meta = {"y_start": 1, "y_end": 2, "x_start": 1, "x_end": 2, "z_start": 3, "z_end": 4}
    bbox = np.ones((2, 2, 2), dtype=np.uint8)
    full = embed_bbox_mask_in_full_volume(bbox, les_meta, (10, 6, 6))
    assert int(full.sum()) == 8
    assert full[3:5, 1:3, 1:3].all()
