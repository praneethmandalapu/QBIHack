"""Tests for .les cuboid bright-fraction sweeps."""

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

from dce_phases import DcePhase  # noqa: E402
from les_cuboid_brightness import (  # noqa: E402
    bright_fraction_sweep,
    compute_cuboid_brightness_table,
    default_thresholds,
    extract_phase_z_band_full_xy,
    values_in_phase_cuboid,
)


def test_bright_fraction_sweep_monotone() -> None:
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    thresholds = np.array([0.0, 0.5, 0.75, 1.0])
    fractions, counts = bright_fraction_sweep(values, thresholds)
    assert fractions.tolist() == [1.0, 0.6, 0.4, 0.2]
    assert counts.tolist() == [5, 3, 2, 1]


def test_phase_cuboid_brightness_table() -> None:
    volume = np.zeros((8, 10, 10), dtype=np.float32)
    volume[2:5, 3:7, 4:8] = 100.0
    volume[2:5, 3:7, 4:8:2] = 200.0

    les_meta = {
        "y_start": 3,
        "y_end": 6,
        "x_start": 4,
        "x_end": 7,
        "z_start": 2,
        "z_end": 4,
    }
    expert = np.zeros_like(volume, dtype=np.uint8)
    expert[3, 4, 5] = 1
    expert[3, 5, 6] = 1

    phase = DcePhase(index=1, z_start=0, z_end=8, acquisition_time="")
    rows = compute_cuboid_brightness_table(
        [volume],
        [phase],
        les_meta,
        expert,
        thresholds=np.array([0.5, 0.9]),
        normalize=True,
    )
    assert len(rows) == 2
    assert rows[0].phase_index == 1
    assert rows[0].cuboid_voxels == 4 * 4 * 3
    assert rows[0].les_voxels == 2
    assert 0.0 < rows[0].les_fraction < 1.0
    assert rows[0].bright_fraction >= rows[1].bright_fraction


def test_z_band_full_xy_uses_p1_local_z_on_all_phases() -> None:
    les_meta = {"y_start": 2, "y_end": 4, "x_start": 2, "x_end": 4, "z_start": 3, "z_end": 4}
    p1 = DcePhase(index=1, z_start=0, z_end=10, acquisition_time="")
    p2 = DcePhase(index=2, z_start=10, z_end=20, acquisition_time="")

    vol1 = np.arange(10 * 8 * 8, dtype=np.float32).reshape(10, 8, 8)
    vol2 = (np.arange(10 * 8 * 8, dtype=np.float32) + 500.0).reshape(10, 8, 8)

    slab1 = extract_phase_z_band_full_xy(vol1, les_meta, p1, reference_phase=p1)
    slab2 = extract_phase_z_band_full_xy(vol2, les_meta, p2, reference_phase=p1)

    assert slab1.shape == (2, 8, 8)
    assert slab2.shape == (2, 8, 8)
    assert np.array_equal(slab1, vol1[3:5, :, :])
    assert np.array_equal(slab2, vol2[3:5, :, :])


def test_values_in_phase_cuboid_tight_yx() -> None:
    volume = np.ones((20, 10, 10), dtype=np.float32)
    les_meta = {"y_start": 0, "y_end": 2, "x_start": 0, "x_end": 2, "z_start": 0, "z_end": 2}
    ref = DcePhase(index=1, z_start=0, z_end=20, acquisition_time="")
    phase = DcePhase(index=1, z_start=0, z_end=20, acquisition_time="")
    values, count = values_in_phase_cuboid(volume, les_meta, phase, reference_phase=ref, normalize=False)
    assert count == 3 * 3 * 3
    assert values.size == count


def test_default_thresholds_step() -> None:
    values = default_thresholds(step=0.25)
    assert values.tolist() == [0.25, 0.5, 0.75, 1.0]
