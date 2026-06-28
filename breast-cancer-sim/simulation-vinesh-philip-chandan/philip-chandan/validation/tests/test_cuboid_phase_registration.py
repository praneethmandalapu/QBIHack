"""Tests for P1 z-band slab registration."""

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

from cuboid_phase_registration import align_phase_z_bands_to_p1  # noqa: E402
from dce_phases import DcePhase  # noqa: E402


def test_register_rigid_slab_uses_inplane_when_z_thin() -> None:
    from cuboid_phase_registration import register_rigid_slab

    fixed = np.zeros((7, 32, 32), dtype=np.float32)
    fixed[3, 10:22, 10:22] = 1.0
    moving = np.roll(fixed[3:4], shift=2, axis=2)
    aligned, metrics = register_rigid_slab(
        fixed,
        moving,
        spacing_zyx=(3.0, 1.0, 1.0),
        moving_phase=4,
        number_of_iterations=40,
    )
    assert aligned.shape == moving.shape
    assert metrics.ncc_after >= metrics.ncc_before


def test_align_z_band_improves_ncc_after_shift() -> None:
    les_meta = {"y_start": 0, "y_end": 3, "x_start": 0, "x_end": 3, "z_start": 2, "z_end": 5}
    p1 = DcePhase(index=1, z_start=0, z_end=8, acquisition_time="")
    p2 = DcePhase(index=2, z_start=8, z_end=16, acquisition_time="")

    rng = np.random.default_rng(0)
    base = rng.random((4, 16, 16), dtype=np.float32)
    vol1 = np.zeros((8, 16, 16), dtype=np.float32)
    vol1[2:6] = base
    vol2 = np.zeros((8, 16, 16), dtype=np.float32)
    vol2[2:6] = np.roll(base, shift=2, axis=2)

    result = align_phase_z_bands_to_p1(
        [vol1, vol2],
        [p1, p2],
        les_meta,
        spacing_mm=(3.0, 1.0, 1.0),
        number_of_iterations=80,
    )

    assert result.slabs_raw[1].shape == (4, 16, 16)
    p2_metrics = result.metrics[1]
    assert p2_metrics.ncc_after >= p2_metrics.ncc_before
