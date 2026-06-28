"""Tests for DCE phase splitting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

VALIDATION_DIR = Path(__file__).resolve().parents[1]
STRETCH_DIR = VALIDATION_DIR.parent / "stretch"
PHILIP_CHANDAN_DIR = VALIDATION_DIR.parent

sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(STRETCH_DIR))
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))

from dce_phases import (  # noqa: E402
    DcePhase,
    detect_cad_markers,
    infer_equal_phase_ranges,
    lesion_z_in_phase,
    mask_for_phase,
    mip_along_z,
    mip_as_volume,
    phase_ranges_from_dicom,
    resolve_phase_ranges,
    select_phases,
    split_dce_phases,
)


def test_infer_equal_phase_ranges() -> None:
    phases = infer_equal_phase_ranges(352, n_phases=4)
    assert len(phases) == 4
    assert phases[0] == DcePhase(index=1, z_start=0, z_end=88, acquisition_time="")
    assert phases[-1].z_end == 352


def test_mip_and_cad_markers() -> None:
    volume = np.zeros((10, 8, 8), dtype=np.float32)
    volume[4, 4, 4] = 100.0
    volume[4, 7, 7] = 80.0
    mip = mip_along_z(volume)
    assert mip.shape == (8, 8)
    assert float(mip[4, 4]) == 100.0
    mip_vol = mip_as_volume(volume)
    assert mip_vol.shape == (1, 8, 8)

    markers = detect_cad_markers(volume, percentile=50.0, min_distance=2, max_markers=3)
    assert markers.shape[0] >= 1
    assert markers.shape[1] == 3


def test_split_and_mask_for_phase() -> None:
    volume = np.arange(352 * 4, dtype=np.float32).reshape(352, 2, 2)
    phases = infer_equal_phase_ranges(352, n_phases=4)
    chunks = split_dce_phases(volume, phases)
    assert len(chunks) == 4
    assert chunks[0].shape == (88, 2, 2)
    assert chunks[1][0, 0, 0] == volume[88, 0, 0]

    mask = np.zeros((352, 2, 2), dtype=np.uint8)
    mask[22, 1, 1] = 1
    cropped = mask_for_phase(mask, phases[0])
    assert cropped.shape == (88, 2, 2)
    assert cropped.sum() == 1
    assert lesion_z_in_phase(mask, phases[0]) == 22
    assert lesion_z_in_phase(mask, phases[1]) is None


def test_phase_ranges_from_local_dicom() -> None:
    dicom_dir = PHILIP_CHANDAN_DIR.parents[1] / "data/raw/tcia/luminal_a/TCGA-AR-A1AX/2002-09-12"
    if not dicom_dir.is_dir():
        pytest.skip("local VIBRANT DICOM not downloaded")

    phases = phase_ranges_from_dicom(dicom_dir)
    assert len(phases) == 4
    assert phases[0].n_slices == 88
    assert phases[-1].z_end == 352
    assert phases[0].acquisition_time

    resolved = resolve_phase_ranges(volume_shape=(352, 256, 256), dicom_dir=dicom_dir)
    assert [phase.z_start for phase in resolved] == [0, 88, 176, 264]


def test_select_phases_keeps_p1_p3_only() -> None:
    phases = [
        DcePhase(index=1, z_start=0, z_end=116, acquisition_time=""),
        DcePhase(index=2, z_start=116, z_end=232, acquisition_time=""),
        DcePhase(index=3, z_start=232, z_end=348, acquisition_time=""),
        DcePhase(index=4, z_start=348, z_end=400, acquisition_time=""),
        DcePhase(index=5, z_start=400, z_end=464, acquisition_time=""),
    ]
    volumes = split_dce_phases(np.zeros((464, 2, 2), dtype=np.float32), phases)
    picked_phases, picked_volumes = select_phases(phases, volumes, (1, 2, 3))
    assert [p.index for p in picked_phases] == [1, 2, 3]
    assert len(picked_volumes) == 3
    assert picked_volumes[0].shape[0] == 116
