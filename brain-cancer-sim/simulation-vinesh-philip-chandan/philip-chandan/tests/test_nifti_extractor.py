"""Unit tests for nifti_extractor on UCSF-ALPTDG NIfTI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nifti_extractor import (
    REPO_ROOT,
    extract_spacing,
    extract_volume,
    load_expert_mask,
    resolve_ucsf_paths,
    resolve_ucsf_supplementary_paths,
    validate_nifti_pair,
)

SPIKE_PATIENT_ID = "100002"
UCSF_ROOT = REPO_ROOT / "data" / "raw" / "ucsf_alptdg"


@pytest.fixture(scope="module")
def spike_paths() -> tuple[Path, Path]:
    patient_dir = UCSF_ROOT / SPIKE_PATIENT_ID
    if not patient_dir.is_dir():
        pytest.skip(f"UCSF patient not on disk: {patient_dir}")
    return resolve_ucsf_paths(patient_dir, "baseline")


def test_validate_nifti_pair_ok(spike_paths: tuple[Path, Path]) -> None:
    mr_path, seg_path = spike_paths
    report = validate_nifti_pair(mr_path, seg_path)
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["shape"] is not None
    assert len(report["spacing_mm"]) == 3


def test_extract_volume_shape_dtype(spike_paths: tuple[Path, Path]) -> None:
    mr_path, seg_path = spike_paths
    volume = extract_volume(mr_path)
    mask = load_expert_mask(seg_path, volume.shape)

    assert volume.ndim == 3
    assert volume.dtype == np.float32
    assert mask.shape == volume.shape
    assert mask.dtype == np.float32
    assert int(mask.sum()) > 0


def test_extract_spacing_positive(spike_paths: tuple[Path, Path]) -> None:
    mr_path, _ = spike_paths
    spacing = extract_spacing(mr_path)
    assert len(spacing) == 3
    assert all(value > 0 for value in spacing)


def test_resolve_ucsf_supplementary_paths(spike_paths: tuple[Path, Path]) -> None:
    mr_path, _ = spike_paths
    extras = resolve_ucsf_supplementary_paths(mr_path.parent, "baseline")
    assert "t1ce" in extras
    assert "flair" in extras
    assert "t2" in extras
