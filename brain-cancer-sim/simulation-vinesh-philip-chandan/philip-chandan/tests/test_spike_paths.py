"""Tests for patient-id path layout in spike_paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from spike_paths import (
    pde_input_metadata,
    pde_input_npy,
    pde_input_npy_legacy_grid,
    raw_extract_metadata,
    raw_extract_npy,
    raw_extract_npy_legacy,
    resolve_pde_input_npy,
    resolve_raw_extract_npy,
    slug_to_patient_timepoint,
)


def test_slug_to_patient_timepoint() -> None:
    assert slug_to_patient_timepoint("glioma_ucsf_100002_baseline") == ("100002", "baseline")
    assert slug_to_patient_timepoint("glioma_ucsf_100002_followup") == ("100002", "followup")


def test_raw_extract_paths_use_patient_id_layout() -> None:
    slug = "glioma_ucsf_100002_followup"
    npy = raw_extract_npy(slug)
    assert npy.parent.name == "100002"
    assert npy.name == "followup.npy"
    assert raw_extract_metadata(slug).name == "followup.json"


def test_pde_input_paths_use_patient_id_layout() -> None:
    slug = "glioma_ucsf_100002_followup"
    npy = pde_input_npy(slug, grid_size=64)
    assert npy.parent.name == "g64"
    assert npy.parent.parent.name == "100002"
    assert npy.name == "followup.npy"
    assert pde_input_metadata(slug, grid_size=64).name == "followup.json"


def test_resolve_raw_extract_prefers_patient_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import spike_paths as sp

    monkeypatch.setattr(sp, "RAW_EXTRACT_PHILIP_CHANDAN", tmp_path)
    slug = "glioma_ucsf_100002_baseline"
    legacy = raw_extract_npy_legacy(slug)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"legacy")
    assert resolve_raw_extract_npy(slug) == legacy

    primary = raw_extract_npy(slug)
    primary.parent.mkdir(parents=True)
    primary.write_bytes(b"new")
    assert resolve_raw_extract_npy(slug) == primary


def test_resolve_pde_input_prefers_patient_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import spike_paths as sp

    monkeypatch.setattr(sp, "PDE_INPUT_VINESH", tmp_path)
    slug = "glioma_ucsf_100002_baseline"
    legacy_grid = pde_input_npy_legacy_grid(slug, grid_size=64)
    legacy_grid.parent.mkdir(parents=True, exist_ok=True)
    legacy_grid.write_bytes(b"legacy-grid")
    assert resolve_pde_input_npy(slug, grid_size=64) == legacy_grid

    primary = pde_input_npy(slug, grid_size=64)
    primary.parent.mkdir(parents=True)
    primary.write_bytes(b"new")
    assert resolve_pde_input_npy(slug, grid_size=64) == primary
