"""Tests for patient-centric volume path helpers."""

from __future__ import annotations

import sys
from pathlib import Path

SPIKE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPIKE_ROOT))

from spike_paths import (  # noqa: E402
    longitudinal_slice_plot_path,
    pde_input_npy,
    raw_extract_npy,
    resolve_pde_input_npy,
    resolve_raw_extract_npy,
    slug_to_tcga_timepoint,
)


def test_slug_to_tcga_timepoint() -> None:
    assert slug_to_tcga_timepoint("luminal_a_TCGA-AR-A1AX_baseline") == (
        "TCGA-AR-A1AX",
        "baseline",
    )
    assert slug_to_tcga_timepoint("basal_TCGA-AR-A1AQ_followup") == (
        "TCGA-AR-A1AQ",
        "followup",
    )


def test_nested_raw_paths() -> None:
    path = raw_extract_npy("luminal_a_TCGA-AR-A1AX_baseline")
    assert path.parent.name == "TCGA-AR-A1AX"
    assert path.name == "baseline.npy"


def test_nested_pde_paths() -> None:
    path = pde_input_npy("basal_TCGA-AR-A1AQ_followup", grid_size=64)
    assert path.parent.name == "g64"
    assert path.parent.parent.name == "TCGA-AR-A1AQ"
    assert path.name == "followup.npy"


def test_resolve_prefers_nested_over_legacy(tmp_path: Path, monkeypatch) -> None:
    import spike_paths as mod

    monkeypatch.setattr(mod, "RAW_EXTRACT_PHILIP_CHANDAN", tmp_path)
    slug = "luminal_a_TCGA-AR-A1AX_baseline"
    nested = raw_extract_npy(slug)
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"nested")
    legacy = tmp_path / f"{slug}.npy"
    legacy.write_bytes(b"legacy")
    assert resolve_raw_extract_npy(slug) == nested


def test_longitudinal_qc_path() -> None:
    path = longitudinal_slice_plot_path("TCGA-AR-A1AX")
    assert path.name == "TCGA-AR-A1AX_longitudinal_mid-z-overlay.png"


def test_resolve_pde_prefers_nested(tmp_path: Path, monkeypatch) -> None:
    import spike_paths as mod

    monkeypatch.setattr(mod, "PDE_INPUT_VINESH", tmp_path)
    slug = "luminal_a_TCGA-AR-A1AX_baseline"
    nested = pde_input_npy(slug, grid_size=64)
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"nested")
    legacy = tmp_path / f"{slug}.npy"
    legacy.write_bytes(b"legacy")
    assert resolve_pde_input_npy(slug) == nested
