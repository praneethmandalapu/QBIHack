"""Tests for PDE burden vs WT growth comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pde_burden_compare import (
    build_patient_burden_row,
    build_pde_burden_report,
    capture_pct,
    count_pde_tumor_voxels,
    growth_pct,
    pde_mm3_from_voxels,
)


def test_count_pde_tumor_voxels(tmp_path: Path) -> None:
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    vol[1:3, 1:3, 1:3] = 0.5
    path = tmp_path / "pde.npy"
    np.save(path, vol)
    assert count_pde_tumor_voxels(path) == 8


def test_growth_pct_and_capture() -> None:
    assert growth_pct(228.0, 8137.0) == pytest.approx(2.802, rel=1e-3)
    assert capture_pct(8137.0, 8137.0) == pytest.approx(100.0)
    assert capture_pct(None, 100.0) is None


def test_build_patient_burden_row_spike_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pde_burden_compare as mod

    pde_dir = tmp_path / "pde-input-vinesh" / "100002" / "g64"
    pde_dir.mkdir(parents=True)
    for tp, count in (("baseline", 8137), ("followup", 8365)):
        vol = np.zeros((64, 64, 64), dtype=np.float32)
        vol.ravel()[:count] = 0.5
        np.save(pde_dir / f"{tp}.npy", vol)

    monkeypatch.setattr(mod, "PDE_INPUT_VINESH", tmp_path / "pde-input-vinesh")

    wt_row = {
        "patient_id": "100002",
        "grade": "2",
        "idh_status": "mut",
        "interval_days": 183.0,
        "baseline": {"computed_mm3": 8137.0},
        "followup": {"computed_mm3": 8365.0},
        "computed_growth_pct": 2.802,
        "workbook_growth_pct": 2.802,
    }
    row = build_patient_burden_row(wt_row)
    assert row.baseline is not None and row.baseline.capture_pct == pytest.approx(100.0, rel=1e-3)
    assert row.followup is not None and row.followup.capture_pct == pytest.approx(100.0, rel=1e-3)
    assert row.pde_growth_pct == pytest.approx(2.802, rel=1e-2)
    assert row.growth_pct_delta == pytest.approx(0.0, abs=0.1)
    assert row.qc_flags == ("ok",)


def test_build_pde_burden_report_excludes_resection_cavity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pde_burden_compare as mod

    monkeypatch.setattr(mod, "_no_resection_patient_ids", lambda: frozenset({"100002"}))
    monkeypatch.setattr(mod, "_excluded_resection_patient_ids", lambda: ["100130"])

    wt_report = {
        "patients": [
            {"patient_id": "100002", "baseline": {}, "followup": {}},
            {"patient_id": "100130", "baseline": {}, "followup": {}},
        ]
    }
    report = build_pde_burden_report(wt_report)
    assert report["patient_count"] == 1
    assert report["patients"][0]["patient_id"] == "100002"
    assert report["excluded_resection_cavity_patient_ids"] == ["100130"]
    assert report["cohort_selection"] == "no_resection_cavity"


def test_build_pde_burden_report_empty_without_wt() -> None:
    report = build_pde_burden_report({"patients": []})
    assert report["patient_count"] == 0
