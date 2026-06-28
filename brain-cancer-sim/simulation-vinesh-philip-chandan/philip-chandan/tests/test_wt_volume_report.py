"""Tests for WT volume reporting and UCSF workbook comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from nifti_extractor import compute_wt_volume_mm3
from wt_volume_report import (
    build_patient_volume_row,
    build_volume_report,
    format_volume_table,
    is_ucsf_dataset,
    load_ucsf_workbook,
)


def test_compute_wt_volume_mm3_counts_wt_labels_only() -> None:
    labels = np.zeros((2, 2, 2), dtype=np.int16)
    labels[0, 0, 0] = 1
    labels[0, 0, 1] = 2
    labels[0, 1, 0] = 3
    labels[0, 1, 1] = 4  # resection cavity — excluded from WT
    spacing = (1.0, 1.0, 1.0)
    assert compute_wt_volume_mm3(labels, spacing) == 3.0


def test_is_ucsf_dataset() -> None:
    assert is_ucsf_dataset("ucsf_longitudinal_glioma")
    assert is_ucsf_dataset("UCSF Longitudinal Glioma (UCSF-LPTDG)")
    assert not is_ucsf_dataset("mu_glioma_post")


def test_load_ucsf_workbook(tmp_path: Path) -> None:
    csv_path = tmp_path / "master.csv"
    csv_path.write_text(
        "subjectid,wt_volume_label1_plus_2_plus_3_t1,wt_volume_label1_plus_2_plus_3_t2,"
        "wt_change,wt_growth_pct,days_from_1st_scan_to_2nd_scan,grade,idh\n"
        "100002,8137,8365,228,2.8,183,2.0,mut\n",
        encoding="utf-8",
    )
    workbook = load_ucsf_workbook(csv_path)
    assert workbook["100002"]["wt_t1_mm3"] == 8137.0
    assert workbook["100002"]["wt_t2_mm3"] == 8365.0
    assert workbook["100002"]["wt_change_mm3"] == 228.0


def _patient(patient_id: str = "100002") -> dict[str, Any]:
    root = Path("data/raw/ucsf_alptdg") / patient_id
    return {
        "patient_id": patient_id,
        "dataset_key": "ucsf_longitudinal_glioma",
        "grade": "2",
        "idh_status": "mut",
        "timepoints": [
            {
                "label": "baseline",
                "mr_path": str(root / f"{patient_id}_time1_t1ce.nii.gz"),
                "segmentation_path": str(root / f"{patient_id}_time1_seg.nii.gz"),
            },
            {
                "label": "followup",
                "mr_path": str(root / f"{patient_id}_time2_t1ce.nii.gz"),
                "segmentation_path": str(root / f"{patient_id}_time2_seg.nii.gz"),
            },
        ],
    }


def test_build_patient_volume_row_compares_workbook(tmp_path: Path) -> None:
    patient = _patient()
    seg_baseline = tmp_path / "baseline_seg.nii.gz"
    seg_followup = tmp_path / "followup_seg.nii.gz"
    seg_baseline.touch()
    seg_followup.touch()
    patient["timepoints"][0]["segmentation_path"] = str(seg_baseline)
    patient["timepoints"][1]["segmentation_path"] = str(seg_followup)

    workbook = {
        "100002": {
            "wt_t1_mm3": 8137.0,
            "wt_t2_mm3": 8365.0,
            "wt_change_mm3": 228.0,
            "wt_growth_pct": 2.8,
            "interval_days": 183.0,
            "grade": 2.0,
            "idh": "mut",
        }
    }

    with patch("wt_volume_report.wt_volume_from_segmentation", side_effect=[8200.0, 8400.0]):
        row = build_patient_volume_row(patient, workbook=workbook)

    assert row is not None
    assert row.baseline is not None and row.followup is not None
    assert row.baseline.computed_mm3 == 8200.0
    assert row.baseline.workbook_mm3 == 8137.0
    assert row.baseline.delta_mm3 == 63.0
    assert row.computed_delta_mm3 == 200.0
    assert row.delta_delta_mm3 == pytest.approx(-28.0)


def test_build_volume_report_skips_non_ucsf_when_comparing_workbook() -> None:
    ucsf = _patient("100002")
    other = _patient("999999")
    other["dataset_key"] = "mu_glioma_post"

    with patch("wt_volume_report.build_patient_volume_row", return_value=None) as mocked:
        rows, meta = build_volume_report(
            [ucsf, other],
            compare_ucsf_workbook=True,
            workbook_csv=Path(__file__).resolve().parents[3] / "data/processed/ucsf_longitudinal_master.csv",
        )

    assert meta["compare_ucsf_workbook"] is True
    assert mocked.call_count == 1


def test_format_volume_table_includes_workbook_columns() -> None:
    from wt_volume_report import PatientVolumeRow, TimepointVolume

    row = PatientVolumeRow(
        patient_id="100002",
        dataset_key="ucsf_longitudinal_glioma",
        grade="2",
        idh_status="mut",
        interval_days=183.0,
        baseline=TimepointVolume("baseline", "/b.nii.gz", 8200.0, 8137.0, 63.0, 0.8),
        followup=TimepointVolume("followup", "/f.nii.gz", 8400.0, 8365.0, 35.0, 0.4),
        computed_delta_mm3=200.0,
        computed_growth_pct=2.4,
        workbook_delta_mm3=228.0,
        workbook_growth_pct=2.8,
        delta_delta_mm3=-28.0,
    )
    table = format_volume_table([row], compare_ucsf_workbook=True)
    assert "Baseline comp" in table
    assert "100002" in table
    assert "8,200" in table
