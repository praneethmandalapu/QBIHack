"""Unit tests for brain cohort_discovery (mocked HTTP + local fixtures)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cohort.cohort_discovery import (
    analyze_patient_imaging_tcia,
    audit_cohort,
    build_patient_report,
    find_longitudinal_local,
    find_longitudinal_tcia,
    list_tcia_patients,
    normalize_study_date,
    scan_local_dataset_inventory,
)

FIXTURES = Path(__file__).parent / "fixtures" / "cohort_discovery"
LOCAL_FIXTURES = FIXTURES / "local_inventory"


def _load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _mock_fetcher(responses: dict[str, object]):
    def fetch(url: str) -> bytes:
        for key, payload in responses.items():
            if key in url:
                if isinstance(payload, bytes):
                    return payload
                return json.dumps(payload).encode("utf-8")
        raise AssertionError(f"Unexpected URL: {url}")

    return fetch


def test_normalize_study_date_formats() -> None:
    assert normalize_study_date("20020912") == "2002-09-12"
    assert normalize_study_date("2002-09-12") == "2002-09-12"
    assert normalize_study_date("") == "unknown-study"


def test_list_tcia_patients_mocked() -> None:
    patients = _load_fixture("patients_upenn.json")
    fetcher = _mock_fetcher({"getPatient": patients})
    ids = list_tcia_patients("UPENN-GBM", fetcher=fetcher)
    assert ids == ["UPENN-GBM-00001", "UPENN-GBM-00002"]


def test_analyze_patient_longitudinal_tcia() -> None:
    series = _load_fixture("series_longitudinal.json")
    report = analyze_patient_imaging_tcia(
        "GLIO-001",
        collection="MU-Glioma-Post",
        series_list=series,
    )
    assert report["has_mri"] is True
    assert report["longitudinal"] is True
    assert report["study_dates"] == ["2020-01-15", "2020-07-20"]
    assert report["span_days"] == 187
    assert len(report["studies"]) == 2
    assert report["studies"][0]["contrast_available"] is True


def test_find_longitudinal_tcia_mocked() -> None:
    patients = _load_fixture("patients_glioma_test.json")
    series_long = _load_fixture("series_longitudinal.json")
    series_single = _load_fixture("series_single.json")

    def fetcher(url: str) -> bytes:
        if "getPatient" in url:
            return json.dumps(patients).encode("utf-8")
        if "GLIO-001" in url:
            return json.dumps(series_long).encode("utf-8")
        if "GLIO-002" in url:
            return json.dumps(series_single).encode("utf-8")
        raise AssertionError(url)

    from cohort.datasets import DatasetSpec

    test_spec = DatasetSpec(
        key="test_tcia",
        label="Test TCIA",
        disease="Glioma",
        access="tcia_nbia",
        longitudinal=True,
        segmentation="none",
        format="dicom",
        raw_dir="data/raw/test_tcia",
        tcia_collection="TEST-COL",
    )
    with patch.dict(
        "cohort.datasets.DATASET_REGISTRY",
        {"test_tcia": test_spec},
        clear=False,
    ):
        matches = find_longitudinal_tcia("test_tcia", fetcher=fetcher)
    assert len(matches) == 1
    assert matches[0]["patient_id"] == "GLIO-001"


def test_scan_local_inventory_longitudinal(tmp_path: Path) -> None:
    patient_root = tmp_path / "mu_glioma_post" / "P001"
    baseline = patient_root / "baseline"
    followup = patient_root / "followup"
    baseline.mkdir(parents=True)
    followup.mkdir(parents=True)
    (baseline / "t1c.nii.gz").write_bytes(b"x" * 1024)
    (baseline / "seg.nii.gz").write_bytes(b"y" * 512)
    (followup / "t1c.nii.gz").write_bytes(b"z" * 2048)
    (followup / "seg.nii.gz").write_bytes(b"w" * 512)

    reports = scan_local_dataset_inventory("mu_glioma_post", raw_root=tmp_path)
    assert len(reports) == 1
    report = reports[0]
    assert report["patient_id"] == "P001"
    assert report["longitudinal"] is True
    assert report["segmentation_available"] is True
    assert report["has_mri"] is True


def test_find_longitudinal_local_requires_segmentation(tmp_path: Path) -> None:
    patient_root = tmp_path / "ucsf_alptdg" / "P002"
    tp1 = patient_root / "tp1"
    tp2 = patient_root / "tp2"
    tp1.mkdir(parents=True)
    tp2.mkdir(parents=True)
    (tp1 / "flair.nii.gz").write_bytes(b"a")
    (tp2 / "flair.nii.gz").write_bytes(b"b")

    without_seg = find_longitudinal_local("ucsf_longitudinal_glioma", raw_root=tmp_path)
    assert without_seg == []

    (tp1 / "tumor_seg.nii.gz").write_bytes(b"c")
    (tp2 / "tumor_seg.nii.gz").write_bytes(b"d")
    with_seg = find_longitudinal_local("ucsf_longitudinal_glioma", raw_root=tmp_path)
    assert len(with_seg) == 1
    assert with_seg[0]["patient_id"] == "P002"


def test_find_longitudinal_ucsf_time1_time2_filenames(tmp_path: Path) -> None:
    patient_root = tmp_path / "ucsf_alptdg" / "100002"
    patient_root.mkdir(parents=True)
    for name in (
        "100002_time1_t1ce.nii.gz",
        "100002_time1_seg.nii.gz",
        "100002_time2_t1ce.nii.gz",
        "100002_time2_seg.nii.gz",
    ):
        (patient_root / name).write_bytes(b"x")

    matches = find_longitudinal_local("ucsf_longitudinal_glioma", raw_root=tmp_path)
    assert len(matches) == 1
    report = matches[0]
    assert report["patient_id"] == "100002"
    assert report["longitudinal"] is True
    assert set(report["study_dates"]) == {"baseline", "followup"}


def test_build_patient_report_local(tmp_path: Path) -> None:
    patient_root = tmp_path / "mu_glioma_post" / "P003" / "baseline"
    patient_root.mkdir(parents=True)
    (patient_root / "t1c.nii.gz").write_bytes(b"mr")
    (patient_root / "seg.nii.gz").write_bytes(b"seg")

    report = build_patient_report(
        "P003",
        dataset_key="mu_glioma_post",
        raw_root=tmp_path,
    )
    assert report["has_mri"] is True
    assert report["segmentation_available"] is True
    assert report["longitudinal"] is False
    assert "not longitudinal" in "; ".join(report["issues"])


def test_audit_cohort_stub(tmp_path: Path) -> None:
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(
        json.dumps(
            {
                "version": "test",
                "primary": [
                    {
                        "patient_id": "P004",
                        "dataset_key": "mu_glioma_post",
                        "grade": "III",
                        "idh_status": "mutant",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    patient_root = tmp_path / "raw" / "mu_glioma_post" / "P004"
    for label in ("baseline", "followup"):
        tp = patient_root / label
        tp.mkdir(parents=True)
        (tp / "t1c.nii.gz").write_bytes(b"1")
        (tp / "seg.nii.gz").write_bytes(b"2")

    result = audit_cohort(cohort_path, raw_root=tmp_path / "raw")
    assert result["cohort_version"] == "test"
    assert len(result["reports"]) == 1
    assert result["reports"][0]["ok"] is True
    assert result["primary_ok"] is True
