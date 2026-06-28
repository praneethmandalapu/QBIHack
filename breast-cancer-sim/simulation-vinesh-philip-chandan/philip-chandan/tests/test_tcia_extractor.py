"""Unit and integration tests for tcia_extractor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from tcia_extractor import (
    RAW_TCIA_DIR,
    REPO_ROOT,
    extract_volume,
    extract_volume_with_spacing,
    extract_volume_with_spacing_for_timepoint,
    iter_cohort_patients,
    list_timepoints,
    load_cohort,
    read_series,
    resolve_dicom_dir,
    resolve_study_dir,
    sort_slices,
    subtype_slug,
    validate_series,
)
from tests.conftest import _write_synthetic_slice

import sys

SPIKE_ROOT = REPO_ROOT.parent / "simulation-vinesh-philip-chandan"
sys.path.insert(0, str(SPIKE_ROOT))
from spike_paths import resolve_raw_extract_metadata, resolve_raw_extract_npy  # noqa: E402


def test_validate_series_ok(synthetic_dicom_dir: Path) -> None:
    report = validate_series(synthetic_dicom_dir)

    assert report["ok"] is True
    assert report["n_slices"] == 5
    assert report["shape"] == (5, 32, 32)
    assert report["spacing_mm"] == [2.0, 1.0, 1.0]
    assert report["errors"] == []


def test_extract_volume_shape_dtype(synthetic_dicom_dir: Path) -> None:
    volume = extract_volume(synthetic_dicom_dir)

    assert volume.shape == (5, 32, 32)
    assert volume.dtype == np.float32


def test_extract_volume_monotonic_slices(synthetic_dicom_dir: Path) -> None:
    volume = extract_volume(synthetic_dicom_dir)

    slice_means = [float(volume[index].mean()) for index in range(volume.shape[0])]
    assert slice_means == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_validate_series_empty_dir(tmp_path: Path) -> None:
    report = validate_series(tmp_path)

    assert report["ok"] is False
    assert report["n_slices"] == 0
    assert "no DICOM image slices found" in report["errors"][0]


def test_validate_series_mismatched_rows(tmp_path: Path) -> None:
    series_uid = generate_uid()
    _write_synthetic_slice(
        tmp_path,
        instance_number=1,
        z_position=0.0,
        pixel_value=10,
        rows=32,
        columns=32,
        series_uid=series_uid,
    )
    _write_synthetic_slice(
        tmp_path,
        instance_number=2,
        z_position=2.0,
        pixel_value=20,
        rows=16,
        columns=16,
        series_uid=series_uid,
    )

    report = validate_series(tmp_path)

    assert report["ok"] is False
    assert any("inconsistent Rows/Columns" in error for error in report["errors"])


def test_resolve_dicom_dir_primary_pair() -> None:
    luminal_dir = resolve_dicom_dir("TCGA-AR-A1AX", "Luminal A")
    basal_dir = resolve_dicom_dir("TCGA-AR-A1AQ", "Basal-like")

    assert luminal_dir == RAW_TCIA_DIR / "luminal_a" / "TCGA-AR-A1AX"
    assert basal_dir == RAW_TCIA_DIR / "basal" / "TCGA-AR-A1AQ"
    assert resolve_study_dir("TCGA-AR-A1AX", "Luminal A", "2002-09-12") == luminal_dir / "2002-09-12"


def test_load_cohort_primary_count() -> None:
    cohort = load_cohort()

    assert len(cohort["primary"]) == 2
    assert cohort["primary"][0]["tcga_id"] == "TCGA-AR-A1AX"
    assert cohort["primary"][1]["tcga_id"] == "TCGA-AR-A1AQ"
    assert cohort["primary"][0]["imaging"]["longitudinal"] is True
    assert len(cohort["primary"][0]["imaging"]["timepoints"]) == 2


def test_subtype_slug() -> None:
    assert subtype_slug("Luminal A") == "luminal_a"
    assert subtype_slug("Basal-like") == "basal"


def test_sort_slices_uses_instance_number() -> None:
    datasets = []
    for instance_number in [3, 1, 2]:
        dataset = Dataset()
        dataset.InstanceNumber = instance_number
        dataset.ImagePositionPatient = [0.0, 0.0, float(instance_number)]
        datasets.append(dataset)

    sorted_datasets = sort_slices(datasets)
    instance_numbers = [int(dataset.InstanceNumber) for dataset in sorted_datasets]

    assert instance_numbers == [1, 2, 3]


def test_iter_cohort_patients_primary_only() -> None:
    patients = list(iter_cohort_patients(include_backups=False))

    assert len(patients) == 2
    assert patients[0]["tcga_id"] == "TCGA-AR-A1AX"
    assert patients[1]["tcga_id"] == "TCGA-AR-A1AQ"
    assert patients[0]["use_les_mask"] is True
    assert len(list_timepoints(patients[0])) == 2


def test_read_series_filters_non_image_files(tmp_path: Path) -> None:
    series_uid = generate_uid()
    _write_synthetic_slice(
        tmp_path,
        instance_number=1,
        z_position=0.0,
        pixel_value=10,
        series_uid=series_uid,
    )
    (tmp_path / "notes.txt").write_text("not dicom", encoding="utf-8")

    datasets = read_series(tmp_path)

    assert len(datasets) == 1


def test_extract_volume_with_spacing(synthetic_dicom_dir: Path) -> None:
    volume, spacing_mm = extract_volume_with_spacing(synthetic_dicom_dir)

    assert volume.shape == (5, 32, 32)
    assert spacing_mm == [2.0, 1.0, 1.0]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("subtype", "tcga_id", "study_date"),
    [
        ("Luminal A", "TCGA-AR-A1AX", "2002-09-12"),
        ("Luminal A", "TCGA-AR-A1AX", "2003-09-24"),
        ("Basal-like", "TCGA-AR-A1AQ", "2001-11-21"),
        ("Basal-like", "TCGA-AR-A1AQ", "2003-05-07"),
    ],
)
def test_extract_primary_longitudinal_timepoint(
    subtype: str,
    tcga_id: str,
    study_date: str,
) -> None:
    dicom_dir = resolve_study_dir(tcga_id, subtype, study_date)
    if not dicom_dir.exists():
        pytest.skip(f"DICOM not downloaded: {dicom_dir}")

    volume, spacing_mm = extract_volume_with_spacing_for_timepoint(tcga_id, subtype, study_date)
    assert volume.ndim == 3
    assert volume.size > 0
    assert len(spacing_mm) == 3


@pytest.mark.integration
@pytest.mark.parametrize(
    ("subtype", "tcga_id"),
    [
        ("Luminal A", "TCGA-BH-A0BQ"),
    ],
)
def test_extract_backup_patient(subtype: str, tcga_id: str) -> None:
    dicom_dir = resolve_dicom_dir(tcga_id, subtype)
    if not dicom_dir.exists():
        pytest.skip(f"DICOM not downloaded: {dicom_dir}")

    volume, spacing_mm = extract_volume_with_spacing(dicom_dir)
    assert volume.ndim == 3
    assert volume.size > 0
    assert len(spacing_mm) == 3
    assert all(value > 0 for value in spacing_mm)


@pytest.mark.integration
@pytest.mark.parametrize(
    "slug",
    [
        "luminal_a_TCGA-AR-A1AX_baseline",
        "luminal_a_TCGA-AR-A1AX_followup",
        "basal_TCGA-AR-A1AQ_baseline",
        "basal_TCGA-AR-A1AQ_followup",
    ],
)
def test_extract_matches_saved_raw_export(slug: str) -> None:
    """Regression: SimpleITK extraction must match previously exported raw volumes."""
    sidecar_path = resolve_raw_extract_metadata(slug)
    npy_path = resolve_raw_extract_npy(slug)
    if not sidecar_path.exists() or not npy_path.exists():
        pytest.skip(f"Saved raw export not present: {slug}")

    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    dicom_dir = REPO_ROOT / meta["source_dicom_dir"]
    if not dicom_dir.exists():
        pytest.skip(f"DICOM not downloaded: {dicom_dir}")

    volume, spacing_mm = extract_volume_with_spacing(dicom_dir)
    saved = np.load(npy_path)

    assert volume.shape == tuple(saved.shape) == tuple(meta["shape"])
    assert spacing_mm == meta["spacing_mm"]
    assert np.array_equal(volume, saved)
