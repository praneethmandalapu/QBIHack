"""Extract 3D tumor volume from TCIA DICOM series."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import MRImageStorage

REPO_ROOT = Path(__file__).resolve().parents[2]
COHORT_PATH = Path(__file__).parent / "cohort.json"
RAW_TCIA_DIR = REPO_ROOT / "data" / "raw" / "tcia"

IMAGE_STORAGE_SOP_CLASSES = {
    MRImageStorage,
    "1.2.840.10008.5.1.4.1.1.2",  # CT Image Storage
    "1.2.840.10008.5.1.4.1.1.4",  # MR Image Storage
    "1.2.840.10008.5.1.4.1.1.128",  # Positron Emission Tomography Image Storage
}

SUBTYPE_SLUGS = {
    "Luminal A": "luminal_a",
    "Basal-like": "basal",
    "Luminal B": "luminal_b",
}


def subtype_slug(subtype: str) -> str:
    """Map cohort subtype label to on-disk directory slug."""
    if subtype in SUBTYPE_SLUGS:
        return SUBTYPE_SLUGS[subtype]
    return subtype.lower().replace("-", "_").replace(" ", "_")


def load_cohort(path: Path | None = None) -> dict[str, Any]:
    """Parse cohort.json and return primary, backups, and later patient records."""
    cohort_path = path or COHORT_PATH
    with cohort_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dicom_dir(tcga_id: str, subtype: str) -> Path:
    """Return expected local DICOM directory for a cohort patient."""
    return RAW_TCIA_DIR / subtype_slug(subtype) / tcga_id


def _patient_record(
    subtype: str,
    tcga_id: str,
    *,
    notes: str = "",
    imaging: dict[str, Any] | None = None,
    priority: int | None = None,
    cohort_group: str = "primary",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "subtype": subtype,
        "tcga_id": tcga_id,
        "notes": notes,
        "cohort_group": cohort_group,
        "dicom_dir": resolve_dicom_dir(tcga_id, subtype),
        "imaging": imaging or {},
        "use_les_mask": bool((imaging or {}).get("use_les_mask", False)),
    }
    if priority is not None:
        record["priority"] = priority
    return record


def iter_cohort_patients(include_backups: bool = False) -> Iterator[dict[str, Any]]:
    """Yield patient records from cohort.json with resolved dicom_dir paths."""
    cohort = load_cohort()

    for entry in cohort.get("primary", []):
        yield _patient_record(
            entry["subtype"],
            entry["tcga_id"],
            notes=entry.get("notes", ""),
            imaging=entry.get("imaging"),
            cohort_group="primary",
        )

    if not include_backups:
        return

    for subtype, backup_entries in cohort.get("backups", {}).items():
        for entry in backup_entries:
            yield _patient_record(
                subtype,
                entry["tcga_id"],
                notes=entry.get("notes", ""),
                priority=entry.get("priority"),
                cohort_group="backup",
            )

    for subtype, later_entries in cohort.get("later", {}).items():
        for entry in later_entries:
            yield _patient_record(
                subtype,
                entry["tcga_id"],
                notes=entry.get("notes", ""),
                cohort_group="later",
            )


def list_timepoints(patient: dict[str, Any]) -> list[dict[str, Any]]:
    """Return imaging timepoints for a cohort patient record."""
    imaging = patient.get("imaging", {})
    timepoints = imaging.get("timepoints", [])
    if timepoints:
        return timepoints

    if imaging.get("longitudinal"):
        return []

    return [{"label": "baseline", "study_date": None, "relative_day": 0}]


def resolve_study_dir(tcga_id: str, subtype: str, study_date: str | None = None) -> Path:
    """Return DICOM directory for a patient, optionally scoped to one study date."""
    patient_dir = resolve_dicom_dir(tcga_id, subtype)
    if study_date:
        return patient_dir / study_date
    return patient_dir


def iter_study_dirs(patient: dict[str, Any]) -> Iterator[Path]:
    """Yield local DICOM directories for each longitudinal timepoint."""
    timepoints = list_timepoints(patient)
    if len(timepoints) == 1 and timepoints[0].get("study_date") is None:
        yield patient["dicom_dir"]
        return

    for timepoint in timepoints:
        study_date = timepoint.get("study_date")
        if study_date:
            yield resolve_study_dir(patient["tcga_id"], patient["subtype"], study_date)


def _is_dicom_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in {".json", ".txt", ".md", ".xml", ".les"}:
        return False
    if path.suffix.lower() == ".dcm":
        return True
    try:
        pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def iter_dicom_files(dicom_dir: Path) -> list[Path]:
    """Recursively find DICOM image files under a directory."""
    if not dicom_dir.exists():
        return []

    files = [path for path in dicom_dir.rglob("*") if _is_dicom_file(path)]
    return sorted(files)


def _is_image_dataset(dataset: Dataset) -> bool:
    sop_class = str(getattr(dataset, "SOPClassUID", ""))
    if sop_class in IMAGE_STORAGE_SOP_CLASSES:
        return True
    modality = str(getattr(dataset, "Modality", "")).upper()
    return modality in {"MR", "CT", "PT"}


def read_series(dicom_dir: Path) -> list[Dataset]:
    """Load image DICOM datasets from a directory, skipping non-image objects."""
    datasets: list[Dataset] = []
    for path in iter_dicom_files(dicom_dir):
        dataset = pydicom.dcmread(path, force=True)
        if _is_image_dataset(dataset):
            datasets.append(dataset)
    return datasets


def _slice_sort_key(dataset: Dataset) -> tuple[float, float]:
    instance_number = float(getattr(dataset, "InstanceNumber", 0) or 0)
    position = getattr(dataset, "ImagePositionPatient", None)
    z_position = float(position[2]) if position is not None and len(position) >= 3 else 0.0
    return (instance_number, z_position)


def sort_slices(datasets: list[Dataset]) -> list[Dataset]:
    """Sort slices by InstanceNumber, falling back to ImagePositionPatient Z."""
    return sorted(datasets, key=_slice_sort_key)


def _series_shape(datasets: list[Dataset]) -> tuple[int, int, int] | None:
    if not datasets:
        return None
    rows = int(datasets[0].Rows)
    cols = int(datasets[0].Columns)
    return (len(datasets), rows, cols)


def _series_spacing_mm(datasets: list[Dataset]) -> list[float] | None:
    if not datasets:
        return None

    first = datasets[0]
    pixel_spacing = getattr(first, "PixelSpacing", None)
    slice_thickness = float(getattr(first, "SliceThickness", 1.0) or 1.0)

    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        return [slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1])]

    spacing_between = getattr(first, "SpacingBetweenSlices", None)
    if spacing_between is not None:
        slice_thickness = float(spacing_between)

    return [slice_thickness, 1.0, 1.0]


def validate_series(dicom_dir: Path) -> dict[str, Any]:
    """Validate a DICOM series for consistent dimensions and slice ordering."""
    errors: list[str] = []
    datasets = read_series(dicom_dir)

    if not datasets:
        return {
            "ok": False,
            "n_slices": 0,
            "shape": None,
            "spacing_mm": None,
            "errors": ["no DICOM image slices found"],
        }

    sorted_datasets = sort_slices(datasets)
    rows = int(sorted_datasets[0].Rows)
    cols = int(sorted_datasets[0].Columns)

    instance_numbers = [
        int(getattr(dataset, "InstanceNumber", index))
        for index, dataset in enumerate(sorted_datasets, start=1)
    ]
    if len(set(instance_numbers)) != len(instance_numbers):
        errors.append("duplicate InstanceNumber values detected")

    for dataset in sorted_datasets:
        if int(dataset.Rows) != rows or int(dataset.Columns) != cols:
            errors.append("inconsistent Rows/Columns across slices")
            break

    shape = _series_shape(sorted_datasets)
    return {
        "ok": not errors,
        "n_slices": len(sorted_datasets),
        "shape": shape,
        "spacing_mm": _series_spacing_mm(sorted_datasets),
        "errors": errors,
    }


def extract_volume(dicom_dir: Path) -> np.ndarray:
    """Load DICOM stack and return a 3D numpy volume with shape (Z, Y, X)."""
    report = validate_series(dicom_dir)
    if not report["ok"]:
        raise ValueError("; ".join(report["errors"]))

    datasets = sort_slices(read_series(dicom_dir))
    volume = np.stack(
        [dataset.pixel_array.astype(np.float32) for dataset in datasets],
        axis=0,
    )
    return volume


def extract_volume_with_spacing(dicom_dir: Path) -> tuple[np.ndarray, list[float]]:
    """Return a 3D volume and voxel spacing in mm as [dz, dy, dx]."""
    report = validate_series(dicom_dir)
    if not report["ok"]:
        raise ValueError("; ".join(report["errors"]))
    spacing_mm = report["spacing_mm"]
    if spacing_mm is None:
        raise ValueError("DICOM series is missing voxel spacing metadata")
    return extract_volume(dicom_dir), spacing_mm


def extract_volume_for_patient(tcga_id: str, subtype: str) -> np.ndarray:
    """Extract a volume for a cohort patient using the standard on-disk layout."""
    return extract_volume(resolve_dicom_dir(tcga_id, subtype))


def extract_volume_with_spacing_for_patient(
    tcga_id: str,
    subtype: str,
    *,
    study_date: str | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Extract volume and spacing for a cohort patient or one longitudinal timepoint."""
    return extract_volume_with_spacing(resolve_study_dir(tcga_id, subtype, study_date))


def extract_volume_for_timepoint(
    tcga_id: str,
    subtype: str,
    study_date: str,
) -> np.ndarray:
    """Extract a 3D volume for one longitudinal study date."""
    return extract_volume(resolve_study_dir(tcga_id, subtype, study_date))


def extract_volume_with_spacing_for_timepoint(
    tcga_id: str,
    subtype: str,
    study_date: str,
) -> tuple[np.ndarray, list[float]]:
    """Extract volume and spacing for one longitudinal study date."""
    return extract_volume_with_spacing(resolve_study_dir(tcga_id, subtype, study_date))
