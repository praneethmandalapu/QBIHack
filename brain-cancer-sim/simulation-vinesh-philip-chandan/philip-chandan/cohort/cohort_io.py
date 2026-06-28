"""Load and iterate brain imaging cohort.json entries."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cohort import COHORT_PATH, RAW_DATA_DIR, REPO_ROOT
from cohort.datasets import DatasetSpec, get_dataset


def load_cohort(path: Path | None = None) -> dict[str, Any]:
    cohort_path = path or COHORT_PATH
    with cohort_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_patient_entry(entry: dict[str, Any], *, group: str) -> dict[str, Any]:
    dataset_key = entry.get("dataset_key") or entry.get("dataset", "")
    if dataset_key in ("UCSF Longitudinal Glioma",):
        dataset_key = "ucsf_longitudinal_glioma"
    elif dataset_key in ("MU-Glioma-Post",):
        dataset_key = "mu_glioma_post"

    patient_id = entry.get("patient_id") or entry.get("tcga_id") or "TBD"
    return {
        "cohort_group": group,
        "patient_id": patient_id,
        "dataset_key": dataset_key,
        "disease": entry.get("disease", "Glioma"),
        "grade": entry.get("grade"),
        "idh_status": entry.get("idh_status"),
        "mgmt_status": entry.get("mgmt_status"),
        "timepoints": list(entry.get("timepoints", [])),
        "notes": entry.get("notes", ""),
        "backup": bool(entry.get("backup", group != "primary")),
        "raw_dir": resolve_patient_raw_dir(dataset_key, patient_id),
    }


def resolve_repo_path(raw_path: str, *, repo_root: Path | None = None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    root = repo_root or REPO_ROOT
    return root / path


def filter_cohort_timepoints(
    patient: dict[str, Any],
    selection: set[str] | None,
) -> list[dict[str, Any]]:
    """Return timepoints with MR + segmentation paths, optionally filtered by label."""
    matched: list[dict[str, Any]] = []
    for timepoint in patient.get("timepoints", []):
        label = str(timepoint.get("label", "")).lower()
        if selection is not None and label not in selection:
            continue
        if not timepoint.get("mr_path") or not timepoint.get("segmentation_path"):
            continue
        matched.append(timepoint)
    return matched


def resolve_patient_raw_dir(
    dataset_key: str,
    patient_id: str,
    *,
    raw_root: Path | None = None,
) -> Path:
    root = raw_root or RAW_DATA_DIR
    if not dataset_key or dataset_key == "TBD" or patient_id == "TBD":
        return root / "unknown" / patient_id
    try:
        spec = get_dataset(dataset_key)
        rel = spec.raw_dir.removeprefix("data/raw/")
        return root / rel / patient_id
    except KeyError:
        slug = str(dataset_key).lower().replace(" ", "_").replace("-", "_")
        return root / slug / patient_id


def iter_cohort_entries(
    cohort: dict[str, Any] | None = None,
    *,
    include_backups: bool = False,
) -> Iterator[dict[str, Any]]:
    data = cohort or load_cohort()

    for entry in data.get("primary", []):
        yield _normalize_patient_entry(entry, group="primary")

    for entry in data.get("patients", []):
        group = "backup" if entry.get("backup") else "primary"
        if group == "backup" and not include_backups:
            continue
        yield _normalize_patient_entry(entry, group=group)

    if include_backups:
        for disease, backup_entries in data.get("backups", {}).items():
            for entry in backup_entries:
                normalized = _normalize_patient_entry(entry, group="backup")
                normalized.setdefault("disease", disease)
                yield normalized
