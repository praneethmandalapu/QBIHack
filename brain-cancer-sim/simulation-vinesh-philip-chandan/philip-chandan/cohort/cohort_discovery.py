"""Brain imaging cohort discovery for longitudinal glioma MRI selection.

Queries TCIA NBIA where collections are indexed, scans local NIfTI inventory under
data/raw/, and validates entries in cohort.json before download or patient lock.

Usage (from brain-cancer-sim/):
  python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py list-datasets
  python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py audit
  python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py find-longitudinal --dataset mu_glioma_post
  python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py scan-local --dataset ucsf_longitudinal_glioma
  python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py discover-ucsf
  python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py recommend-pair
  python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py show PATIENT_ID --dataset mu_glioma_post --json

Discovery log (UCSF-ALPTDG, refreshed via ``discover-ucsf``):
  - Imaging eligible: 2 visit timepoints (baseline + followup) with expert ``*_seg.nii.gz``
    masks on disk under ``data/raw/ucsf_alptdg/<patient_id>/``.
  - Same-ID genomics eligible: imaging eligible **and** IDH + grade present for the same
    ``subjectid`` in ``data/processed/ucsf_longitudinal_master.csv`` (from
    ``models-praneeth/clean_ucsf.py``). This is UCSF-native molecular data, not TCGA barcodes.
  - ``cohort.json`` holds a **curated** primary/backup pair (7 patients as of rev1-ucsf),
    not the full eligible inventory — see ``cohort/cohort_discovery_ucsf.json`` for the
    complete audited list and ``missing_from_cohort_json`` IDs.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cohort import COHORT_DISCOVERY_UCSF_PATH, COHORT_PATH, RAW_DATA_DIR, REPO_ROOT, UCSF_MASTER_CSV
from cohort.cohort_io import iter_cohort_entries, load_cohort, resolve_patient_raw_dir
from cohort.datasets import DatasetSpec, PREFERRED_DATASET_KEYS, get_dataset, iter_datasets
from nifti_extractor import resection_cavity_from_segmentation

NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

MR_SERIES_HINTS = ("t1", "flair", "t2", "dwi", "perf", "vibrant", "post", "contrast")
SEG_SUFFIXES = ("seg", "segmentation", "mask", "label")
MR_SUFFIXES = (".nii.gz", ".nii", ".mha")
TIMEPOINT_DIR_NAMES = ("baseline", "followup", "follow_up", "tp1", "tp2", "visit1", "visit2")
# UCSF-ALPTDG encodes visits in filenames: {patient_id}_time1_t1ce.nii.gz
TIMEPOINT_STEM_ALIASES = (
    ("time1", "baseline"),
    ("time2", "followup"),
)
UCSF_VISIT_LABELS = frozenset({"baseline", "followup"})

HttpFetcher = Callable[[str], bytes]


def _default_fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def _nbia_url(path: str, params: dict[str, str] | None = None) -> str:
    query = urllib.parse.urlencode(params or {})
    url = f"{NBIA_BASE}/{path}"
    if query:
        url = f"{url}?{query}"
    return url


def normalize_study_date(raw: str) -> str:
    """Normalize study dates to YYYY-MM-DD."""
    value = str(raw).strip()
    if not value:
        return "unknown-study"
    if len(value) >= 10 and value[4] == "-" and value[7] == "-":
        return value[:10]
    if len(value) >= 8 and value[:8].isdigit():
        digits = value[:8]
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return value[:10]


def _parse_date(study_date: str) -> datetime | None:
    if study_date == "unknown-study":
        return None
    try:
        return datetime.strptime(study_date, "%Y-%m-%d")
    except ValueError:
        return None


def _span_days(study_dates: list[str]) -> int | None:
    parsed = sorted(date for date in (_parse_date(value) for value in study_dates) if date is not None)
    if len(parsed) < 2:
        return None
    return (parsed[-1] - parsed[0]).days


def _series_study_date(series: dict[str, Any]) -> str:
    return normalize_study_date(str(series.get("StudyDate", "")))


def group_series_by_normalized_study(series_list: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in series_list:
        study_date = _series_study_date(entry)
        grouped.setdefault(study_date, []).append(entry)
    return grouped


def _nbia_patient_id(entry: dict[str, Any]) -> str | None:
    for key in ("PatientId", "PatientID"):
        value = entry.get(key)
        if value:
            return str(value)
    return None


def list_tcia_patients(
    collection: str,
    *,
    fetcher: HttpFetcher | None = None,
) -> list[str]:
    fetch = fetcher or _default_fetch
    payload = fetch(_nbia_url("getPatient", {"Collection": collection}))
    if not payload.strip():
        return []
    patients = json.loads(payload)
    ids = [_nbia_patient_id(entry) for entry in patients]
    return sorted(patient_id for patient_id in ids if patient_id)


def list_tcia_mr_series(
    patient_id: str,
    collection: str,
    *,
    fetcher: HttpFetcher | None = None,
) -> list[dict[str, Any]]:
    fetch = fetcher or _default_fetch
    payload = fetch(
        _nbia_url(
            "getSeries",
            {"Collection": collection, "PatientID": patient_id, "Modality": "MR"},
        )
    )
    if not payload.strip():
        return []
    return json.loads(payload)


def _series_size_bytes(series: dict[str, Any]) -> int:
    raw_size = series.get("FileSize", 0)
    try:
        return int(raw_size or 0)
    except (TypeError, ValueError):
        return 0


def _series_summary(series: dict[str, Any]) -> dict[str, Any]:
    return {
        "SeriesDescription": series.get("SeriesDescription", ""),
        "ImageCount": int(series.get("ImageCount", 0) or 0),
        "FileSize": _series_size_bytes(series),
        "SeriesInstanceUID": series.get("SeriesInstanceUID", ""),
    }


def _is_contrast_series(series: dict[str, Any]) -> bool:
    description = str(series.get("SeriesDescription", "")).lower()
    return any(token in description for token in ("+c", "post", "t1c", "contrast", "gad"))


def _pick_best_series(series_list: list[dict[str, Any]], *, prefer_contrast: bool = True) -> dict[str, Any] | None:
    if not series_list:
        return None

    def score(series: dict[str, Any]) -> tuple[int, int, int]:
        description = str(series.get("SeriesDescription", "")).lower()
        contrast = int(prefer_contrast and _is_contrast_series(series))
        hint = int(any(token in description for token in MR_SERIES_HINTS))
        return (contrast, hint, _series_size_bytes(series))

    return max(series_list, key=score)


def analyze_patient_imaging_tcia(
    patient_id: str,
    *,
    collection: str,
    series_list: list[dict[str, Any]] | None = None,
    fetcher: HttpFetcher | None = None,
) -> dict[str, Any]:
    if series_list is None:
        series_list = list_tcia_mr_series(patient_id, collection, fetcher=fetcher)

    grouped = group_series_by_normalized_study(series_list)
    study_dates = sorted(date for date in grouped if date != "unknown-study")
    longitudinal = len(study_dates) >= 2

    studies: list[dict[str, Any]] = []
    total_download_size = 0
    for study_date in study_dates:
        best = _pick_best_series(grouped[study_date], prefer_contrast=True)
        if best is None:
            continue
        size_bytes = _series_size_bytes(best)
        total_download_size += size_bytes
        studies.append(
            {
                "study_date": study_date,
                "best_series": _series_summary(best),
                "contrast_available": _is_contrast_series(best),
                "download_size_bytes": size_bytes,
            }
        )

    return {
        "patient_id": patient_id,
        "dataset_key": None,
        "source": "tcia_nbia",
        "tcia_collection": collection,
        "has_mri": bool(series_list),
        "series_count": len(series_list),
        "longitudinal": longitudinal,
        "study_dates": study_dates,
        "span_days": _span_days(study_dates),
        "studies": studies,
        "segmentation_available": False,
        "total_download_size_bytes": total_download_size,
    }


def _looks_like_nifti(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in MR_SUFFIXES)


def _looks_like_segmentation(path: Path) -> bool:
    name = path.name.lower()
    return _looks_like_nifti(path) and any(token in name for token in SEG_SUFFIXES)


def _looks_like_mr_volume(path: Path) -> bool:
    if not _looks_like_nifti(path):
        return False
    return not _looks_like_segmentation(path)


def _infer_timepoint_label(path: Path, patient_root: Path) -> str:
    relative_parts = path.relative_to(patient_root).parts
    if len(relative_parts) >= 2:
        candidate = relative_parts[0]
        if candidate.lower() not in ("mr", "seg", "masks", "images"):
            return candidate
    stem = path.stem.lower().replace(".nii", "")
    if "subtraction" in stem:
        return "subtraction"
    for token, label in TIMEPOINT_STEM_ALIASES:
        if token in stem:
            return label
    for token in TIMEPOINT_DIR_NAMES:
        if token in stem:
            return token
    return "unknown-timepoint"


def scan_patient_local_inventory(
    patient_dir: Path,
    *,
    patient_id: str,
    dataset: DatasetSpec,
) -> dict[str, Any]:
    if not patient_dir.is_dir():
        return {
            "patient_id": patient_id,
            "dataset_key": dataset.key,
            "source": "local",
            "has_mri": False,
            "longitudinal": False,
            "study_dates": [],
            "span_days": None,
            "studies": [],
            "segmentation_available": False,
            "timepoints": [],
            "total_download_size_bytes": 0,
            "issues": ["patient directory missing"],
            "ok": False,
        }

    timepoint_map: dict[str, dict[str, Any]] = {}
    total_size = 0

    for path in sorted(patient_dir.rglob("*")):
        if not path.is_file():
            continue
        total_size += path.stat().st_size
        label = _infer_timepoint_label(path, patient_dir)
        bucket = timepoint_map.setdefault(
            label,
            {
                "label": label,
                "mr_paths": [],
                "segmentation_paths": [],
            },
        )
        if _looks_like_segmentation(path):
            bucket["segmentation_paths"].append(str(path))
        elif _looks_like_mr_volume(path):
            bucket["mr_paths"].append(str(path))

    timepoints = []
    study_dates: list[str] = []
    for label, bucket in sorted(timepoint_map.items()):
        has_mr = bool(bucket["mr_paths"])
        has_seg = bool(bucket["segmentation_paths"])
        if not has_mr and not has_seg:
            continue
        study_date = label if label != "unknown-timepoint" else f"tp-{len(timepoints) + 1}"
        study_dates.append(study_date)
        timepoints.append(
            {
                "label": label,
                "study_date": study_date,
                "mr_paths": bucket["mr_paths"],
                "segmentation_paths": bucket["segmentation_paths"],
                "segmentation_available": has_seg,
            }
        )

    longitudinal = len(timepoints) >= 2
    segmentation_available = any(tp["segmentation_available"] for tp in timepoints)

    return {
        "patient_id": patient_id,
        "dataset_key": dataset.key,
        "source": "local",
        "has_mri": any(tp["mr_paths"] for tp in timepoints),
        "longitudinal": longitudinal,
        "study_dates": study_dates,
        "span_days": _span_days(study_dates) if all(d.startswith("tp-") is False for d in study_dates) else None,
        "studies": [
            {
                "study_date": tp["study_date"],
                "mr_paths": tp["mr_paths"],
                "segmentation_paths": tp["segmentation_paths"],
                "segmentation_available": tp["segmentation_available"],
            }
            for tp in timepoints
        ],
        "timepoints": timepoints,
        "segmentation_available": segmentation_available,
        "total_download_size_bytes": total_size,
        "issues": [],
        "ok": True,
    }


def scan_local_dataset_inventory(
    dataset_key: str,
    *,
    raw_root: Path | None = None,
) -> list[dict[str, Any]]:
    dataset = get_dataset(dataset_key)
    dataset_root = (raw_root or RAW_DATA_DIR) / Path(dataset.raw_dir).name
    if not dataset_root.is_dir():
        return []

    reports: list[dict[str, Any]] = []
    for patient_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        report = scan_patient_local_inventory(
            patient_dir,
            patient_id=patient_dir.name,
            dataset=dataset,
        )
        reports.append(report)
    return reports


def find_longitudinal_local(
    dataset_key: str,
    *,
    require_segmentation: bool = True,
    raw_root: Path | None = None,
) -> list[dict[str, Any]]:
    matches = [
        report
        for report in scan_local_dataset_inventory(dataset_key, raw_root=raw_root)
        if report["longitudinal"] and report["has_mri"]
    ]
    if require_segmentation:
        matches = [report for report in matches if report["segmentation_available"]]
    matches.sort(key=lambda report: (report.get("span_days") or 0, report["patient_id"]), reverse=True)
    return matches


def find_longitudinal_tcia(
    dataset_key: str,
    *,
    fetcher: HttpFetcher | None = None,
) -> list[dict[str, Any]]:
    dataset = get_dataset(dataset_key)
    if dataset.access != "tcia_nbia" or not dataset.tcia_collection:
        return []

    matches: list[dict[str, Any]] = []
    for patient_id in list_tcia_patients(dataset.tcia_collection, fetcher=fetcher):
        report = analyze_patient_imaging_tcia(
            patient_id,
            collection=dataset.tcia_collection,
            fetcher=fetcher,
        )
        report["dataset_key"] = dataset.key
        if report["longitudinal"]:
            matches.append(report)

    matches.sort(key=lambda report: (report.get("span_days") or 0, report["patient_id"]), reverse=True)
    return matches


def find_longitudinal_patients(
    dataset_key: str,
    *,
    require_segmentation: bool = True,
    fetcher: HttpFetcher | None = None,
    raw_root: Path | None = None,
) -> list[dict[str, Any]]:
    dataset = get_dataset(dataset_key)

    if dataset.access == "tcia_nbia":
        return find_longitudinal_tcia(dataset_key, fetcher=fetcher)

    local_matches = find_longitudinal_local(
        dataset_key,
        require_segmentation=require_segmentation,
        raw_root=raw_root,
    )
    if local_matches:
        return local_matches

    return []


def _is_ucsf_subject_id(patient_id: str) -> bool:
    return patient_id.isdigit()


def _has_visit_segmentation_masks(report: dict[str, Any]) -> bool:
    timepoints = report.get("timepoints") or []
    visit_labels = {
        tp["label"]
        for tp in timepoints
        if tp.get("segmentation_available") and tp.get("label") in UCSF_VISIT_LABELS
    }
    return len(visit_labels) >= 2


def _pick_segmentation_path(timepoint: dict[str, Any]) -> Path | None:
    paths = [Path(value) for value in timepoint.get("segmentation_paths") or []]
    if not paths:
        return None
    seg_paths = [path for path in paths if "seg" in path.name.lower()]
    return seg_paths[0] if seg_paths else paths[0]


def scan_resection_cavity_for_report(report: dict[str, Any]) -> dict[str, Any]:
    """Add per-visit label-4 (resection cavity) stats from local expert segmentations."""
    visits: dict[str, Any] = {}
    for visit_label in sorted(UCSF_VISIT_LABELS):
        timepoint = next(
            (tp for tp in (report.get("timepoints") or []) if tp.get("label") == visit_label),
            None,
        )
        if timepoint is None:
            visits[visit_label] = {
                "has_resection_cavity": None,
                "resection_cavity_mm3": None,
                "labels_present": [],
                "segmentation_path": None,
            }
            continue
        seg_path = _pick_segmentation_path(timepoint)
        if seg_path is None or not seg_path.is_file():
            visits[visit_label] = {
                "has_resection_cavity": None,
                "resection_cavity_mm3": None,
                "labels_present": [],
                "segmentation_path": str(seg_path) if seg_path else None,
            }
            continue
        stats = resection_cavity_from_segmentation(seg_path)
        visits[visit_label] = {
            "has_resection_cavity": stats["has_resection_cavity"],
            "resection_cavity_mm3": stats["resection_cavity_mm3"],
            "labels_present": stats["labels_present"],
            "segmentation_path": str(seg_path),
        }

    baseline = visits.get("baseline", {})
    followup = visits.get("followup", {})
    return {
        "resection_cavity": visits,
        "has_resection_cavity_baseline": baseline.get("has_resection_cavity"),
        "has_resection_cavity_followup": followup.get("has_resection_cavity"),
        "has_resection_cavity_any_visit": any(
            visit.get("has_resection_cavity") for visit in visits.values()
        ),
    }


def passes_resection_exclusion(
    resection: dict[str, Any],
    *,
    exclude_baseline: bool = False,
    exclude_followup: bool = False,
) -> bool:
    """Return True when patient passes resection-cavity filters (not excluded)."""
    if exclude_baseline and resection.get("has_resection_cavity_baseline"):
        return False
    if exclude_followup and resection.get("has_resection_cavity_followup"):
        return False
    return True


def load_ucsf_master_lookup(
    master_csv: Path | None = None,
) -> dict[str, dict[str, Any]]:
    path = master_csv or UCSF_MASTER_CSV
    if not path.is_file():
        return {}

    import csv

    lookup: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            subject_id = str(row.get("subjectid", "")).strip()
            if not subject_id:
                continue
            lookup[subject_id] = row
    return lookup


def discover_ucsf_cohort(
    *,
    cohort_path: Path | None = None,
    master_csv: Path | None = None,
    raw_root: Path | None = None,
    exclude_resection_baseline: bool = False,
    exclude_resection_followup: bool = False,
) -> dict[str, Any]:
    """Audit UCSF-ALPTDG patients against imaging + same-ID genomics (IDH/grade) criteria."""
    master_lookup = load_ucsf_master_lookup(master_csv)
    imaging_reports = find_longitudinal_local(
        "ucsf_longitudinal_glioma",
        require_segmentation=True,
        raw_root=raw_root,
    )

    cohort = load_cohort(cohort_path)
    cohort_ids = {
        str(entry["patient_id"])
        for entry in iter_cohort_entries(cohort, include_backups=True)
    }

    patients: list[dict[str, Any]] = []
    imaging_eligible_ids: list[str] = []
    genomics_eligible_ids: list[str] = []
    resection_filtered_ids: list[str] = []

    for report in imaging_reports:
        patient_id = str(report["patient_id"])
        if not _is_ucsf_subject_id(patient_id):
            continue
        if not _has_visit_segmentation_masks(report):
            continue

        resection = scan_resection_cavity_for_report(report)
        if not passes_resection_exclusion(
            resection,
            exclude_baseline=exclude_resection_baseline,
            exclude_followup=exclude_resection_followup,
        ):
            resection_filtered_ids.append(patient_id)
            continue

        clinical = master_lookup.get(patient_id, {})
        idh = (clinical.get("idh") or "").strip()
        grade = (clinical.get("grade") or "").strip()
        has_same_id_genomics = bool(
            idh and grade and idh.lower() != "nan" and grade.lower() != "nan"
        )

        entry = {
            "patient_id": patient_id,
            "imaging_eligible": True,
            "same_id_genomics_idh_grade": has_same_id_genomics,
            "in_cohort_json": patient_id in cohort_ids,
            "study_dates": report.get("study_dates", []),
            "span_days": report.get("span_days"),
            "idh_status": idh or None,
            "grade": grade or None,
            "mgmt": clinical.get("mgmt") or None,
            "who_2021_diagnosis": clinical.get("who_2021_diagnosis") or None,
            "wt_growth_pct": clinical.get("wt_growth_pct") or None,
            **resection,
        }
        patients.append(entry)
        imaging_eligible_ids.append(patient_id)
        if has_same_id_genomics:
            genomics_eligible_ids.append(patient_id)

    patients.sort(key=lambda row: row["patient_id"])
    imaging_set = set(imaging_eligible_ids)
    genomics_set = set(genomics_eligible_ids)
    missing_from_cohort = sorted(genomics_set - cohort_ids)

    master_path = master_csv or UCSF_MASTER_CSV
    try:
        genomics_source = str(master_path.relative_to(REPO_ROOT))
    except ValueError:
        genomics_source = str(master_path)

    exclusion_notes: list[str] = []
    if exclude_resection_baseline:
        exclusion_notes.append("exclude label 4 (resection cavity) at baseline")
    if exclude_resection_followup:
        exclusion_notes.append("exclude label 4 (resection cavity) at follow-up")

    return {
        "dataset_key": "ucsf_longitudinal_glioma",
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "criteria": {
            "imaging": (
                "UCSF-ALPTDG local NIfTI: baseline + followup visit folders with expert "
                "segmentation masks (*_time1_seg.nii.gz, *_time2_seg.nii.gz)"
            ),
            "same_id_genomics_idh_grade": (
                "Same UCSF subjectid with IDH + grade in "
                "data/processed/ucsf_longitudinal_master.csv (UCSF clinical workbook; "
                "not TCGA expression data)"
            ),
            "resection_cavity_exclusions": exclusion_notes or None,
        },
        "counts": {
            "imaging_two_timepoint_expert_seg": len(imaging_set),
            "same_id_genomics_idh_grade": len(genomics_set),
            "in_cohort_json": len(cohort_ids & genomics_set),
            "missing_from_cohort_json": len(missing_from_cohort),
            "excluded_resection_cavity": len(resection_filtered_ids),
        },
        "excluded_resection_cavity_patient_ids": sorted(resection_filtered_ids),
        "genomics_source": genomics_source,
        "cohort_json_patient_ids": sorted(cohort_ids),
        "missing_from_cohort_json": missing_from_cohort,
        "patients": patients,
        "notes": (
            "cohort.json is curated for demo pairs/backups, not a full inventory. "
            "Regenerate this file after downloading UCSF NIfTI or refreshing clean_ucsf.py output. "
            "Label 4 (RC) = resection cavity; excluded from WT (labels 1+2+3) but present in many "
            "postoperative UCSF segmentations."
        ),
    }


def write_ucsf_discovery_json(
    result: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    path = output_path or COHORT_DISCOVERY_UCSF_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def print_ucsf_discovery_summary(result: dict[str, Any]) -> None:
    counts = result["counts"]
    print(f"UCSF discovery ({result['dataset_key']})")
    print(f"  Imaging (2 TP + expert seg): {counts['imaging_two_timepoint_expert_seg']}")
    print(f"  Same-ID genomics (IDH+grade): {counts['same_id_genomics_idh_grade']}")
    if counts.get("excluded_resection_cavity"):
        print(f"  Excluded (label 4 resection cavity): {counts['excluded_resection_cavity']}")
        criteria = result.get("criteria", {}).get("resection_cavity_exclusions")
        if criteria:
            print(f"    filters: {', '.join(criteria)}")
    print(f"  Listed in cohort.json: {counts['in_cohort_json']}")
    print(f"  Genomics-eligible but not in cohort.json: {counts['missing_from_cohort_json']}")
    if result.get("missing_from_cohort_json"):
        sample = ", ".join(result["missing_from_cohort_json"][:12])
        suffix = "..." if len(result["missing_from_cohort_json"]) > 12 else ""
        print(f"    sample: {sample}{suffix}")


def build_patient_report(
    patient_id: str,
    *,
    dataset_key: str,
    cohort_grade: str | None = None,
    cohort_idh: str | None = None,
    imaging: dict[str, Any] | None = None,
    fetcher: HttpFetcher | None = None,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    dataset = get_dataset(dataset_key)

    if imaging is None:
        patient_dir = resolve_patient_raw_dir(dataset_key, patient_id, raw_root=raw_root)
        if patient_dir.is_dir():
            imaging = scan_patient_local_inventory(patient_dir, patient_id=patient_id, dataset=dataset)
        elif dataset.access == "tcia_nbia" and dataset.tcia_collection:
            imaging = analyze_patient_imaging_tcia(
                patient_id,
                collection=dataset.tcia_collection,
                fetcher=fetcher,
            )
            imaging["dataset_key"] = dataset.key
        else:
            imaging = {
                "patient_id": patient_id,
                "dataset_key": dataset.key,
                "source": dataset.access,
                "has_mri": False,
                "longitudinal": False,
                "study_dates": [],
                "span_days": None,
                "studies": [],
                "segmentation_available": False,
                "total_download_size_bytes": 0,
            }

    issues: list[str] = list(imaging.get("issues", []))
    if not imaging.get("has_mri"):
        issues.append("missing MRI volumes")
    if not imaging.get("longitudinal"):
        issues.append("not longitudinal (fewer than 2 timepoints / study dates)")
    if dataset.segmentation != "none" and not imaging.get("segmentation_available"):
        issues.append("missing expert segmentation mask(s)")

    download_ready = dataset.access in {"tcia_nbia", "tcia_nifti", "ucsf_portal"}
    if not imaging.get("has_mri") and download_ready:
        issues.append(f"download required — see: {dataset.download_notes or dataset.portal_url or 'COHORT.md'}")

    return {
        **imaging,
        "cohort_grade": cohort_grade,
        "cohort_idh": cohort_idh,
        "dataset_label": dataset.label,
        "download_notes": dataset.download_notes,
        "portal_url": dataset.portal_url,
        "issues": issues,
        "ok": not issues,
    }


def audit_cohort(
    cohort_path: Path | None = None,
    *,
    include_backups: bool = False,
    fetcher: HttpFetcher | None = None,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    cohort = load_cohort(cohort_path)
    reports: list[dict[str, Any]] = []

    for entry in iter_cohort_entries(cohort, include_backups=include_backups):
        dataset_key = entry["dataset_key"]
        if not dataset_key or dataset_key == "TBD":
            reports.append(
                {
                    "patient_id": entry["patient_id"],
                    "cohort_group": entry["cohort_group"],
                    "dataset_key": dataset_key,
                    "ok": False,
                    "issues": ["dataset not selected"],
                }
            )
            continue

        try:
            report = build_patient_report(
                entry["patient_id"],
                dataset_key=dataset_key,
                cohort_grade=entry.get("grade"),
                cohort_idh=entry.get("idh_status"),
                fetcher=fetcher,
                raw_root=raw_root,
            )
        except KeyError as exc:
            report = {
                "patient_id": entry["patient_id"],
                "cohort_group": entry["cohort_group"],
                "dataset_key": dataset_key,
                "ok": False,
                "issues": [str(exc)],
            }
        else:
            report["cohort_group"] = entry["cohort_group"]
        reports.append(report)

    primary_reports = [report for report in reports if report.get("cohort_group") == "primary"]
    primary_ok = bool(primary_reports) and all(report.get("ok") for report in primary_reports)

    return {
        "cohort_version": cohort.get("version"),
        "reports": reports,
        "primary_ok": primary_ok,
    }


def recommend_pair(
    *,
    fetcher: HttpFetcher | None = None,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    access_notes: dict[str, str] = {}

    for dataset_key in PREFERRED_DATASET_KEYS:
        dataset = get_dataset(dataset_key)
        access_notes[dataset_key] = dataset.download_notes
        candidates[dataset_key] = find_longitudinal_patients(
            dataset_key,
            require_segmentation=True,
            fetcher=fetcher,
            raw_root=raw_root,
        )

    def pick_best(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not reports:
            return None
        return max(
            reports,
            key=lambda report: (
                int(report.get("segmentation_available", False)),
                report.get("span_days") or 0,
                report.get("total_download_size_bytes", 0),
            ),
        )

    recommended: dict[str, Any] = {}
    for dataset_key in PREFERRED_DATASET_KEYS:
        best = pick_best(candidates[dataset_key])
        recommended[dataset_key] = best

    spike = None
    for dataset_key in PREFERRED_DATASET_KEYS:
        if candidates[dataset_key]:
            spike = {"dataset_key": dataset_key, "patient": candidates[dataset_key][0]}
            break

    return {
        "preferred_datasets": list(PREFERRED_DATASET_KEYS),
        "candidates": {key: len(values) for key, values in candidates.items()},
        "recommended": recommended,
        "spike_patient": spike,
        "access_notes": access_notes,
    }


def list_datasets_table(*, preferred_only: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in iter_datasets(preferred_only=preferred_only):
        rows.append(
            {
                "key": dataset.key,
                "label": dataset.label,
                "disease": dataset.disease,
                "access": dataset.access,
                "longitudinal": dataset.longitudinal,
                "segmentation": dataset.segmentation,
                "format": dataset.format,
                "tcia_collection": dataset.tcia_collection,
                "portal_url": dataset.portal_url,
                "raw_dir": dataset.raw_dir,
                "growth_model_score": dataset.growth_model_score,
            }
        )
    return rows


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.2f} GB"
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.1f} MB"
    if num_bytes >= 1_000:
        return f"{num_bytes / 1_000:.1f} KB"
    return f"{num_bytes} B"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))

    print(render(headers))
    print(render(["-" * width for width in widths]))
    for row in rows:
        print(render(row))


def print_datasets_table(rows: list[dict[str, Any]]) -> None:
    table_rows = [
        [
            row["key"],
            row["label"],
            row["access"],
            "yes" if row["longitudinal"] else "no",
            row["segmentation"],
            row["raw_dir"],
        ]
        for row in rows
    ]
    _print_table(["key", "label", "access", "long", "segmentation", "raw_dir"], table_rows)


def print_audit_table(result: dict[str, Any]) -> None:
    rows: list[list[str]] = []
    for report in result["reports"]:
        study_summary = ", ".join(report.get("study_dates", [])) or "-"
        rows.append(
            [
                report.get("cohort_group", "-"),
                report.get("patient_id", "-"),
                report.get("dataset_key") or report.get("dataset_label") or "-",
                "yes" if report.get("has_mri") else "no",
                "yes" if report.get("longitudinal") else "no",
                "yes" if report.get("segmentation_available") else "no",
                study_summary,
                str(report.get("span_days") if report.get("span_days") is not None else "-"),
                "ok" if report.get("ok") else "; ".join(report.get("issues", ["failed"])),
            ]
        )
    _print_table(
        ["group", "patient", "dataset", "mri", "long", "seg", "dates", "span_d", "status"],
        rows,
    )


def print_longitudinal_table(reports: list[dict[str, Any]]) -> None:
    rows = [
        [
            report["patient_id"],
            report.get("dataset_key") or "-",
            ", ".join(report.get("study_dates", [])),
            str(report.get("span_days") if report.get("span_days") is not None else "-"),
            "yes" if report.get("segmentation_available") else "no",
            _format_bytes(report.get("total_download_size_bytes", 0)),
        ]
        for report in reports
    ]
    _print_table(["patient", "dataset", "dates", "span_d", "seg", "size"], rows)


def print_patient_table(report: dict[str, Any]) -> None:
    print(f"Patient: {report['patient_id']}")
    print(f"  Dataset: {report.get('dataset_label') or report.get('dataset_key') or '-'}")
    print(f"  MRI: {'yes' if report.get('has_mri') else 'no'}")
    print(f"  Longitudinal: {'yes' if report.get('longitudinal') else 'no'}")
    print(f"  Segmentation: {'yes' if report.get('segmentation_available') else 'no'}")
    if report.get("study_dates"):
        print(f"  Study dates / timepoints: {', '.join(report['study_dates'])}")
    if report.get("span_days") is not None:
        print(f"  Span: {report['span_days']} days")
    print(f"  On-disk size: {_format_bytes(report.get('total_download_size_bytes', 0))}")
    if report.get("portal_url"):
        print(f"  Portal: {report['portal_url']}")
    if report.get("download_notes"):
        print(f"  Download: {report['download_notes']}")
    for study in report.get("studies", []):
        if study.get("best_series"):
            best = study["best_series"]
            print(
                f"  [{study['study_date']}] {best['SeriesDescription']} "
                f"({best['ImageCount']} images, {_format_bytes(best['FileSize'])})"
            )
        elif study.get("mr_paths"):
            print(f"  [{study['study_date']}] MR files: {len(study['mr_paths'])}")
            if study.get("segmentation_paths"):
                print(f"    Segmentation files: {len(study['segmentation_paths'])}")
    if report.get("issues"):
        print(f"  Issues: {'; '.join(report['issues'])}")


def print_recommend_pair(result: dict[str, Any]) -> None:
    print("Preferred datasets:", ", ".join(result["preferred_datasets"]))
    for dataset_key, count in result["candidates"].items():
        print(f"  {dataset_key}: {count} local/API longitudinal candidate(s)")
        note = result["access_notes"].get(dataset_key)
        if note and count == 0:
            print(f"    Access: {note}")
    spike = result.get("spike_patient")
    if spike:
        patient = spike["patient"]
        print(
            f"Suggested spike: {patient['patient_id']} "
            f"({spike['dataset_key']}, "
            f"{', '.join(patient.get('study_dates', []))})"
        )
    else:
        print("Suggested spike: none yet — download a preferred dataset first.")
    for dataset_key, report in result["recommended"].items():
        if report is None:
            print(f"Best {dataset_key}: none")
            continue
        print(
            f"Best {dataset_key}: {report['patient_id']} "
            f"({', '.join(report.get('study_dates', []))}, "
            f"seg={'yes' if report.get('segmentation_available') else 'no'})"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--cohort",
        type=Path,
        default=COHORT_PATH,
        help="Path to cohort.json",
    )
    common.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    common.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_DATA_DIR,
        help="Root directory for local NIfTI inventory (default: data/raw/)",
    )

    parser = argparse.ArgumentParser(
        description="Discover longitudinal brain MRI cohorts from TCIA and local inventory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list-datasets",
        parents=[common],
        help="Show candidate datasets from DATASETS.md registry",
    )
    list_parser.add_argument(
        "--preferred-only",
        action="store_true",
        help="Only show UCSF + MU-Glioma-Post spike targets",
    )

    audit_parser = subparsers.add_parser(
        "audit",
        parents=[common],
        help="Validate cohort.json entries against local inventory / TCIA",
    )
    audit_parser.add_argument(
        "--include-backups",
        action="store_true",
        help="Also audit backup patients",
    )

    find_parser = subparsers.add_parser(
        "find-longitudinal",
        parents=[common],
        help="List longitudinal patients for one dataset",
    )
    find_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset registry key, e.g. mu_glioma_post or ucsf_longitudinal_glioma",
    )
    find_parser.add_argument(
        "--allow-missing-segmentation",
        action="store_true",
        help="Include patients without expert segmentation masks",
    )

    scan_parser = subparsers.add_parser(
        "scan-local",
        parents=[common],
        help="Scan data/raw/ inventory for one dataset",
    )
    scan_parser.add_argument("--dataset", required=True, help="Dataset registry key")

    subparsers.add_parser(
        "recommend-pair",
        parents=[common],
        help="Suggest spike patient from preferred datasets",
    )

    discover_ucsf_parser = subparsers.add_parser(
        "discover-ucsf",
        parents=[common],
        help="Audit UCSF-ALPTDG imaging + same-ID genomics eligibility; write cohort_discovery_ucsf.json",
    )
    discover_ucsf_parser.add_argument(
        "--output",
        type=Path,
        default=COHORT_DISCOVERY_UCSF_PATH,
        help="Path for discovery JSON (default: cohort/cohort_discovery_ucsf.json)",
    )
    discover_ucsf_parser.add_argument(
        "--master-csv",
        type=Path,
        default=UCSF_MASTER_CSV,
        help="UCSF longitudinal master table from clean_ucsf.py",
    )
    discover_ucsf_parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing cohort_discovery_ucsf.json (stdout / --json only)",
    )
    discover_ucsf_parser.add_argument(
        "--exclude-resection-baseline",
        action="store_true",
        help="Drop patients with label 4 (resection cavity) at baseline",
    )
    discover_ucsf_parser.add_argument(
        "--exclude-resection-followup",
        action="store_true",
        help="Drop patients with label 4 (resection cavity) at follow-up",
    )

    show_parser = subparsers.add_parser(
        "show",
        parents=[common],
        help="Detailed report for one patient ID",
    )
    show_parser.add_argument("patient_id", help="Patient folder name / TCIA PatientID")
    show_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset registry key",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    emit_json = args.json

    if args.command == "list-datasets":
        rows = list_datasets_table(preferred_only=args.preferred_only)
        if emit_json:
            print(json.dumps(rows, indent=2))
        else:
            print_datasets_table(rows)
        return 0

    if args.command == "audit":
        result = audit_cohort(
            args.cohort,
            include_backups=args.include_backups,
            raw_root=args.raw_root,
        )
        if emit_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Cohort version: {result.get('cohort_version', 'unknown')}")
            print_audit_table(result)
        return 0 if result["primary_ok"] else 1

    if args.command == "find-longitudinal":
        matches = find_longitudinal_patients(
            args.dataset,
            require_segmentation=not args.allow_missing_segmentation,
            raw_root=args.raw_root,
        )
        if emit_json:
            print(json.dumps(matches, indent=2))
        else:
            print(f"Longitudinal candidates for {args.dataset}: {len(matches)} patient(s)")
            if not matches:
                dataset = get_dataset(args.dataset)
                print(f"Access: {dataset.download_notes or dataset.portal_url or 'see COHORT.md'}")
            print_longitudinal_table(matches)
        return 0

    if args.command == "scan-local":
        reports = scan_local_dataset_inventory(args.dataset, raw_root=args.raw_root)
        if emit_json:
            print(json.dumps(reports, indent=2))
        else:
            print(f"Local inventory for {args.dataset}: {len(reports)} patient folder(s)")
            print_longitudinal_table(reports)
        return 0

    if args.command == "recommend-pair":
        result = recommend_pair(raw_root=args.raw_root)
        if emit_json:
            print(json.dumps(result, indent=2))
        else:
            print_recommend_pair(result)
        return 0

    if args.command == "discover-ucsf":
        result = discover_ucsf_cohort(
            cohort_path=args.cohort,
            master_csv=args.master_csv,
            raw_root=args.raw_root,
            exclude_resection_baseline=args.exclude_resection_baseline,
            exclude_resection_followup=args.exclude_resection_followup,
        )
        if not args.no_write:
            path = write_ucsf_discovery_json(result, args.output)
            if not emit_json:
                print(f"Wrote {path}")
        if emit_json:
            print(json.dumps(result, indent=2))
        elif not args.no_write:
            print_ucsf_discovery_summary(result)
        return 0

    if args.command == "show":
        report = build_patient_report(
            args.patient_id,
            dataset_key=args.dataset,
            raw_root=args.raw_root,
        )
        if emit_json:
            print(json.dumps(report, indent=2))
        else:
            print_patient_table(report)
        return 0 if report.get("ok") else 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
