"""Download TCIA DICOM series for cohort patients.

Series discovery and download use IDC Index (primary) with tcia-utils NBIA helpers
as fallback. Public function signatures and on-disk cohort layout are unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from tcia_extractor import iter_cohort_patients, load_cohort, resolve_dicom_dir, resolve_study_dir

DEFAULT_COLLECTION = "TCGA-BRCA"

IDC_COLLECTION_IDS = {
    "TCGA-BRCA": "tcga_brca",
}


@lru_cache(maxsize=1)
def _get_idc_client():
    from idc_index import IDCClient

    return IDCClient()


def _idc_collection_id(collection: str) -> str:
    return IDC_COLLECTION_IDS.get(collection, collection.lower().replace("-", "_"))


def normalize_study_date(raw: str) -> str:
    """Normalize TCIA/IDC study dates to YYYY-MM-DD."""
    value = str(raw).strip()
    if not value:
        return "unknown-study"
    if len(value) >= 8 and value[:8].isdigit():
        digits = value[:8]
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    parts = value.split("-")
    if len(parts) >= 3:
        first, second, third = parts[0], parts[1], parts[2]
        if len(first) == 4 and first.isdigit() and second.isdigit() and third.isdigit():
            return f"{first}-{second.zfill(2)}-{third.zfill(2)}"
        if len(third) == 4 and third.isdigit() and first.isdigit() and second.isdigit():
            return f"{third}-{first.zfill(2)}-{second.zfill(2)}"
    return value[:10]


def _study_date(series: dict[str, Any]) -> str:
    return normalize_study_date(str(series.get("StudyDate", "")))


def _series_dict_from_idc_row(row: dict[str, Any]) -> dict[str, Any]:
    image_count = int(row.get("ImageCount", 0) or 0)
    size_mb = row.get("series_size_MB")
    file_size = int(float(size_mb) * 1_000_000) if size_mb not in (None, "") else 0
    return {
        "PatientID": row.get("PatientID", ""),
        "StudyDate": normalize_study_date(str(row.get("StudyDate", ""))),
        "SeriesInstanceUID": row.get("SeriesInstanceUID", ""),
        "SeriesDescription": row.get("SeriesDescription", ""),
        "Modality": row.get("Modality", "MR"),
        "ImageCount": image_count,
        "FileSize": file_size,
        "Collection": DEFAULT_COLLECTION,
    }


def _list_mr_series_idc(patient_id: str, collection: str) -> list[dict[str, Any]]:
    idc_collection = _idc_collection_id(collection)
    query = f"""
        SELECT
            PatientID,
            StudyDate,
            SeriesInstanceUID,
            SeriesDescription,
            Modality,
            instanceCount AS ImageCount,
            series_size_MB
        FROM index
        WHERE collection_id = '{idc_collection}'
          AND PatientID = '{patient_id}'
          AND Modality = 'MR'
    """
    dataframe = _get_idc_client().sql_query(query)
    if dataframe.empty:
        return []
    return [_series_dict_from_idc_row(row) for row in dataframe.to_dict(orient="records")]


def _list_mr_series_nbia(patient_id: str, collection: str) -> list[dict[str, Any]]:
    from tcia_utils import nbia

    series_list = nbia.getSeries(collection=collection, patientId=patient_id, modality="MR")
    if not series_list:
        return []

    normalized: list[dict[str, Any]] = []
    for entry in series_list:
        record = dict(entry)
        record["StudyDate"] = normalize_study_date(str(entry.get("StudyDate", "")))
        normalized.append(record)
    return normalized


def list_mr_series(
    patient_id: str,
    collection: str = DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """List MR series metadata for a patient from IDC Index, falling back to NBIA."""
    try:
        series_list = _list_mr_series_idc(patient_id, collection)
        if series_list:
            return series_list
    except Exception:
        pass
    return _list_mr_series_nbia(patient_id, collection)


def group_series_by_study(series_list: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in series_list:
        grouped[_study_date(entry)].append(entry)
    return dict(grouped)


def pick_series(
    series_list: list[dict[str, Any]],
    *,
    prefer_contrast: bool = True,
) -> dict[str, Any] | None:
    if not series_list:
        return None

    def score(entry: dict[str, Any]) -> tuple[int, int]:
        description = str(entry.get("SeriesDescription", "")).lower()
        image_count = int(entry.get("ImageCount", 0) or 0)
        contrast_bonus = 0
        if prefer_contrast and any(token in description for token in ("+c", "post", "vibrant", "t1")):
            contrast_bonus = -1000
        calibration_penalty = 500 if "cal" in description or "asset" in description else 0
        loc_penalty = 200 if "loc" in description else 0
        return (contrast_bonus + calibration_penalty + loc_penalty, -image_count)

    return sorted(series_list, key=score)[0]


def _download_series_idc(series_uid: str, destination_dir: Path) -> None:
    _get_idc_client().download_dicom_series(
        series_uid,
        str(destination_dir),
        dirTemplate=None,
        quiet=True,
        show_progress_bar=True,
    )


def _download_series_nbia(series_uid: str, destination_dir: Path) -> None:
    from tcia_utils import nbia

    nbia.downloadSeries([series_uid], path=str(destination_dir), input_type="list")


def download_series_to_dir(
    series: dict[str, Any],
    destination_dir: Path,
) -> dict[str, Any]:
    series_uid = series["SeriesInstanceUID"]
    destination_dir.mkdir(parents=True, exist_ok=True)

    try:
        _download_series_idc(series_uid, destination_dir)
    except Exception:
        _download_series_nbia(series_uid, destination_dir)

    return {
        "dicom_dir": destination_dir,
        "series_uid": series_uid,
        "series_description": series.get("SeriesDescription"),
        "image_count": series.get("ImageCount"),
        "study_date": _study_date(series),
    }


def download_patient_mr(
    tcga_id: str,
    subtype: str,
    *,
    collection: str = DEFAULT_COLLECTION,
    output_dir: Path | None = None,
    prefer_contrast: bool = True,
) -> dict[str, Any]:
    """Download the best MR series for a patient into the cohort DICOM layout."""
    destination = output_dir or resolve_dicom_dir(tcga_id, subtype)
    series_list = list_mr_series(tcga_id, collection=collection)
    if not series_list:
        raise FileNotFoundError(f"No MR series found on TCIA for {tcga_id}")

    chosen = pick_series(series_list, prefer_contrast=prefer_contrast)
    assert chosen is not None
    result = download_series_to_dir(chosen, destination)
    return {
        "tcga_id": tcga_id,
        "subtype": subtype,
        **result,
    }


def download_patient_longitudinal(
    tcga_id: str,
    subtype: str,
    *,
    collection: str = DEFAULT_COLLECTION,
    prefer_contrast: bool = True,
    study_dates: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Download the best contrast MR series for each longitudinal study date."""
    series_list = list_mr_series(tcga_id, collection=collection)
    if not series_list:
        raise FileNotFoundError(f"No MR series found on TCIA for {tcga_id}")

    grouped = group_series_by_study(series_list)
    selected_dates = study_dates or sorted(grouped)
    if study_dates:
        missing = [study_date for study_date in study_dates if study_date not in grouped]
        if missing:
            raise FileNotFoundError(
                f"Requested study dates not found for {tcga_id}: {', '.join(missing)}"
            )

    downloads: list[dict[str, Any]] = []
    for study_date in selected_dates:
        chosen = pick_series(grouped[study_date], prefer_contrast=prefer_contrast)
        if chosen is None:
            continue
        destination = resolve_study_dir(tcga_id, subtype, study_date)
        result = download_series_to_dir(chosen, destination)
        downloads.append(
            {
                "tcga_id": tcga_id,
                "subtype": subtype,
                **result,
            }
        )
    if not downloads:
        raise FileNotFoundError(f"No downloadable MR studies found for {tcga_id}")
    return downloads


def download_cohort(
    *,
    include_backups: bool = False,
    primary_only: bool = True,
    prefer_contrast: bool = True,
    longitudinal: bool = False,
) -> list[dict[str, Any]]:
    """Download MR for cohort patients that have imaging on TCIA."""
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    patients = list(iter_cohort_patients(include_backups=include_backups))
    if primary_only:
        patients = [patient for patient in patients if patient["cohort_group"] == "primary"]

    for patient in patients:
        tcga_id = patient["tcga_id"]
        subtype = patient["subtype"]
        imaging = patient.get("imaging", {})
        try:
            if longitudinal or imaging.get("longitudinal"):
                study_dates = [
                    timepoint["study_date"]
                    for timepoint in imaging.get("timepoints", [])
                    if timepoint.get("study_date")
                ]
                patient_results = download_patient_longitudinal(
                    tcga_id,
                    subtype,
                    prefer_contrast=prefer_contrast,
                    study_dates=study_dates or None,
                )
                results.extend(patient_results)
                for result in patient_results:
                    print(
                        f"OK {subtype} {tcga_id} {result['study_date']}: "
                        f"{result['series_description']} ({result['image_count']} slices)"
                    )
            else:
                result = download_patient_mr(
                    tcga_id,
                    subtype,
                    prefer_contrast=prefer_contrast,
                )
                results.append(result)
                print(
                    f"OK {subtype} {tcga_id}: "
                    f"{result['series_description']} ({result['image_count']} slices)"
                )
        except FileNotFoundError as exc:
            errors.append(str(exc))
            print(f"SKIP {subtype} {tcga_id}: {exc}")

    if errors and not results:
        raise RuntimeError("\n".join(errors))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TCIA MR DICOM for cohort patients.")
    parser.add_argument("--tcga-id", help="Single TCGA barcode to download")
    parser.add_argument("--subtype", help="Subtype label, e.g. 'Luminal A'")
    parser.add_argument(
        "--include-backups",
        action="store_true",
        help="Also attempt backup/later patients from cohort.json",
    )
    parser.add_argument(
        "--all-primary",
        action="store_true",
        help="Download all primary cohort patients",
    )
    parser.add_argument(
        "--longitudinal",
        action="store_true",
        help="Download one contrast series per MR study date into dated subfolders",
    )
    parser.add_argument(
        "--no-contrast-preference",
        action="store_true",
        help="Do not prefer post-contrast series",
    )
    args = parser.parse_args()

    prefer_contrast = not args.no_contrast_preference

    if args.tcga_id:
        if not args.subtype:
            raise SystemExit("--subtype is required with --tcga-id")
        if args.longitudinal:
            cohort = load_cohort()
            study_dates = None
            for entry in cohort.get("primary", []):
                if entry["tcga_id"] == args.tcga_id:
                    study_dates = [
                        timepoint["study_date"]
                        for timepoint in entry.get("imaging", {}).get("timepoints", [])
                        if timepoint.get("study_date")
                    ]
            results = download_patient_longitudinal(
                args.tcga_id,
                args.subtype,
                prefer_contrast=prefer_contrast,
                study_dates=study_dates,
            )
            print(json.dumps(results, indent=2, default=str))
            return

        result = download_patient_mr(
            args.tcga_id,
            args.subtype,
            prefer_contrast=prefer_contrast,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    if args.all_primary or args.include_backups:
        download_cohort(
            include_backups=args.include_backups,
            primary_only=not args.include_backups,
            prefer_contrast=prefer_contrast,
            longitudinal=args.longitudinal,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
