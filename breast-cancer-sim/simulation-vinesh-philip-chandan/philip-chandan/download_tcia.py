"""Download TCIA DICOM series for cohort patients via the public NBIA REST API."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from tcia_extractor import iter_cohort_patients, load_cohort, resolve_dicom_dir, resolve_study_dir

NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
DEFAULT_COLLECTION = "TCGA-BRCA"


def _api_get(path: str, params: dict[str, str] | None = None) -> bytes:
    query = urllib.parse.urlencode(params or {})
    url = f"{NBIA_BASE}/{path}"
    if query:
        url = f"{url}?{query}"
    with urllib.request.urlopen(url) as response:
        return response.read()


def _study_date(series: dict) -> str:
    raw_date = str(series.get("StudyDate", ""))
    return raw_date[:10] if raw_date else "unknown-study"


def list_mr_series(
    patient_id: str,
    collection: str = DEFAULT_COLLECTION,
) -> list[dict]:
    payload = _api_get(
        "getSeries",
        {
            "Collection": collection,
            "PatientID": patient_id,
            "Modality": "MR",
        },
    )
    if not payload.strip():
        return []
    return json.loads(payload)


def group_series_by_study(series_list: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in series_list:
        grouped[_study_date(entry)].append(entry)
    return dict(grouped)


def pick_series(
    series_list: list[dict],
    *,
    prefer_contrast: bool = True,
) -> dict | None:
    if not series_list:
        return None

    def score(entry: dict) -> tuple[int, int]:
        description = str(entry.get("SeriesDescription", "")).lower()
        image_count = int(entry.get("ImageCount", 0) or 0)
        contrast_bonus = 0
        if prefer_contrast and any(token in description for token in ("+c", "post", "vibrant", "t1")):
            contrast_bonus = -1000
        calibration_penalty = 500 if "cal" in description or "asset" in description else 0
        loc_penalty = 200 if "loc" in description else 0
        return (contrast_bonus + calibration_penalty + loc_penalty, -image_count)

    return sorted(series_list, key=score)[0]


def download_series_zip(series_uid: str, destination_zip: Path) -> Path:
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    payload = _api_get("getImage", {"SeriesInstanceUID": series_uid})
    destination_zip.write_bytes(payload)
    return destination_zip


def extract_zip(zip_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination_dir)
    return destination_dir


def download_series_to_dir(
    series: dict,
    destination_dir: Path,
) -> dict:
    series_uid = series["SeriesInstanceUID"]
    zip_path = destination_dir / f"{series_uid}.zip"
    download_series_zip(series_uid, zip_path)
    extract_zip(zip_path, destination_dir)
    zip_path.unlink(missing_ok=True)
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
) -> dict:
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
) -> list[dict]:
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

    downloads: list[dict] = []
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
) -> list[dict]:
    """Download MR for cohort patients that have imaging on TCIA."""
    results: list[dict] = []
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
