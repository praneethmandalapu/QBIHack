"""TCIA cohort discovery helper for TCGA-BRCA longitudinal MRI selection.

Queries the public TCIA NBIA REST API and cross-checks PAM50 subtypes via
cBioPortal so we can validate or update patients in cohort.json without
manual API spelunking.

Usage (from breast-cancer-sim/):
  python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py audit
  python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py find-longitudinal --subtype "Luminal A"
  python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py recommend-pair
  python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py show TCGA-AR-A1AX --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cohort import COHORT_PATH
from download_tcia import DEFAULT_COLLECTION, list_mr_series, pick_series
from tcia_extractor import load_cohort

NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
CBIO_STUDY = "brca_tcga_pan_can_atlas_2018"
CBIO_BASE = "https://www.cbioportal.org/api"
CBIO_DELAY_SECONDS = 0.25

PAM50_TO_COHORT: dict[str, str] = {
    "BRCA_LumA": "Luminal A",
    "BRCA_Basal": "Basal-like",
    "BRCA_LumB": "Luminal B",
    "BRCA_Her2": "HER2-enriched",
}

COHORT_TO_PAM50: dict[str, set[str]] = {
    "Luminal A": {"BRCA_LumA"},
    "Basal-like": {"BRCA_Basal"},
    "Luminal B": {"BRCA_LumB"},
    "HER2-enriched": {"BRCA_Her2"},
}

HttpFetcher = Callable[[str], bytes]
_last_cbio_call = 0.0


def _default_fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def _nbia_url(path: str, params: dict[str, str] | None = None) -> str:
    query = urllib.parse.urlencode(params or {})
    url = f"{NBIA_BASE}/{path}"
    if query:
        url = f"{url}?{query}"
    return url


def _cbio_url(path: str) -> str:
    return f"{CBIO_BASE}/{path.lstrip('/')}"


def normalize_study_date(raw: str) -> str:
    """Normalize TCIA StudyDate values to YYYY-MM-DD."""
    value = str(raw).strip()
    if not value:
        return "unknown-study"
    if len(value) >= 10 and value[4] == "-" and value[7] == "-":
        return value[:10]
    if len(value) >= 8 and value[:8].isdigit():
        digits = value[:8]
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return value[:10]


def _series_study_date(series: dict[str, Any]) -> str:
    return normalize_study_date(str(series.get("StudyDate", "")))


def group_series_by_normalized_study(series_list: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in series_list:
        study_date = _series_study_date(entry)
        grouped.setdefault(study_date, []).append(entry)
    return grouped


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
    return any(token in description for token in ("+c", "post", "vibrant", "t1"))


def list_tcia_patients(
    collection: str = DEFAULT_COLLECTION,
    *,
    fetcher: HttpFetcher | None = None,
) -> list[str]:
    fetch = fetcher or _default_fetch
    payload = fetch(_nbia_url("getPatient", {"Collection": collection}))
    if not payload.strip():
        return []
    patients = json.loads(payload)
    return sorted(str(entry.get("PatientID", "")) for entry in patients if entry.get("PatientID"))


def fetch_pam50_subtype(
    tcga_id: str,
    *,
    fetcher: HttpFetcher | None = None,
    delay_seconds: float = CBIO_DELAY_SECONDS,
) -> dict[str, Any]:
    global _last_cbio_call

    fetch = fetcher or _default_fetch
    if delay_seconds > 0:
        elapsed = time.monotonic() - _last_cbio_call
        if elapsed < delay_seconds:
            time.sleep(delay_seconds - elapsed)

    url = _cbio_url(f"studies/{CBIO_STUDY}/patients/{tcga_id}/clinical-data")
    payload = fetch(url)
    _last_cbio_call = time.monotonic()

    if not payload.strip():
        return {"pam50_raw": None, "pam50_label": None}

    clinical_rows = json.loads(payload)
    subtype_value = None
    for row in clinical_rows:
        if str(row.get("clinicalAttributeId", "")).upper() == "SUBTYPE":
            subtype_value = row.get("value")
            break

    pam50_raw = str(subtype_value) if subtype_value is not None else None
    pam50_label = PAM50_TO_COHORT.get(pam50_raw or "", None)
    return {"pam50_raw": pam50_raw, "pam50_label": pam50_label}


def analyze_patient_imaging(
    tcga_id: str,
    *,
    collection: str = DEFAULT_COLLECTION,
    series_list: list[dict[str, Any]] | None = None,
    fetcher: HttpFetcher | None = None,
) -> dict[str, Any]:
    if series_list is None:
        if fetcher is not None:
            payload = fetcher(
                _nbia_url(
                    "getSeries",
                    {"Collection": collection, "PatientID": tcga_id, "Modality": "MR"},
                )
            )
            series_list = json.loads(payload) if payload.strip() else []
        else:
            series_list = list_mr_series(tcga_id, collection=collection)

    grouped = group_series_by_normalized_study(series_list)
    study_dates = sorted(date for date in grouped if date != "unknown-study")
    longitudinal = len(study_dates) >= 2

    studies: list[dict[str, Any]] = []
    total_download_size = 0
    for study_date in study_dates:
        best = pick_series(grouped[study_date], prefer_contrast=True)
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
        "tcga_id": tcga_id,
        "has_mri": bool(series_list),
        "series_count": len(series_list),
        "longitudinal": longitudinal,
        "study_dates": study_dates,
        "span_days": _span_days(study_dates),
        "studies": studies,
        "total_download_size_bytes": total_download_size,
    }


def build_patient_report(
    tcga_id: str,
    *,
    cohort_subtype: str | None = None,
    collection: str = DEFAULT_COLLECTION,
    series_list: list[dict[str, Any]] | None = None,
    pam50: dict[str, Any] | None = None,
    fetcher: HttpFetcher | None = None,
    cbio_fetcher: HttpFetcher | None = None,
    skip_cbio: bool = False,
) -> dict[str, Any]:
    imaging = analyze_patient_imaging(
        tcga_id,
        collection=collection,
        series_list=series_list,
        fetcher=fetcher,
    )
    pam50_info = pam50 if pam50 is not None else (
        {} if skip_cbio else fetch_pam50_subtype(tcga_id, fetcher=cbio_fetcher or fetcher)
    )

    pam50_raw = pam50_info.get("pam50_raw")
    pam50_label = pam50_info.get("pam50_label")
    subtype_match = None
    if cohort_subtype and pam50_label:
        subtype_match = pam50_label == cohort_subtype

    issues: list[str] = []
    if not imaging["has_mri"]:
        issues.append("missing MRI on TCIA")
    if not imaging["longitudinal"]:
        issues.append("not longitudinal (fewer than 2 MR study dates)")
    if cohort_subtype and pam50_label and not subtype_match:
        issues.append(
            f"PAM50 mismatch: cohort={cohort_subtype}, cBioPortal={pam50_label} ({pam50_raw})"
        )
    if cohort_subtype and pam50_label is None and pam50_raw is None and not skip_cbio:
        issues.append("PAM50 subtype unavailable on cBioPortal")

    return {
        **imaging,
        "pam50_raw": pam50_raw,
        "pam50_label": pam50_label,
        "cohort_subtype": cohort_subtype,
        "subtype_match": subtype_match,
        "issues": issues,
        "ok": not issues,
    }


def iter_cohort_entries(
    cohort: dict[str, Any],
    *,
    include_backups: bool = False,
    include_later: bool = False,
) -> Iterator[tuple[str, str, str]]:
    for entry in cohort.get("primary", []):
        yield ("primary", entry["subtype"], entry["tcga_id"])

    if include_backups:
        for subtype, backup_entries in cohort.get("backups", {}).items():
            for entry in backup_entries:
                yield ("backup", subtype, entry["tcga_id"])

    if include_later:
        for subtype, later_entries in cohort.get("later", {}).items():
            for entry in later_entries:
                yield ("later", subtype, entry["tcga_id"])


def audit_cohort(
    cohort_path: Path | None = None,
    *,
    include_backups: bool = False,
    include_later: bool = False,
    fetcher: HttpFetcher | None = None,
    cbio_fetcher: HttpFetcher | None = None,
) -> dict[str, Any]:
    cohort = load_cohort(cohort_path)
    reports: list[dict[str, Any]] = []

    for group, subtype, tcga_id in iter_cohort_entries(
        cohort,
        include_backups=include_backups,
        include_later=include_later,
    ):
        report = build_patient_report(
            tcga_id,
            cohort_subtype=subtype,
            fetcher=fetcher,
            cbio_fetcher=cbio_fetcher,
        )
        report["cohort_group"] = group
        reports.append(report)

    primary_reports = [report for report in reports if report["cohort_group"] == "primary"]
    primary_ok = all(report["ok"] for report in primary_reports)

    return {
        "cohort_version": cohort.get("version"),
        "reports": reports,
        "primary_ok": primary_ok,
    }


def find_longitudinal_patients(
    subtype: str,
    *,
    collection: str = DEFAULT_COLLECTION,
    fetcher: HttpFetcher | None = None,
    cbio_fetcher: HttpFetcher | None = None,
) -> list[dict[str, Any]]:
    expected_pam50 = COHORT_TO_PAM50.get(subtype, set())
    matches: list[dict[str, Any]] = []

    for tcga_id in list_tcia_patients(collection=collection, fetcher=fetcher):
        imaging = analyze_patient_imaging(tcga_id, collection=collection, fetcher=fetcher)
        if not imaging["longitudinal"]:
            continue

        pam50 = fetch_pam50_subtype(tcga_id, fetcher=cbio_fetcher or fetcher)
        if expected_pam50 and pam50.get("pam50_raw") not in expected_pam50:
            continue

        report = build_patient_report(
            tcga_id,
            cohort_subtype=subtype,
            series_list=None,
            pam50=pam50,
            collection=collection,
            fetcher=fetcher,
            skip_cbio=True,
        )
        matches.append(report)

    matches.sort(key=lambda report: (report.get("span_days") or 0, report["tcga_id"]), reverse=True)
    return matches


def recommend_pair(
    *,
    collection: str = DEFAULT_COLLECTION,
    fetcher: HttpFetcher | None = None,
    cbio_fetcher: HttpFetcher | None = None,
) -> dict[str, Any]:
    luma_candidates = find_longitudinal_patients(
        "Luminal A",
        collection=collection,
        fetcher=fetcher,
        cbio_fetcher=cbio_fetcher,
    )
    basal_candidates = find_longitudinal_patients(
        "Basal-like",
        collection=collection,
        fetcher=fetcher,
        cbio_fetcher=cbio_fetcher,
    )

    def pair_score(luma: dict[str, Any], basal: dict[str, Any]) -> tuple[int, int, int]:
        luma_contrast = all(study.get("contrast_available") for study in luma.get("studies", []))
        basal_contrast = all(study.get("contrast_available") for study in basal.get("studies", []))
        min_span = min(luma.get("span_days") or 0, basal.get("span_days") or 0)
        total_span = (luma.get("span_days") or 0) + (basal.get("span_days") or 0)
        return (int(luma_contrast and basal_contrast), min_span, total_span)

    best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_score: tuple[int, int, int] | None = None

    for luma in luma_candidates:
        for basal in basal_candidates:
            score = pair_score(luma, basal)
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (luma, basal)

    return {
        "luminal_a_candidates": len(luma_candidates),
        "basal_candidates": len(basal_candidates),
        "recommended": {
            "Luminal A": best_pair[0] if best_pair else None,
            "Basal-like": best_pair[1] if best_pair else None,
        },
    }


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


def print_audit_table(result: dict[str, Any]) -> None:
    rows: list[list[str]] = []
    for report in result["reports"]:
        study_summary = ", ".join(report.get("study_dates", [])) or "-"
        span = str(report.get("span_days") if report.get("span_days") is not None else "-")
        rows.append(
            [
                report["cohort_group"],
                report["tcga_id"],
                report.get("cohort_subtype") or "-",
                "yes" if report["has_mri"] else "no",
                str(report["series_count"]),
                "yes" if report["longitudinal"] else "no",
                study_summary,
                span,
                report.get("pam50_label") or report.get("pam50_raw") or "-",
                "ok" if report["ok"] else "; ".join(report["issues"]),
            ]
        )

    _print_table(
        [
            "group",
            "tcga_id",
            "subtype",
            "mri",
            "series",
            "long",
            "study_dates",
            "span_d",
            "pam50",
            "status",
        ],
        rows,
    )


def print_patient_table(report: dict[str, Any]) -> None:
    print(f"Patient: {report['tcga_id']}")
    print(f"  MRI on TCIA: {'yes' if report['has_mri'] else 'no'} ({report['series_count']} series)")
    print(f"  Longitudinal: {'yes' if report['longitudinal'] else 'no'}")
    if report.get("study_dates"):
        print(f"  Study dates: {', '.join(report['study_dates'])}")
    if report.get("span_days") is not None:
        print(f"  Span: {report['span_days']} days")
    print(
        f"  PAM50: {report.get('pam50_label') or '-'} "
        f"({report.get('pam50_raw') or 'n/a'})"
    )
    if report.get("cohort_subtype"):
        print(f"  Cohort subtype: {report['cohort_subtype']}")
    print(f"  Total download size: {_format_bytes(report.get('total_download_size_bytes', 0))}")

    for study in report.get("studies", []):
        best = study["best_series"]
        print(
            f"  [{study['study_date']}] {best['SeriesDescription']} "
            f"({best['ImageCount']} images, {_format_bytes(best['FileSize'])})"
        )
    if report.get("issues"):
        print(f"  Issues: {'; '.join(report['issues'])}")


def print_longitudinal_table(reports: list[dict[str, Any]]) -> None:
    rows = [
        [
            report["tcga_id"],
            report.get("pam50_label") or report.get("pam50_raw") or "-",
            ", ".join(report.get("study_dates", [])),
            str(report.get("span_days") if report.get("span_days") is not None else "-"),
            _format_bytes(report.get("total_download_size_bytes", 0)),
        ]
        for report in reports
    ]
    _print_table(["tcga_id", "pam50", "study_dates", "span_d", "download"], rows)


def print_recommend_pair(result: dict[str, Any]) -> None:
    print(
        f"Longitudinal candidates: Luminal A={result['luminal_a_candidates']}, "
        f"Basal-like={result['basal_candidates']}"
    )
    recommended = result["recommended"]
    for subtype in ("Luminal A", "Basal-like"):
        report = recommended.get(subtype)
        if report is None:
            print(f"{subtype}: no candidate found")
            continue
        print(
            f"{subtype}: {report['tcga_id']} "
            f"({', '.join(report.get('study_dates', []))}, "
            f"{report.get('span_days')} days, "
            f"{_format_bytes(report.get('total_download_size_bytes', 0))})"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--cohort",
        type=Path,
        default=COHORT_PATH,
        help="Path to cohort.json (default: philip-chandan/cohort/cohort.json)",
    )
    common.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    parser = argparse.ArgumentParser(
        description="Search TCIA TCGA-BRCA imaging and cross-check PAM50 subtypes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit",
        parents=[common],
        help="Validate cohort IDs against TCIA + PAM50",
    )
    audit_parser.add_argument(
        "--include-backups",
        action="store_true",
        help="Also audit backup patients",
    )
    audit_parser.add_argument(
        "--include-later",
        action="store_true",
        help="Also audit later-phase patients",
    )

    find_parser = subparsers.add_parser(
        "find-longitudinal",
        parents=[common],
        help="List TCIA patients with longitudinal MR and matching PAM50 subtype",
    )
    find_parser.add_argument(
        "--subtype",
        required=True,
        help='Cohort subtype label, e.g. "Luminal A" or "Basal-like"',
    )

    subparsers.add_parser(
        "recommend-pair",
        parents=[common],
        help="Suggest best LumA + Basal longitudinal pair with contrast series",
    )

    show_parser = subparsers.add_parser(
        "show",
        parents=[common],
        help="Detailed report for one TCGA barcode",
    )
    show_parser.add_argument("tcga_id", help="TCGA barcode, e.g. TCGA-AR-A1AX")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    emit_json = args.json

    if args.command == "audit":
        result = audit_cohort(
            args.cohort,
            include_backups=args.include_backups,
            include_later=args.include_later,
        )
        if emit_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Cohort version: {result.get('cohort_version', 'unknown')}")
            print_audit_table(result)
        return 0 if result["primary_ok"] else 1

    if args.command == "find-longitudinal":
        matches = find_longitudinal_patients(args.subtype)
        if emit_json:
            print(json.dumps(matches, indent=2))
        else:
            print(f"Longitudinal MR + PAM50 match for {args.subtype}: {len(matches)} patient(s)")
            print_longitudinal_table(matches)
        return 0

    if args.command == "recommend-pair":
        result = recommend_pair()
        if emit_json:
            print(json.dumps(result, indent=2))
        else:
            print_recommend_pair(result)
        return 0

    if args.command == "show":
        cohort = load_cohort(args.cohort)
        cohort_subtype = None
        for group, subtype, tcga_id in iter_cohort_entries(
            cohort,
            include_backups=True,
            include_later=True,
        ):
            if tcga_id == args.tcga_id:
                cohort_subtype = subtype
                break

        report = build_patient_report(args.tcga_id, cohort_subtype=cohort_subtype)
        if emit_json:
            print(json.dumps(report, indent=2))
        else:
            print_patient_table(report)
        return 0 if report["ok"] or cohort_subtype is None else 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
