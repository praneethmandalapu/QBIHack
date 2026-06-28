"""Longitudinal whole-tumor (WT) volume report from expert segmentations.

Computes WT volume (BraTS labels 1+2+3) from on-disk segmentation NIfTI and,
when requested, compares against UCSF clinical workbook values in
``data/processed/ucsf_longitudinal_master.csv``.

UCSF workbook comparison applies only to UCSF-LPTDG cohort patients
(``--compare-ucsf-workbook``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cohort import UCSF_MASTER_CSV
from cohort.cohort_io import filter_cohort_timepoints, iter_cohort_entries, resolve_repo_path
from nifti_extractor import wt_volume_from_segmentation

UCSF_DATASET_KEYS = frozenset(
    {
        "ucsf_longitudinal_glioma",
        "UCSF Longitudinal Glioma",
        "UCSF Longitudinal Glioma (UCSF-LPTDG)",
    }
)

WORKBOOK_WT_T1 = "wt_volume_label1_plus_2_plus_3_t1"
WORKBOOK_WT_T2 = "wt_volume_label1_plus_2_plus_3_t2"
WORKBOOK_WT_CHANGE = "wt_change"
WORKBOOK_WT_GROWTH_PCT = "wt_growth_pct"
WORKBOOK_INTERVAL_DAYS = "days_from_1st_scan_to_2nd_scan"

BASELINE_LABELS = frozenset({"baseline", "time1", "tp1"})
FOLLOWUP_LABELS = frozenset({"followup", "time2", "tp2"})


@dataclass(frozen=True)
class TimepointVolume:
    label: str
    seg_path: str
    computed_mm3: float
    workbook_mm3: float | None = None
    delta_mm3: float | None = None
    delta_pct: float | None = None


@dataclass(frozen=True)
class PatientVolumeRow:
    patient_id: str
    dataset_key: str
    grade: str | float | None
    idh_status: str | None
    interval_days: float | None
    baseline: TimepointVolume | None
    followup: TimepointVolume | None
    computed_delta_mm3: float | None
    computed_growth_pct: float | None
    workbook_delta_mm3: float | None
    workbook_growth_pct: float | None
    delta_delta_mm3: float | None


def is_ucsf_dataset(dataset_key: str | None) -> bool:
    key = str(dataset_key or "").strip()
    if key in UCSF_DATASET_KEYS:
        return True
    lowered = key.lower()
    return "ucsf" in lowered and "glioma" in lowered


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_ucsf_workbook(csv_path: Path = UCSF_MASTER_CSV) -> dict[str, dict[str, float]]:
    """Load per-patient workbook WT volumes keyed by subject ID."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"UCSF workbook CSV not found: {csv_path}\n"
            "Run models-praneeth/clean_ucsf.py after placing Table S1 xlsx under data/raw/ucsf_glioma/."
        )

    by_patient: dict[str, dict[str, float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = str(row.get("subjectid", "")).strip()
            if not patient_id:
                continue
            by_patient[patient_id] = {
                "wt_t1_mm3": _parse_float(row.get(WORKBOOK_WT_T1)) or 0.0,
                "wt_t2_mm3": _parse_float(row.get(WORKBOOK_WT_T2)) or 0.0,
                "wt_change_mm3": _parse_float(row.get(WORKBOOK_WT_CHANGE)),
                "wt_growth_pct": _parse_float(row.get(WORKBOOK_WT_GROWTH_PCT)),
                "interval_days": _parse_float(row.get(WORKBOOK_INTERVAL_DAYS)),
                "grade": _parse_float(row.get("grade")),
                "idh": (row.get("idh") or "").strip() or None,
            }
    return by_patient


def _timepoint_bucket(label: str) -> str | None:
    normalized = label.strip().lower()
    if normalized in BASELINE_LABELS:
        return "baseline"
    if normalized in FOLLOWUP_LABELS:
        return "followup"
    return None


def _growth_pct(initial_mm3: float, final_mm3: float) -> float | None:
    if initial_mm3 <= 0:
        return None
    return 100.0 * (final_mm3 - initial_mm3) / initial_mm3


def _compare_timepoint(
    *,
    label: str,
    seg_path: Path,
    workbook_mm3: float | None,
) -> TimepointVolume:
    computed = wt_volume_from_segmentation(seg_path)
    delta_mm3 = None
    delta_pct = None
    if workbook_mm3 is not None:
        delta_mm3 = computed - workbook_mm3
        if workbook_mm3 > 0:
            delta_pct = 100.0 * delta_mm3 / workbook_mm3
    return TimepointVolume(
        label=label,
        seg_path=str(seg_path),
        computed_mm3=computed,
        workbook_mm3=workbook_mm3,
        delta_mm3=delta_mm3,
        delta_pct=delta_pct,
    )


def build_patient_volume_row(
    patient: dict[str, Any],
    *,
    workbook: dict[str, dict[str, float]] | None,
) -> PatientVolumeRow | None:
    patient_id = str(patient["patient_id"])
    dataset_key = str(patient.get("dataset_key", ""))
    timepoints = filter_cohort_timepoints(patient, None)
    by_bucket: dict[str, dict[str, Any]] = {}
    for timepoint in timepoints:
        bucket = _timepoint_bucket(str(timepoint.get("label", "")))
        if bucket is None:
            continue
        seg_raw = timepoint.get("segmentation_path")
        if not seg_raw:
            continue
        seg_path = resolve_repo_path(str(seg_raw))
        if not seg_path.is_file():
            print(
                f"SKIP volume {patient_id} {timepoint['label']}: seg missing {seg_path}",
                file=sys.stderr,
            )
            continue
        by_bucket[bucket] = {"label": str(timepoint["label"]), "seg_path": seg_path}

    if "baseline" not in by_bucket or "followup" not in by_bucket:
        print(
            f"SKIP volume {patient_id}: need baseline + followup segmentations on disk",
            file=sys.stderr,
        )
        return None

    wb = (workbook or {}).get(patient_id, {})
    baseline = _compare_timepoint(
        label=by_bucket["baseline"]["label"],
        seg_path=by_bucket["baseline"]["seg_path"],
        workbook_mm3=wb.get("wt_t1_mm3") if workbook is not None else None,
    )
    followup = _compare_timepoint(
        label=by_bucket["followup"]["label"],
        seg_path=by_bucket["followup"]["seg_path"],
        workbook_mm3=wb.get("wt_t2_mm3") if workbook is not None else None,
    )

    computed_delta = followup.computed_mm3 - baseline.computed_mm3
    computed_growth = _growth_pct(baseline.computed_mm3, followup.computed_mm3)
    workbook_delta = wb.get("wt_change_mm3") if workbook is not None else None
    workbook_growth = wb.get("wt_growth_pct") if workbook is not None else None
    delta_delta = None
    if workbook_delta is not None:
        delta_delta = computed_delta - workbook_delta

    grade = patient.get("grade")
    if grade is None and workbook is not None:
        grade = wb.get("grade")
    idh_status = patient.get("idh_status")
    if idh_status is None and workbook is not None:
        idh_status = wb.get("idh")

    interval_days = wb.get("interval_days") if workbook is not None else None

    return PatientVolumeRow(
        patient_id=patient_id,
        dataset_key=dataset_key,
        grade=grade,
        idh_status=str(idh_status) if idh_status is not None else None,
        interval_days=interval_days,
        baseline=baseline,
        followup=followup,
        computed_delta_mm3=computed_delta,
        computed_growth_pct=computed_growth,
        workbook_delta_mm3=workbook_delta,
        workbook_growth_pct=workbook_growth,
        delta_delta_mm3=delta_delta,
    )


def build_volume_report(
    patients: list[dict[str, Any]],
    *,
    compare_ucsf_workbook: bool,
    workbook_csv: Path = UCSF_MASTER_CSV,
) -> tuple[list[PatientVolumeRow], dict[str, Any]]:
    ucsf_patients = [patient for patient in patients if is_ucsf_dataset(patient.get("dataset_key"))]
    non_ucsf = [patient for patient in patients if not is_ucsf_dataset(patient.get("dataset_key"))]

    if compare_ucsf_workbook and non_ucsf:
        skipped = ", ".join(str(p["patient_id"]) for p in non_ucsf)
        print(
            f"Note: UCSF workbook comparison skipped for non-UCSF patients: {skipped}",
            file=sys.stderr,
        )

    workbook: dict[str, dict[str, float]] | None = None
    if compare_ucsf_workbook:
        if not ucsf_patients:
            raise ValueError("No UCSF patients in selection; workbook comparison not applicable.")
        workbook = load_ucsf_workbook(workbook_csv)
        target_patients = ucsf_patients
    else:
        target_patients = patients

    rows: list[PatientVolumeRow] = []
    for patient in target_patients:
        row = build_patient_volume_row(patient, workbook=workbook if compare_ucsf_workbook else None)
        if row is not None:
            rows.append(row)

    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "compare_ucsf_workbook": compare_ucsf_workbook,
        "workbook_csv": str(workbook_csv) if compare_ucsf_workbook else None,
        "patient_count": len(rows),
    }
    return rows, meta


def _fmt_mm3(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}%"


def _fmt_days(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}"


def format_volume_table(rows: list[PatientVolumeRow], *, compare_ucsf_workbook: bool) -> str:
    if not rows:
        return "No longitudinal volume rows (need baseline + followup segmentations)."

    headers = [
        "Patient",
        "Grade",
        "IDH",
        "Baseline WT (mm³)",
        "Follow-up WT (mm³)",
        "Δ volume (mm³)",
        "Δ %",
        "Days",
    ]
    if compare_ucsf_workbook:
        headers[3:7] = [
            "Baseline comp",
            "Baseline book",
            "Follow-up comp",
            "Follow-up book",
            "Δ comp",
            "Δ book",
            "Δ mismatch",
            "Δ % comp",
            "Δ % book",
        ]

    lines = ["\t".join(headers)]
    for row in rows:
        assert row.baseline is not None and row.followup is not None
        if compare_ucsf_workbook:
            lines.append(
                "\t".join(
                    [
                        row.patient_id,
                        str(row.grade or "—"),
                        row.idh_status or "—",
                        _fmt_mm3(row.baseline.computed_mm3),
                        _fmt_mm3(row.baseline.workbook_mm3),
                        _fmt_mm3(row.followup.computed_mm3),
                        _fmt_mm3(row.followup.workbook_mm3),
                        _fmt_mm3(row.computed_delta_mm3),
                        _fmt_mm3(row.workbook_delta_mm3),
                        _fmt_mm3(row.delta_delta_mm3),
                        _fmt_pct(row.computed_growth_pct),
                        _fmt_pct(row.workbook_growth_pct),
                        _fmt_days(row.interval_days),
                    ]
                )
            )
        else:
            lines.append(
                "\t".join(
                    [
                        row.patient_id,
                        str(row.grade or "—"),
                        row.idh_status or "—",
                        _fmt_mm3(row.baseline.computed_mm3),
                        _fmt_mm3(row.followup.computed_mm3),
                        _fmt_mm3(row.computed_delta_mm3),
                        _fmt_pct(row.computed_growth_pct),
                        _fmt_days(row.interval_days),
                    ]
                )
            )
    return "\n".join(lines)


def write_volume_report_json(
    path: Path,
    rows: list[PatientVolumeRow],
    meta: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        "patients": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_volume_report(
    patients: list[dict[str, Any]],
    *,
    compare_ucsf_workbook: bool = False,
    workbook_csv: Path = UCSF_MASTER_CSV,
    report_json: Path | None = None,
) -> list[PatientVolumeRow]:
    rows, meta = build_volume_report(
        patients,
        compare_ucsf_workbook=compare_ucsf_workbook,
        workbook_csv=workbook_csv,
    )
    title = "WT volume report (computed from expert seg)"
    if compare_ucsf_workbook:
        title += " vs UCSF workbook"
    print(f"\n{title}\n")
    print(format_volume_table(rows, compare_ucsf_workbook=compare_ucsf_workbook))
    if report_json is not None:
        out = write_volume_report_json(report_json, rows, meta)
        print(f"\nWrote {out}")
    return rows


def _resolve_patients_for_cli(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.patient_id:
        for entry in iter_cohort_entries(include_backups=True):
            if str(entry.get("patient_id")) == args.patient_id:
                return [entry]
        raise SystemExit(f"Patient ID not found in cohort: {args.patient_id}")

    if not args.all_primary and not args.include_backups:
        raise SystemExit("Specify --all-primary, --include-backups, or --patient-id")

    patients = list(iter_cohort_entries(include_backups=args.include_backups))
    if args.all_primary and not args.include_backups:
        patients = [patient for patient in patients if patient.get("cohort_group") == "primary"]
    return patients


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report longitudinal WT volumes from expert segmentations.",
    )
    parser.add_argument("--patient-id", help="Single patient ID")
    parser.add_argument("--all-primary", action="store_true", help="All primary cohort patients")
    parser.add_argument("--include-backups", action="store_true", help="Include backup patients")
    parser.add_argument(
        "--compare-ucsf-workbook",
        action="store_true",
        help="Compare computed volumes to UCSF Table S1 workbook (UCSF patients only)",
    )
    parser.add_argument(
        "--workbook-csv",
        type=Path,
        default=UCSF_MASTER_CSV,
        help="Path to ucsf_longitudinal_master.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Write machine-readable report JSON (default when used from export_all_raw)",
    )
    args = parser.parse_args()

    patients = _resolve_patients_for_cli(args)
    run_volume_report(
        patients,
        compare_ucsf_workbook=args.compare_ucsf_workbook,
        workbook_csv=args.workbook_csv,
        report_json=args.report_json,
    )


if __name__ == "__main__":
    main()
