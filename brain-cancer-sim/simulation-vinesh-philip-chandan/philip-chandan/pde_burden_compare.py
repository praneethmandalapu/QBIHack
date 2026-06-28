"""Compare PDE prep tumor burden (64³ voxels) against WT volume growth spec.

PDE inputs count voxels with value > 0 after resample to 1 mm and crop to 64³.
At 1 mm isotropic spacing each voxel is ~1 mm³ when the tumor fits inside the crop.
WT volumes in ``wt_volume_report.json`` use BraTS labels 1+2+3 only (excludes label 4
resection cavity); PDE prep seeds from ``mask > 0``, so per-timepoint capture can exceed
100% when cavity labels are present. Longitudinal growth % is still the primary check:
does PDE voxel delta track ``computed_growth_pct`` from the expert WT spec?
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cohort.cohort_io import is_no_resection_cavity, iter_cohort_entries
from handoff_contract import default_grid_size
from spike_paths import PDE_INPUT_VINESH, RAW_EXTRACT_PHILIP_CHANDAN


def _no_resection_patient_ids() -> frozenset[str]:
    return frozenset(
        str(entry["patient_id"])
        for entry in iter_cohort_entries(include_backups=True)
        if is_no_resection_cavity(entry)
    )


def _excluded_resection_patient_ids() -> list[str]:
    return sorted(
        str(entry["patient_id"])
        for entry in iter_cohort_entries(include_backups=True)
        if not is_no_resection_cavity(entry)
    )

WT_VOLUME_REPORT_JSON = RAW_EXTRACT_PHILIP_CHANDAN / "wt_volume_report.json"
PDE_BURDEN_COMPARE_JSON = PDE_INPUT_VINESH / "pde_burden_compare.json"

GROWTH_MATCH_TOLERANCE_PCT = 5.0
CAPTURE_LOW_THRESHOLD_PCT = 95.0
CAPTURE_HIGH_THRESHOLD_PCT = 105.0


@dataclass(frozen=True)
class TimepointBurden:
    label: str
    wt_mm3: float | None
    pde_voxels: int | None
    pde_mm3: float | None
    capture_pct: float | None


@dataclass(frozen=True)
class PatientBurdenRow:
    patient_id: str
    grade: str | float | None
    idh_status: str | None
    interval_days: float | None
    baseline: TimepointBurden | None
    followup: TimepointBurden | None
    wt_growth_pct: float | None
    workbook_growth_pct: float | None
    pde_growth_pct: float | None
    growth_pct_delta: float | None
    growth_pct_delta_workbook: float | None
    qc_flags: tuple[str, ...]


def count_pde_tumor_voxels(pde_npy_path: Path, *, background: float = 0.0) -> int:
    volume = np.load(pde_npy_path)
    return int((volume > background).sum())


def pde_mm3_from_voxels(voxel_count: int) -> float:
    """Tumor burden in mm³ at 1 mm PDE grid spacing (one voxel ≈ one mm³)."""
    return float(voxel_count)


def growth_pct(delta: float, baseline: float) -> float | None:
    if baseline <= 0:
        return None
    return 100.0 * delta / baseline


def capture_pct(pde_mm3: float | None, wt_mm3: float | None) -> float | None:
    if pde_mm3 is None or wt_mm3 is None or wt_mm3 <= 0:
        return None
    return 100.0 * pde_mm3 / wt_mm3


def _pde_npy_path(patient_id: str, timepoint: str, *, grid_size: int | None = None) -> Path:
    size = grid_size or default_grid_size()
    return PDE_INPUT_VINESH / patient_id / f"g{size}" / f"{timepoint}.npy"


def _timepoint_burden(
    label: str,
    wt_mm3: float | None,
    patient_id: str,
    *,
    grid_size: int | None = None,
) -> TimepointBurden:
    npy_path = _pde_npy_path(patient_id, label, grid_size=grid_size)
    pde_voxels: int | None = None
    pde_mm3: float | None = None
    if npy_path.is_file():
        pde_voxels = count_pde_tumor_voxels(npy_path)
        pde_mm3 = pde_mm3_from_voxels(pde_voxels)
    return TimepointBurden(
        label=label,
        wt_mm3=wt_mm3,
        pde_voxels=pde_voxels,
        pde_mm3=pde_mm3,
        capture_pct=capture_pct(pde_mm3, wt_mm3),
    )


def _qc_flags(
    baseline: TimepointBurden | None,
    followup: TimepointBurden | None,
    growth_pct_delta: float | None,
) -> tuple[str, ...]:
    flags: list[str] = []
    for tp in (baseline, followup):
        if tp is None or tp.capture_pct is None:
            flags.append("missing_pde")
            continue
        if tp.capture_pct < CAPTURE_LOW_THRESHOLD_PCT:
            flags.append(f"crop_loss_{tp.label}")
        elif tp.capture_pct > CAPTURE_HIGH_THRESHOLD_PCT:
            flags.append(f"label_inflation_{tp.label}")
    if growth_pct_delta is not None and abs(growth_pct_delta) > GROWTH_MATCH_TOLERANCE_PCT:
        flags.append("growth_mismatch")
    if not flags:
        flags.append("ok")
    return tuple(flags)


def build_patient_burden_row(
    wt_row: dict[str, Any],
    *,
    grid_size: int | None = None,
) -> PatientBurdenRow:
    patient_id = str(wt_row["patient_id"])
    baseline_wt = (wt_row.get("baseline") or {}).get("computed_mm3")
    followup_wt = (wt_row.get("followup") or {}).get("computed_mm3")

    baseline = _timepoint_burden("baseline", baseline_wt, patient_id, grid_size=grid_size)
    followup = _timepoint_burden("followup", followup_wt, patient_id, grid_size=grid_size)

    wt_growth = wt_row.get("computed_growth_pct")
    workbook_growth = wt_row.get("workbook_growth_pct")

    pde_growth: float | None = None
    if baseline.pde_voxels is not None and followup.pde_voxels is not None:
        delta = followup.pde_voxels - baseline.pde_voxels
        pde_growth = growth_pct(float(delta), float(baseline.pde_voxels))

    growth_delta = (pde_growth - wt_growth) if pde_growth is not None and wt_growth is not None else None
    growth_delta_wb = (
        (pde_growth - workbook_growth)
        if pde_growth is not None and workbook_growth is not None
        else None
    )

    return PatientBurdenRow(
        patient_id=patient_id,
        grade=wt_row.get("grade"),
        idh_status=wt_row.get("idh_status"),
        interval_days=wt_row.get("interval_days"),
        baseline=baseline,
        followup=followup,
        wt_growth_pct=wt_growth,
        workbook_growth_pct=workbook_growth,
        pde_growth_pct=pde_growth,
        growth_pct_delta=growth_delta,
        growth_pct_delta_workbook=growth_delta_wb,
        qc_flags=_qc_flags(baseline, followup, growth_delta),
    )


def load_wt_volume_report(path: Path = WT_VOLUME_REPORT_JSON) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_pde_burden_report(
    wt_report: dict[str, Any] | None = None,
    *,
    grid_size: int | None = None,
) -> dict[str, Any]:
    report = wt_report if wt_report is not None else load_wt_volume_report()
    if report is None:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "patient_count": 0,
            "patients": [],
            "notes": "wt_volume_report.json not found",
        }

    allowed = _no_resection_patient_ids()
    rows = [
        build_patient_burden_row(row, grid_size=grid_size)
        for row in report.get("patients", [])
        if str(row.get("patient_id")) in allowed
    ]
    excluded = _excluded_resection_patient_ids()
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "grid_size": grid_size or default_grid_size(),
        "cohort_selection": "no_resection_cavity",
        "excluded_resection_cavity_patient_ids": excluded,
        "comparison": {
            "wt_volume_source": "wt_volume_report.json computed_mm3 (labels 1+2+3)",
            "pde_burden_rule": "voxels > 0 in g64 PDE cube at 1 mm spacing",
            "growth_match_tolerance_pct": GROWTH_MATCH_TOLERANCE_PCT,
            "resection_cavity_note": (
                f"Patients with label 4 at any visit ({', '.join(excluded) or 'none'}) "
                "are omitted — PDE prep seeds mask > 0, which inflates burden vs WT."
            ),
        },
        "patient_count": len(rows),
        "patients": [asdict(row) for row in rows],
    }


def write_pde_burden_report(
    path: Path = PDE_BURDEN_COMPARE_JSON,
    *,
    wt_report: dict[str, Any] | None = None,
    grid_size: int | None = None,
) -> Path:
    payload = build_pde_burden_report(wt_report, grid_size=grid_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def format_growth_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.1f}"


def format_capture(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def pdf_table_rows(burden_report: dict[str, Any]) -> list[list[str]]:
    """Rows for generate_pipeline_report PDF table."""
    rows: list[list[str]] = []
    for row in burden_report.get("patients", []):
        baseline = row.get("baseline") or {}
        followup = row.get("followup") or {}
        flags = ", ".join(row.get("qc_flags") or [])
        rows.append(
            [
                str(row.get("patient_id", "")),
                format_capture(baseline.get("capture_pct")),
                format_capture(followup.get("capture_pct")),
                f"{row.get('wt_growth_pct', 0):+.1f}%" if row.get("wt_growth_pct") is not None else "-",
                f"{row.get('pde_growth_pct', 0):+.1f}%"
                if row.get("pde_growth_pct") is not None
                else "-",
                format_growth_delta(row.get("growth_pct_delta")),
                flags,
            ]
        )
    return rows


def pdf_detail_rows(burden_report: dict[str, Any]) -> list[list[str]]:
    """Per-patient baseline/follow-up burden detail for PDF."""
    rows: list[list[str]] = []
    for row in burden_report.get("patients", []):
        pid = str(row.get("patient_id", ""))
        for tp_key, tp_label in (("baseline", "t1"), ("followup", "t2")):
            tp = row.get(tp_key) or {}
            rows.append(
                [
                    pid,
                    tp_label,
                    f"{tp.get('wt_mm3', 0):,.0f}" if tp.get("wt_mm3") is not None else "-",
                    f"{tp.get('pde_voxels', 0):,}" if tp.get("pde_voxels") is not None else "-",
                    format_capture(tp.get("capture_pct")),
                ]
            )
    return rows
