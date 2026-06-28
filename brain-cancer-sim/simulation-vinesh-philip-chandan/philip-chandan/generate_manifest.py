"""Build manifest.json from raw-extract sidecars, cohort.json, and WT volume QC."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SIM_ROOT = PHILIP_CHANDAN_DIR.parent
REPO_ROOT = SIM_ROOT.parent
MANIFEST_PATH = REPO_ROOT / "data/processed/raw-extract-philip-chandan/manifest.json"
VOLUME_REPORT_PATH = REPO_ROOT / "data/processed/raw-extract-philip-chandan/wt_volume_report.json"

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SIM_ROOT))

from cohort.cohort_io import is_no_resection_cavity, iter_cohort_entries, load_cohort  # noqa: E402
from handoff_contract import contract_version, default_grid_size  # noqa: E402
from spike_paths import (  # noqa: E402
    RAW_EXTRACT_PHILIP_CHANDAN,
    resolve_pde_input_metadata,
    resolve_pde_input_npy,
    slice_plot_path,
)

MANIFEST_VERSION = "1.0.0"
DEMO_STABLE_PATIENT = "100002"
DEMO_AGGRESSIVE_PATIENT = "100118"


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _load_volume_report() -> dict[str, Any]:
    if not VOLUME_REPORT_PATH.is_file():
        return {}
    return json.loads(VOLUME_REPORT_PATH.read_text(encoding="utf-8"))


def _volume_report_by_patient(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["patient_id"]): row for row in report.get("patients", [])}


def _manifest_patient_ids() -> frozenset[str]:
    return frozenset(
        str(entry["patient_id"])
        for entry in iter_cohort_entries(include_backups=True)
        if is_no_resection_cavity(entry)
    )


def _cohort_by_patient() -> dict[str, dict[str, Any]]:
    cohort = load_cohort()
    raw_entries: dict[str, dict[str, Any]] = {}
    for group in ("primary", "patients"):
        for entry in cohort.get(group, []):
            raw_entries[str(entry["patient_id"])] = entry

    lookup: dict[str, dict[str, Any]] = {}
    for entry in iter_cohort_entries(cohort, include_backups=True):
        patient_id = str(entry["patient_id"])
        raw = raw_entries.get(patient_id, {})
        lookup[patient_id] = {
            **entry,
            "diagnosis": raw.get("diagnosis") or raw.get("who_2021_diagnosis"),
            "mgmt_status": entry.get("mgmt_status") or raw.get("mgmt_status") or raw.get("mgmt"),
            "risk_tier": raw.get("risk_tier"),
            "cohort_selection": raw.get("cohort_selection"),
        }
    return lookup


def _discover_raw_sidecars() -> list[tuple[Path, dict[str, Any]]]:
    entries: list[tuple[Path, dict[str, Any]]] = []
    for patient_dir in sorted(RAW_EXTRACT_PHILIP_CHANDAN.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("."):
            continue
        for json_path in sorted(patient_dir.glob("*.json")):
            if json_path.name in ("wt_volume_report.json", "manifest.json"):
                continue
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            if meta.get("slug"):
                entries.append((json_path, meta))
    return entries


def _volume_entry(json_path: Path, meta: dict[str, Any], cohort_row: dict[str, Any]) -> dict[str, Any]:
    slug = str(meta["slug"])
    patient_id = str(meta["patient_id"])
    timepoint = str(meta["timepoint"])
    pde_npy = resolve_pde_input_npy(slug)
    pde_json = resolve_pde_input_metadata(slug)
    pde_meta: dict[str, Any] = {}
    if pde_json.is_file():
        pde_meta = json.loads(pde_json.read_text(encoding="utf-8"))

    return {
        "slug": slug,
        "patient_id": patient_id,
        "disease": meta.get("disease") or cohort_row.get("disease", "Glioma"),
        "grade": cohort_row.get("grade"),
        "idh_status": cohort_row.get("idh_status"),
        "mgmt_status": cohort_row.get("mgmt_status"),
        "timepoint": timepoint,
        "study_date": meta.get("study_date"),
        "raw_npy": _rel(json_path.with_suffix(".npy")),
        "raw_json": _rel(json_path),
        "pde_npy": _rel(pde_npy) if pde_npy.is_file() else None,
        "pde_json": _rel(pde_json) if pde_json.is_file() else None,
        "segmentation_path": meta.get("segmentation_path"),
        "shape": meta.get("shape"),
        "spacing_mm": meta.get("spacing_mm"),
        "pde_shape": pde_meta.get("shape"),
        "pde_spacing_mm": pde_meta.get("spacing_mm"),
        "grid_size": pde_meta.get("grid_size", default_grid_size()),
        "qc_plot": _rel(slice_plot_path(slug, overlay=True)),
    }


def _patient_longitudinal(
    patient_id: str,
    volumes: list[dict[str, Any]],
    cohort_row: dict[str, Any],
    wt_row: dict[str, Any] | None,
) -> dict[str, Any]:
    by_timepoint = {str(v["timepoint"]): v for v in volumes if v["patient_id"] == patient_id}
    baseline = by_timepoint.get("baseline")
    followup = by_timepoint.get("followup")
    interval_days = None
    measured_growth_pct = None
    if wt_row:
        interval_days = wt_row.get("interval_days")
        measured_growth_pct = wt_row.get("computed_growth_pct")

    return {
        "patient_id": patient_id,
        "disease": cohort_row.get("disease", "Glioma"),
        "grade": cohort_row.get("grade"),
        "idh_status": cohort_row.get("idh_status"),
        "mgmt_status": cohort_row.get("mgmt_status"),
        "diagnosis": cohort_row.get("diagnosis"),
        "cohort_group": cohort_row.get("cohort_group"),
        "cohort_selection": cohort_row.get("cohort_selection"),
        "interval_days": round(interval_days) if interval_days is not None else None,
        "measured_growth_pct": measured_growth_pct,
        "baseline_slug": baseline["slug"] if baseline else None,
        "followup_slug": followup["slug"] if followup else None,
        "longitudinal_qc_plot": f"data/qc/slice-plots-philip-chandan/{patient_id}_longitudinal_mid-z-overlay.png",
        "calibrated_params": None,
    }


def build_manifest() -> dict[str, Any]:
    manifest_ids = _manifest_patient_ids()
    cohort_lookup = _cohort_by_patient()
    wt_lookup = _volume_report_by_patient(_load_volume_report())

    excluded_rc = sorted(
        str(entry["patient_id"])
        for entry in iter_cohort_entries(include_backups=True)
        if not is_no_resection_cavity(entry)
    )

    volumes: list[dict[str, Any]] = []
    for json_path, meta in _discover_raw_sidecars():
        patient_id = str(meta["patient_id"])
        if patient_id not in manifest_ids:
            continue
        cohort_row = cohort_lookup.get(patient_id, {})
        volumes.append(_volume_entry(json_path, meta, cohort_row))

    patient_ids = sorted({str(v["patient_id"]) for v in volumes})
    patients = [
        _patient_longitudinal(pid, volumes, cohort_lookup.get(pid, {}), wt_lookup.get(pid))
        for pid in patient_ids
    ]

    return {
        "version": MANIFEST_VERSION,
        "contract_version": contract_version(),
        "generated_at": date.today().isoformat(),
        "description": (
            f"Philip-Chandan brain imaging handoff: {len(manifest_ids)} UCSF longitudinal glioma "
            "patients without resection cavity (label 4) at baseline or follow-up. "
            "Load by slug, patient_id, or demo key."
        ),
        "cohort_selection": "no_resection_cavity",
        "excluded_resection_cavity_patient_ids": excluded_rc,
        "demo": {
            "stable": {
                "patient_id": DEMO_STABLE_PATIENT,
                "label": "stable_idh_mut_grade2",
                "notes": "IDH-mut oligodendroglioma grade 2; low measured WT growth",
            },
            "aggressive": {
                "patient_id": DEMO_AGGRESSIVE_PATIENT,
                "label": "aggressive_idh_wt_gbm",
                "notes": "IDH-WT glioblastoma grade 4; high measured WT growth",
            },
        },
        "patients": patients,
        "volumes": volumes,
    }


def main() -> None:
    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"  patients: {len(manifest['patients'])}")
    print(f"  volumes: {len(manifest['volumes'])}")


if __name__ == "__main__":
    main()
