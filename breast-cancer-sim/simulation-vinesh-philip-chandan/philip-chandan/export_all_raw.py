"""Batch raw extract export for cohort patients and timepoints.

Loops ``cohort.json`` and writes Option B handoff files for Vinesh: raw ``.npy`` +
sidecar ``.json`` per slug, plus optional middle-slice QC PNGs. Uses
``export_raw_extract.export_raw_extract()`` — same contract as the spike export.

**Prerequisite:** DICOM must already be on disk under ``data/raw/tcia/`` (run
``download_tcia.py`` first if needed).

Setup (from repo root)::

    cd breast-cancer-sim
    source .venv/bin/activate   # macOS/Linux
    pip install -r requirements.txt

Windows (call venv python directly)::

    breast-cancer-sim\\.venv\\Scripts\\python.exe simulation-vinesh-philip-chandan\\philip-chandan\\export_all_raw.py --all-primary

Common commands
---------------

All primary patients, every timepoint (rev2 → four slugs)::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py --all-primary

Baselines only (two slugs)::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py \\
      --all-primary --timepoints baseline

Follow-ups only::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py \\
      --all-primary --timepoints followup

Comma-separated timepoint labels::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py \\
      --all-primary --timepoints baseline,followup

Single patient::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py \\
      --tcga-id TCGA-AR-A1AX --subtype "Luminal A" --timepoints all

Re-export without QC PNGs (faster)::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py \\
      --all-primary --no-qc

Typical workflow with downloads::

    python simulation-vinesh-philip-chandan/philip-chandan/download_tcia.py \\
      --all-primary --longitudinal
    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py --all-primary

CLI flags
---------

``--all-primary``
    Export all patients in ``cohort.json`` ``primary`` list.

``--include-backups``
    Also attempt backup/later cohort entries. Skipped with a message if
    ``cohort.json`` has no ``study_date`` for that entry.

``--tcga-id`` + ``--subtype``
    Export one patient (must exist in primary cohort when using ``--tcga-id``).

``--timepoints SELECTION`` (default: ``all``)
    ``all`` — every timepoint with a ``study_date``;
    ``baseline`` or ``followup`` — one label;
    ``baseline,followup`` — comma-separated labels (case-insensitive).

``--no-qc``
    Skip ``data/qc/slice-plots-philip-chandan/{slug}_mid-z.png``.

Outputs (gitignored)
--------------------

Per slug ``{subtype_slug}_{tcga_id}_{timepoint_label}``:

| File | Path |
|------|------|
| Raw volume | ``data/processed/raw-extract-philip-chandan/{slug}.npy`` |
| Sidecar | ``data/processed/raw-extract-philip-chandan/{slug}.json`` |
| QC plot | ``data/qc/slice-plots-philip-chandan/{slug}_mid-z.png`` |

Contract version and spacing come from ``handoff_contract.json``. Vinesh resamples
in ``vinesh/prepare_pde_input.py`` — do not normalize here.

For a one-off spike baseline only, use ``export_raw_extract.py`` instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SPIKE_ROOT))

from export_raw_extract import export_raw_extract  # noqa: E402
from qc_slice_plot import save_middle_slice_plot  # noqa: E402
from tcia_extractor import (  # noqa: E402
    iter_cohort_patients,
    list_timepoints,
    load_cohort,
    subtype_slug,
)


def parse_timepoint_selection(raw: str) -> set[str] | None:
    """Return None for 'all', else lowercase timepoint labels to include."""
    value = raw.strip().lower()
    if value == "all":
        return None
    labels = {part.strip() for part in value.split(",") if part.strip()}
    if not labels:
        raise ValueError("timepoint selection must be 'all' or one or more labels (e.g. baseline,followup)")
    return labels


def filter_timepoints(
    patient: dict[str, Any],
    selection: set[str] | None,
) -> list[dict[str, Any]]:
    """Keep cohort timepoints that have study_date and match the selection."""
    matched: list[dict[str, Any]] = []
    for timepoint in list_timepoints(patient):
        study_date = timepoint.get("study_date")
        if not study_date:
            continue
        label = str(timepoint.get("label", "")).lower()
        if selection is None or label in selection:
            matched.append(timepoint)
    return matched


def build_slug(patient: dict[str, Any], timepoint: dict[str, Any]) -> str:
    return f"{subtype_slug(patient['subtype'])}_{patient['tcga_id']}_{timepoint['label']}"


def resolve_patients(
    *,
    tcga_id: str | None,
    subtype: str | None,
    all_primary: bool,
    include_backups: bool,
) -> list[dict[str, Any]]:
    if tcga_id:
        if not subtype:
            raise SystemExit("--subtype is required with --tcga-id")
        cohort = load_cohort()
        for entry in cohort.get("primary", []):
            if entry["tcga_id"] == tcga_id:
                return [
                    {
                        "subtype": entry["subtype"],
                        "tcga_id": entry["tcga_id"],
                        "imaging": entry.get("imaging", {}),
                        "cohort_group": "primary",
                    }
                ]
        raise SystemExit(f"TCGA ID not found in primary cohort: {tcga_id}")

    if not all_primary and not include_backups:
        raise SystemExit("Specify --all-primary, --include-backups, or --tcga-id with --subtype")

    patients = list(iter_cohort_patients(include_backups=include_backups))
    if all_primary and not include_backups:
        patients = [patient for patient in patients if patient.get("cohort_group") == "primary"]
    return patients


def iter_export_jobs(
    patients: list[dict[str, Any]],
    timepoint_selection: set[str] | None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any], str]]:
    for patient in patients:
        timepoints = filter_timepoints(patient, timepoint_selection)
        if not timepoints:
            label = patient["tcga_id"]
            group = patient.get("cohort_group", "primary")
            print(
                f"SKIP {label} ({group}): no timepoints match selection "
                f"or missing study_date in cohort.json",
                file=sys.stderr,
            )
            continue
        for timepoint in timepoints:
            yield patient, timepoint, build_slug(patient, timepoint)


def export_batch(
    patients: list[dict[str, Any]],
    *,
    timepoint_selection: set[str] | None,
    skip_qc: bool = False,
) -> list[tuple[Path, Path]]:
    written: list[tuple[Path, Path]] = []
    errors: list[str] = []

    for patient, timepoint, slug in iter_export_jobs(patients, timepoint_selection):
        study_date = str(timepoint["study_date"])
        try:
            npy_path, json_path = export_raw_extract(
                patient["tcga_id"],
                patient["subtype"],
                study_date,
                slug=slug,
            )
            written.append((npy_path, json_path))
            print(f"OK {slug}: {npy_path.name}")

            if not skip_qc:
                plot_path = save_middle_slice_plot(
                    patient["tcga_id"],
                    patient["subtype"],
                    study_date,
                    slug=slug,
                )
                print(f"QC {slug}: {plot_path.name}")
        except (ValueError, OSError) as exc:
            message = f"FAIL {slug}: {exc}"
            errors.append(message)
            print(message, file=sys.stderr)

    if errors and not written:
        raise RuntimeError("\n".join(errors))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export raw DICOM extracts for cohort patients and timepoints.",
    )
    parser.add_argument("--tcga-id", help="Single TCGA barcode to export")
    parser.add_argument("--subtype", help="Subtype label, e.g. 'Luminal A'")
    parser.add_argument(
        "--all-primary",
        action="store_true",
        help="Export all primary cohort patients",
    )
    parser.add_argument(
        "--include-backups",
        action="store_true",
        help="Also include backup/later patients from cohort.json",
    )
    parser.add_argument(
        "--timepoints",
        default="all",
        metavar="SELECTION",
        help="Timepoints to export: all (default), baseline, followup, or comma-separated labels",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="Skip middle-slice PNG generation",
    )
    args = parser.parse_args()

    try:
        timepoint_selection = parse_timepoint_selection(args.timepoints)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    patients = resolve_patients(
        tcga_id=args.tcga_id,
        subtype=args.subtype,
        all_primary=args.all_primary,
        include_backups=args.include_backups,
    )
    export_batch(
        patients,
        timepoint_selection=timepoint_selection,
        skip_qc=args.no_qc,
    )


if __name__ == "__main__":
    main()
