"""Batch raw extract export for cohort patients and timepoints.

Loops ``cohort.json`` and writes Option B handoff files for Vinesh: raw ``.npy`` +
sidecar ``.json`` per slug, plus optional middle-slice QC PNGs. Uses
``export_raw_extract.export_raw_extract()`` — same contract as the spike export.

Progress is checkpointed to ``.export_all_raw.state.json`` so runs can be
monitored, interrupted, and resumed without redoing finished slugs.

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

Resume after interrupt (default)::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py --all-primary

Clean restart::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py --all-primary --fresh

Monitor checkpoint::

    jq '{status: .run_status, current: .current_job_id, summary: .summary}' \\
      data/processed/raw-extract-philip-chandan/.export_all_raw.state.json

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

``--resume`` / ``--no-resume``
    Load checkpoint and skip completed jobs (default: resume on).

``--fresh``
    Delete prior checkpoint and start a new run.

``--force``
    Re-export every slug even if already completed.

``--retry-failed``
    Only rerun jobs marked failed in the checkpoint.

``--status-file PATH``
    Override checkpoint JSON path (default beside raw extracts).

Outputs (gitignored)
--------------------

Per slug ``{subtype_slug}_{tcga_id}_{timepoint_label}``:

| File | Path |
|------|------|
| Raw volume | ``data/processed/raw-extract-philip-chandan/{slug}.npy`` |
| Sidecar | ``data/processed/raw-extract-philip-chandan/{slug}.json`` |
| QC plot | ``data/qc/slice-plots-philip-chandan/{slug}_mid-z.png`` |
| QC overlay | ``data/qc/slice-plots-philip-chandan/{slug}_mid-z-overlay.png`` |
| Checkpoint | ``data/processed/raw-extract-philip-chandan/.export_all_raw.state.json`` |

Contract version and spacing come from ``handoff_contract.json``. Vinesh resamples
in ``vinesh/prepare_pde_input.py`` — do not normalize here.

For a one-off spike baseline only, use ``export_raw_extract.py`` instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SPIKE_ROOT))

from batch_job_state import (  # noqa: E402
    RunStatus,
    finalize_run,
    install_interrupt_handler,
    mark_completed,
    mark_failed,
    mark_running,
    next_runnable_job,
    resume_or_init,
    save_run,
)
from export_raw_extract import export_raw_extract  # noqa: E402
from qc_slice_plot import save_middle_slice_overlay_plot, save_middle_slice_plot  # noqa: E402
from spike_paths import RAW_EXTRACT_PHILIP_CHANDAN  # noqa: E402
from tcia_extractor import (  # noqa: E402
    iter_cohort_patients,
    list_timepoints,
    load_cohort,
    subtype_slug,
)

DEFAULT_STATUS_FILE = RAW_EXTRACT_PHILIP_CHANDAN / ".export_all_raw.state.json"


@dataclass(frozen=True)
class ExportJob:
    patient: dict[str, Any]
    timepoint: dict[str, Any]
    slug: str


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
) -> Iterator[ExportJob]:
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
            yield ExportJob(patient=patient, timepoint=timepoint, slug=build_slug(patient, timepoint))


def collect_export_jobs(
    patients: list[dict[str, Any]],
    timepoint_selection: set[str] | None,
) -> list[ExportJob]:
    return list(iter_export_jobs(patients, timepoint_selection))


def build_run_config(
    *,
    patients: list[dict[str, Any]],
    timepoint_selection: set[str] | None,
    skip_qc: bool,
    all_primary: bool,
    include_backups: bool,
    tcga_id: str | None,
    subtype: str | None,
    timepoints_arg: str,
    job_ids: list[str],
) -> dict[str, Any]:
    cohort = load_cohort()
    selection = sorted(timepoint_selection) if timepoint_selection is not None else ["all"]
    return {
        "cohort_version": cohort.get("version", "unknown"),
        "all_primary": all_primary,
        "include_backups": include_backups,
        "tcga_id": tcga_id,
        "subtype": subtype,
        "timepoints": timepoints_arg,
        "timepoint_labels": selection,
        "skip_qc": skip_qc,
        "job_ids": job_ids,
    }


def _run_qc(job: ExportJob, *, skip_qc: bool) -> None:
    if skip_qc:
        return

    import numpy as np

    study_date = str(job.timepoint["study_date"])
    npy_path = RAW_EXTRACT_PHILIP_CHANDAN / f"{job.slug}.npy"
    volume = np.load(npy_path)
    plot_path = save_middle_slice_plot(
        job.patient["tcga_id"],
        job.patient["subtype"],
        study_date,
        slug=job.slug,
        volume=volume,
    )
    overlay_path = save_middle_slice_overlay_plot(
        job.patient["tcga_id"],
        job.patient["subtype"],
        study_date,
        slug=job.slug,
        volume=volume,
    )
    print(f"QC {job.slug}: {plot_path.name}, {overlay_path.name}")


def _export_one_job(job: ExportJob, *, skip_qc: bool) -> tuple[Path, Path]:
    study_date = str(job.timepoint["study_date"])
    npy_path, json_path = export_raw_extract(
        job.patient["tcga_id"],
        job.patient["subtype"],
        study_date,
        slug=job.slug,
    )
    _run_qc(job, skip_qc=skip_qc)
    return npy_path, json_path


def export_batch(
    patients: list[dict[str, Any]],
    *,
    timepoint_selection: set[str] | None,
    skip_qc: bool = False,
    status_file: Path = DEFAULT_STATUS_FILE,
    fresh: bool = False,
    resume: bool = True,
    force: bool = False,
    retry_failed: bool = False,
    all_primary: bool = False,
    include_backups: bool = False,
    tcga_id: str | None = None,
    subtype: str | None = None,
    timepoints_arg: str = "all",
) -> list[tuple[Path, Path]]:
    jobs = collect_export_jobs(patients, timepoint_selection)
    if not jobs:
        raise RuntimeError("No export jobs matched the selection.")

    job_ids = [job.slug for job in jobs]
    config = build_run_config(
        patients=patients,
        timepoint_selection=timepoint_selection,
        skip_qc=skip_qc,
        all_primary=all_primary,
        include_backups=include_backups,
        tcga_id=tcga_id,
        subtype=subtype,
        timepoints_arg=timepoints_arg,
        job_ids=job_ids,
    )

    try:
        state = resume_or_init(status_file, job_ids, config, fresh=fresh, resume=resume)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    install_interrupt_handler(state, status_file)
    save_run(status_file, state)

    job_by_id = {job.slug: job for job in jobs}
    written: list[tuple[Path, Path]] = []
    errors: list[str] = []

    while True:
        batch_job = next_runnable_job(state, force=force, retry_failed=retry_failed)
        if batch_job is None:
            break

        job = job_by_id[batch_job.job_id]
        mark_running(state, batch_job.job_id)
        save_run(status_file, state)

        started = time.perf_counter()
        try:
            npy_path, json_path = _export_one_job(job, skip_qc=skip_qc)
            duration = time.perf_counter() - started
            mark_completed(
                state,
                batch_job.job_id,
                duration_sec=duration,
                outputs={"npy": str(npy_path), "json": str(json_path)},
            )
            save_run(status_file, state)
            written.append((npy_path, json_path))
            print(f"OK {job.slug}: {npy_path.name} ({duration:.1f}s)")
        except (ValueError, OSError) as exc:
            duration = time.perf_counter() - started
            message = f"FAIL {job.slug}: {exc}"
            mark_failed(state, batch_job.job_id, duration_sec=duration, error=str(exc))
            save_run(status_file, state)
            errors.append(message)
            print(message, file=sys.stderr)

    finalize_run(state)
    save_run(status_file, state)

    summary = state.summary()
    print(
        f"Run {state.run_status.value}: "
        f"completed={summary['completed']} failed={summary['failed']} "
        f"pending={summary['pending']} skipped={summary['skipped']}"
    )

    if errors and not written:
        raise RuntimeError("\n".join(errors))
    if state.run_status == RunStatus.FAILED and errors:
        print(f"Checkpoint: {status_file}", file=sys.stderr)
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
    parser.add_argument(
        "--status-file",
        type=Path,
        default=DEFAULT_STATUS_FILE,
        help="Checkpoint JSON path (default: beside raw extracts)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from checkpoint when config matches (default: on)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete prior checkpoint and start a new run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export all slugs even if already completed",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only rerun jobs marked failed in the checkpoint",
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
        status_file=args.status_file,
        fresh=args.fresh,
        resume=args.resume,
        force=args.force,
        retry_failed=args.retry_failed,
        all_primary=args.all_primary,
        include_backups=args.include_backups,
        tcga_id=args.tcga_id,
        subtype=args.subtype,
        timepoints_arg=args.timepoints,
    )


if __name__ == "__main__":
    main()
