"""Batch raw extract export for brain cohort patients and timepoints.

Loops ``cohort.json`` and writes raw ``.npy`` + sidecar ``.json`` per slug, plus
optional middle-slice QC PNGs with expert mask overlay. Progress is checkpointed
to ``.export_all_raw.state.json`` for monitor / resume.

**Prerequisite:** NIfTI must already be on disk under ``data/raw/`` (see cohort paths).

Setup (from repo root)::

    cd brain-cancer-sim
    source .venv/bin/activate
    pip install -r requirements.txt

Common commands
---------------

All primary patients, every timepoint::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py --all-primary

Resume after interrupt (default)::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py --all-primary

Clean restart::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py --all-primary --fresh

Monitor checkpoint::

    jq '{status: .run_status, current: .current_job_id, summary: .summary}' \\
      data/processed/raw-extract-philip-chandan/.export_all_raw.state.json

Single patient::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py \\
      --patient-id 100002 --timepoints all

Longitudinal WT volume table vs UCSF workbook (UCSF patients only)::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py \\
      --all-primary --compare-ucsf-workbook

Volume report only (no re-export)::

    python simulation-vinesh-philip-chandan/philip-chandan/export_all_raw.py \\
      --all-primary --volume-report-only --compare-ucsf-workbook
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SIM_ROOT = PHILIP_CHANDAN_DIR.parent

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SIM_ROOT))

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
from cohort import REPO_ROOT as COHORT_REPO_ROOT  # noqa: E402
from cohort.cohort_io import (  # noqa: E402
    filter_cohort_timepoints,
    is_no_resection_cavity,
    iter_cohort_entries,
    load_cohort,
    resolve_repo_path as _resolve_repo_path,
)
from export_raw_extract import export_raw_extract  # noqa: E402
from nifti_extractor import load_expert_mask  # noqa: E402
from qc_slice_plot import save_slice_plot  # noqa: E402
from spike_paths import RAW_EXTRACT_PHILIP_CHANDAN, segmentation_mask_path  # noqa: E402

DEFAULT_STATUS_FILE = RAW_EXTRACT_PHILIP_CHANDAN / ".export_all_raw.state.json"
DEFAULT_VOLUME_REPORT_JSON = RAW_EXTRACT_PHILIP_CHANDAN / "wt_volume_report.json"

DATASET_SLUG_TOKENS: dict[str, str] = {
    "ucsf_longitudinal_glioma": "ucsf",
    "mu_glioma_post": "mu",
    "lumiere": "lumiere",
}


@dataclass(frozen=True)
class ExportJob:
    patient: dict[str, Any]
    timepoint: dict[str, Any]
    slug: str
    mr_path: Path
    seg_path: Path
    study_date: str


def parse_timepoint_selection(raw: str) -> set[str] | None:
    value = raw.strip().lower()
    if value == "all":
        return None
    labels = {part.strip() for part in value.split(",") if part.strip()}
    if not labels:
        raise ValueError("timepoint selection must be 'all' or one or more labels (e.g. baseline,followup)")
    return labels


def dataset_slug_token(dataset_key: str) -> str:
    if dataset_key in DATASET_SLUG_TOKENS:
        return DATASET_SLUG_TOKENS[dataset_key]
    return dataset_key.lower().replace("-", "_").replace(" ", "_")


def build_slug(patient: dict[str, Any], timepoint: dict[str, Any]) -> str:
    disease = str(patient.get("disease", "glioma")).lower().replace(" ", "_")
    token = dataset_slug_token(str(patient.get("dataset_key", "unknown")))
    patient_id = patient["patient_id"]
    label = str(timepoint["label"])
    return f"{disease}_{token}_{patient_id}_{label}"


def resolve_repo_path(raw_path: str) -> Path:
    return _resolve_repo_path(raw_path, repo_root=COHORT_REPO_ROOT)


def filter_timepoints(
    patient: dict[str, Any],
    selection: set[str] | None,
) -> list[dict[str, Any]]:
    return filter_cohort_timepoints(patient, selection)


def timepoint_study_date(timepoint: dict[str, Any]) -> str:
    if timepoint.get("study_date"):
        return str(timepoint["study_date"])
    relative_day = timepoint.get("relative_day")
    if relative_day is not None:
        return f"day{relative_day}"
    return str(timepoint.get("label", "unknown"))


def resolve_patients(
    *,
    patient_id: str | None,
    all_primary: bool,
    include_backups: bool,
    no_resection_cavity: bool = False,
) -> list[dict[str, Any]]:
    if patient_id:
        for entry in iter_cohort_entries(include_backups=True):
            if str(entry.get("patient_id")) == patient_id:
                return [entry]
        raise SystemExit(f"Patient ID not found in cohort: {patient_id}")

    if no_resection_cavity:
        return [entry for entry in iter_cohort_entries(include_backups=True) if is_no_resection_cavity(entry)]

    if not all_primary and not include_backups:
        raise SystemExit(
            "Specify --all-primary, --include-backups, --no-resection-cavity, or --patient-id"
        )

    patients = list(iter_cohort_entries(include_backups=include_backups))
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
            label = patient["patient_id"]
            group = patient.get("cohort_group", "primary")
            print(
                f"SKIP {label} ({group}): no timepoints match selection "
                f"or missing mr_path/segmentation_path in cohort.json",
                file=sys.stderr,
            )
            continue
        for timepoint in timepoints:
            mr_path = resolve_repo_path(str(timepoint["mr_path"]))
            seg_path = resolve_repo_path(str(timepoint["segmentation_path"]))
            if not mr_path.is_file():
                print(f"SKIP {patient['patient_id']} {timepoint['label']}: MR missing {mr_path}", file=sys.stderr)
                continue
            if not seg_path.is_file():
                print(
                    f"SKIP {patient['patient_id']} {timepoint['label']}: seg missing {seg_path}",
                    file=sys.stderr,
                )
                continue
            slug = build_slug(patient, timepoint)
            yield ExportJob(
                patient=patient,
                timepoint=timepoint,
                slug=slug,
                mr_path=mr_path,
                seg_path=seg_path,
                study_date=timepoint_study_date(timepoint),
            )


def collect_export_jobs(
    patients: list[dict[str, Any]],
    timepoint_selection: set[str] | None,
) -> list[ExportJob]:
    return list(iter_export_jobs(patients, timepoint_selection))


def build_run_config(
    *,
    skip_qc: bool,
    all_primary: bool,
    include_backups: bool,
    patient_id: str | None,
    timepoints_arg: str,
    timepoint_selection: set[str] | None,
    job_ids: list[str],
) -> dict[str, Any]:
    cohort = load_cohort()
    selection = sorted(timepoint_selection) if timepoint_selection is not None else ["all"]
    return {
        "cohort_version": cohort.get("version", "unknown"),
        "all_primary": all_primary,
        "include_backups": include_backups,
        "patient_id": patient_id,
        "timepoints": timepoints_arg,
        "timepoint_labels": selection,
        "skip_qc": skip_qc,
        "job_ids": job_ids,
    }


def _run_qc(job: ExportJob, npy_path: Path, *, skip_qc: bool) -> None:
    if skip_qc:
        return

    import numpy as np

    volume = np.load(npy_path)
    mask_path = segmentation_mask_path(job.slug)
    mask = load_expert_mask(mask_path, volume.shape)
    plot_path = save_slice_plot(volume, mask, job.slug, overlay=False)
    overlay_path = save_slice_plot(volume, mask, job.slug, overlay=True)
    print(f"QC {job.slug}: {plot_path.name}, {overlay_path.name}")


def _export_one_job(job: ExportJob, *, skip_qc: bool) -> tuple[Path, Path]:
    dataset_key = str(job.patient.get("dataset_key", ""))
    npy_path, json_path = export_raw_extract(
        job.mr_path,
        job.seg_path,
        slug=job.slug,
        patient_id=str(job.patient["patient_id"]),
        dataset=dataset_key,
        disease=str(job.patient.get("disease", "Glioma")),
        timepoint=str(job.timepoint["label"]),
        study_date=job.study_date,
    )
    _run_qc(job, npy_path, skip_qc=skip_qc)
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
    patient_id: str | None = None,
    timepoints_arg: str = "all",
) -> list[tuple[Path, Path]]:
    jobs = collect_export_jobs(patients, timepoint_selection)
    if not jobs:
        raise RuntimeError("No export jobs matched the selection.")

    job_ids = [job.slug for job in jobs]
    config = build_run_config(
        skip_qc=skip_qc,
        all_primary=all_primary,
        include_backups=include_backups,
        patient_id=patient_id,
        timepoints_arg=timepoints_arg,
        timepoint_selection=timepoint_selection,
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


def maybe_run_volume_report(
    patients: list[dict[str, Any]],
    *,
    compare_ucsf_workbook: bool,
    volume_report_only: bool,
    report_json: Path | None,
) -> None:
    if not compare_ucsf_workbook and not volume_report_only:
        return

    from wt_volume_report import run_volume_report

    out_path = report_json
    if out_path is None and compare_ucsf_workbook:
        out_path = DEFAULT_VOLUME_REPORT_JSON

    try:
        run_volume_report(
            patients,
            compare_ucsf_workbook=compare_ucsf_workbook,
            report_json=out_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export raw NIfTI extracts for brain cohort patients and timepoints.",
    )
    parser.add_argument("--patient-id", help="Single UCSF patient ID to export")
    parser.add_argument(
        "--all-primary",
        action="store_true",
        help="Export all primary cohort patients",
    )
    parser.add_argument(
        "--include-backups",
        action="store_true",
        help="Also include backup patients from cohort.json",
    )
    parser.add_argument(
        "--no-resection-cavity",
        action="store_true",
        help="Export cohort patients with no label 4 (resection cavity) at baseline or follow-up",
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
    parser.add_argument(
        "--compare-ucsf-workbook",
        action="store_true",
        help="After export (or with --volume-report-only), print WT volume table and "
        "compare computed seg volumes to UCSF Table S1 workbook (UCSF patients only)",
    )
    parser.add_argument(
        "--volume-report-only",
        action="store_true",
        help="Skip export; only run WT volume report for selected patients",
    )
    parser.add_argument(
        "--volume-report-json",
        type=Path,
        default=None,
        help="Write volume report JSON (default: raw-extract dir/wt_volume_report.json when comparing workbook)",
    )
    args = parser.parse_args()

    if args.volume_report_only and not args.compare_ucsf_workbook:
        print(
            "Note: --volume-report-only without --compare-ucsf-workbook prints computed volumes only.",
            file=sys.stderr,
        )

    try:
        timepoint_selection = parse_timepoint_selection(args.timepoints)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    patients = resolve_patients(
        patient_id=args.patient_id,
        all_primary=args.all_primary,
        include_backups=args.include_backups,
        no_resection_cavity=args.no_resection_cavity,
    )

    if not args.volume_report_only:
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
            patient_id=args.patient_id,
            timepoints_arg=args.timepoints,
        )

    maybe_run_volume_report(
        patients,
        compare_ucsf_workbook=args.compare_ucsf_workbook,
        volume_report_only=args.volume_report_only,
        report_json=args.volume_report_json,
    )


if __name__ == "__main__":
    main()
