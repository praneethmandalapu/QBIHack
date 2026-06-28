"""Launch napari aligned-bbox QC sequentially for rev2 baseline slugs with .les.

Interactive queue: opens ``view_aligned_cuboid_napari`` for each pending slug.
Click **Export mask → .npy** in the dock when satisfied; the slug is checkpointed
so reruns skip finished patients.

Checkpoint: ``data/processed/segmentation-philip-chandan/.aligned_bbox_napari.state.json``

Setup (from repo root)::

    cd breast-cancer-sim
    source .venv/bin/activate

Run pending rev2 baselines (default — skips A1AX if already exported)::

    .venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/validation/run_aligned_bbox_napari_queue.py

List pending / completed::

    .venv/bin/python .../run_aligned_bbox_napari_queue.py --status

One slug only::

    .venv/bin/python .../run_aligned_bbox_napari_queue.py --slug basal_TCGA-AR-A1AQ_baseline

Monitor checkpoint::

    jq '{status: .run_status, summary: .summary, pending: [.jobs[] | select(.status != "completed") | .job_id]}' \\
      data/processed/segmentation-philip-chandan/.aligned_bbox_napari.state.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

VALIDATION_DIR = Path(__file__).resolve().parent
PHILIP_CHANDAN_DIR = VALIDATION_DIR.parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent

sys.path.insert(0, str(SPIKE_ROOT))
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(VALIDATION_DIR))

from aligned_bbox_napari_state import (  # noqa: E402
    DEFAULT_STATE_FILE,
    finalize_queue_run,
    init_queue_run,
    is_export_complete,
    load_run,
    mark_queue_completed,
    mark_queue_running,
    pending_slugs,
    save_queue_run,
)
from batch_job_state import JobStatus, mark_completed, mark_failed, next_runnable_job, should_run_job  # noqa: E402
from tcia_extractor import load_cohort  # noqa: E402
from view_aligned_cuboid_napari import view_aligned_cuboids  # noqa: E402
from view_les_napari import slugs_with_les  # noqa: E402


def resolve_slug_list(
    *,
    slugs: list[str] | None,
    all_rev2: bool,
) -> list[str]:
    if slugs:
        return slugs
    if all_rev2:
        return slugs_with_les()
    raise SystemExit("Provide --slug one or more times, or use --all-rev2 (default).")


def build_run_config(
    slugs: list[str],
    *,
    show_postcontrast_bright: bool,
    registration_iterations: int,
    threshold_step: float,
    save_plots: bool,
) -> dict[str, Any]:
    cohort = load_cohort()
    return {
        "cohort_version": cohort.get("version", "unknown"),
        "workflow": "aligned_bbox_napari_export",
        "slugs": slugs,
        "show_postcontrast_bright": show_postcontrast_bright,
        "registration_iterations": registration_iterations,
        "threshold_step": threshold_step,
        "save_plots": save_plots,
    }


def print_status(state_path: Path) -> None:
    state = load_run(state_path)
    if state is None:
        print(f"No checkpoint at {state_path}")
        eligible = slugs_with_les()
        print(f"Eligible rev2 baselines with .les ({len(eligible)}):")
        for slug in eligible:
            done = "done" if is_export_complete(slug) else "pending"
            print(f"  {slug}  [{done}]")
        return

    print(f"Checkpoint: {state_path}")
    print(f"Run status: {state.run_status.value}")
    print(f"Summary: {state.summary()}")
    for job in state.jobs:
        extra = ""
        if job.outputs.get("mask_npy"):
            extra = f" → {Path(job.outputs['mask_npy']).name}"
        elif job.error:
            extra = f" ({job.error})"
        print(f"  {job.job_id}: {job.status.value}{extra}")


def run_queue(
    slugs: list[str],
    *,
    status_file: Path = DEFAULT_STATE_FILE,
    fresh: bool = False,
    resume: bool = True,
    force: bool = False,
    retry_failed: bool = False,
    show_postcontrast_bright: bool = True,
    registration_iterations: int = 200,
    threshold_step: float = 0.05,
    save_plots: bool = True,
) -> None:
    if not slugs:
        raise SystemExit("No slugs to process.")

    config = build_run_config(
        slugs,
        show_postcontrast_bright=show_postcontrast_bright,
        registration_iterations=registration_iterations,
        threshold_step=threshold_step,
        save_plots=save_plots,
    )
    state = init_queue_run(status_file, slugs, config, fresh=fresh, resume=resume)

    pending = pending_slugs(state, force=force, retry_failed=retry_failed)
    if not pending:
        print("All slugs already exported. Use --force to reopen napari anyway.")
        print_status(status_file)
        return

    print(f"Queue: {len(pending)} slug(s) to review — {', '.join(pending)}")
    print(f"Checkpoint: {status_file}")
    print("Close napari when done with each case; click Export mask → .npy to mark complete.\n")

    while True:
        batch_job = next_runnable_job(state, force=force, retry_failed=retry_failed)
        if batch_job is None:
            break

        slug = batch_job.job_id
        if not force and is_export_complete(slug):
            mark_completed(
                state,
                slug,
                duration_sec=0.0,
                outputs=export_outputs_safe(slug),
            )
            save_queue_run(status_file, state)
            print(f"SKIP {slug}: mask already on disk")
            continue

        mark_queue_running(state, status_file, slug)
        print(f"\n=== Napari: {slug} ===")

        exported_meta: dict[str, Any] = {}
        started = time.perf_counter()

        def on_mask_exported(meta: dict[str, Any]) -> None:
            exported_meta.clear()
            exported_meta.update(meta)
            duration = time.perf_counter() - started
            mark_queue_completed(state, status_file, slug, duration_sec=duration, export_meta=meta)
            print(f"EXPORTED {slug}: {meta.get('mask_npy')}")

        try:
            view_aligned_cuboids(
                slug,
                registration_iterations=registration_iterations,
                save_plots=save_plots,
                threshold_step=threshold_step,
                show_postcontrast_bright=show_postcontrast_bright,
                on_mask_exported=on_mask_exported,
            )
        except Exception as exc:  # noqa: BLE001 — queue continues after napari/setup errors
            duration = time.perf_counter() - started
            mark_failed(state, slug, duration_sec=duration, error=str(exc))
            save_queue_run(status_file, state)
            print(f"FAIL {slug}: {exc}", file=sys.stderr)
            continue

        if is_export_complete(slug):
            if not exported_meta:
                duration = time.perf_counter() - started
                mark_queue_completed(
                    state,
                    status_file,
                    slug,
                    duration_sec=duration,
                    export_meta=read_mask_metadata_safe(slug) or {},
                )
            print(f"OK {slug}: export recorded")
        else:
            job = next(j for j in state.jobs if j.job_id == slug)
            if job.status != JobStatus.COMPLETED:
                job.status = JobStatus.PENDING
                job.error = "Closed napari without export"
                save_queue_run(status_file, state)
            print(f"PENDING {slug}: no export — will reopen on next queue run")

    finalize_queue_run(state, status_file)
    print("\nQueue finished.")
    print_status(status_file)


def export_outputs_safe(slug: str) -> dict[str, str]:
    from aligned_bbox_napari_state import export_outputs

    return export_outputs(slug)


def read_mask_metadata_safe(slug: str) -> dict[str, Any] | None:
    from aligned_bbox_napari_state import read_mask_metadata

    return read_mask_metadata(slug)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential napari aligned-bbox export queue (rev2 baselines with .les).",
    )
    parser.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        help="Baseline manifest slug (repeatable). Default: all rev2 baselines with .les",
    )
    parser.add_argument(
        "--all-rev2",
        action="store_true",
        default=True,
        help="All baseline slugs with local .les (default)",
    )
    parser.add_argument(
        "--no-all-rev2",
        action="store_false",
        dest="all_rev2",
        help="Require explicit --slug",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help=f"Checkpoint JSON (default: {DEFAULT_STATE_FILE.relative_to(PHILIP_CHANDAN_DIR.parents[1])})",
    )
    parser.add_argument("--fresh", action="store_true", help="Delete checkpoint and start new run")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reopen slugs marked failed in the checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reopen napari even for completed slugs",
    )
    parser.add_argument("--status", action="store_true", help="Print checkpoint and exit")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List eligible baseline slugs with .les",
    )
    parser.add_argument(
        "--no-postcontrast-bright",
        action="store_true",
        help="Do not show red P2/P3 overlay on launch",
    )
    parser.add_argument("--registration-iterations", type=int, default=200)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG curve regeneration")
    args = parser.parse_args()

    if args.list:
        for slug in slugs_with_les():
            flag = "done" if is_export_complete(slug) else "pending"
            print(f"{slug}\t{flag}")
        return

    if args.status:
        print_status(args.status_file)
        return

    slugs = resolve_slug_list(slugs=args.slugs, all_rev2=args.all_rev2)
    unknown = [slug for slug in slugs if slug not in slugs_with_les()]
    if unknown:
        eligible = ", ".join(slugs_with_les())
        parser.error(f"Slug(s) not eligible (need baseline + .les): {unknown}. Eligible: {eligible}")

    run_queue(
        slugs,
        status_file=args.status_file,
        fresh=args.fresh,
        resume=not args.no_resume,
        force=args.force,
        retry_failed=args.retry_failed,
        show_postcontrast_bright=not args.no_postcontrast_bright,
        registration_iterations=args.registration_iterations,
        threshold_step=args.threshold_step,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
