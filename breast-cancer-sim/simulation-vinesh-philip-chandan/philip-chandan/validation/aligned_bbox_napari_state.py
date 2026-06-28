"""Checkpoint state for the rev2 aligned-bbox napari export queue."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

VALIDATION_DIR = Path(__file__).resolve().parent
PHILIP_CHANDAN_DIR = VALIDATION_DIR.parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent

sys.path.insert(0, str(SPIKE_ROOT))
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(VALIDATION_DIR))

from batch_job_state import (  # noqa: E402
    BatchJob,
    BatchRunState,
    JobStatus,
    RunStatus,
    finalize_run,
    init_run,
    load_run,
    mark_completed,
    mark_running,
    next_runnable_job,
    resume_or_init,
    save_run,
    should_run_job,
)

REPO_ROOT = PHILIP_CHANDAN_DIR.parents[1]
SEGMENTATION_OUT_DIR = REPO_ROOT / "data" / "processed" / "segmentation-philip-chandan"
METHOD_ID = "aligned_bbox_tumor"
DEFAULT_STATE_FILE = SEGMENTATION_OUT_DIR / ".aligned_bbox_napari.state.json"
QUEUE_SCRIPT = "run_aligned_bbox_napari_queue"


def mask_npy_path(slug: str) -> Path:
    return SEGMENTATION_OUT_DIR / f"{slug}_{METHOD_ID}_mask.npy"


def mask_json_path(slug: str) -> Path:
    return SEGMENTATION_OUT_DIR / f"{slug}_{METHOD_ID}_mask.json"


def read_mask_metadata(slug: str) -> dict[str, Any] | None:
    path = mask_json_path(slug)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_export_complete(slug: str) -> bool:
    """True when napari (or workflow) wrote a non-empty aligned-bbox mask."""
    meta = read_mask_metadata(slug)
    if meta is None:
        return False
    npy = Path(str(meta.get("mask_npy", mask_npy_path(slug))))
    if not npy.is_file():
        npy = mask_npy_path(slug)
    if not npy.is_file():
        return False
    mask_voxels = int(meta.get("mask_voxels") or meta.get("mask_voxels_in_bbox") or 0)
    if mask_voxels <= 0:
        return False
    return True


def export_outputs(slug: str) -> dict[str, str]:
    meta = read_mask_metadata(slug) or {}
    npy = Path(str(meta.get("mask_npy", mask_npy_path(slug))))
    if not npy.is_file():
        npy = mask_npy_path(slug)
    return {
        "mask_npy": str(npy),
        "mask_json": str(mask_json_path(slug)),
        "export_source": str(meta.get("export_source", "")),
        "selected_phase": str(meta.get("selected_phase", "")),
        "threshold": str(meta.get("threshold", "")),
    }


def save_queue_run(path: Path, state: BatchRunState) -> None:
    """Persist queue state with script name for monitoring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_dict()
    payload["script"] = QUEUE_SCRIPT
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def sync_completed_from_disk(state: BatchRunState) -> int:
    """Mark jobs completed when mask sidecar + .npy already exist on disk."""
    synced = 0
    for job in state.jobs:
        if job.status == JobStatus.COMPLETED:
            continue
        if not is_export_complete(job.job_id):
            continue
        mark_completed(
            state,
            job.job_id,
            duration_sec=0.0,
            outputs=export_outputs(job.job_id),
        )
        synced += 1
    return synced


def init_queue_run(
    path: Path,
    slugs: list[str],
    config: dict[str, Any],
    *,
    fresh: bool,
    resume: bool,
) -> BatchRunState:
    state = resume_or_init(path, slugs, config, fresh=fresh, resume=resume)
    sync_completed_from_disk(state)
    save_queue_run(path, state)
    return state


def pending_slugs(
    state: BatchRunState,
    *,
    force: bool = False,
    retry_failed: bool = False,
) -> list[str]:
    return [
        job.job_id
        for job in state.jobs
        if should_run_job(job, force=force, retry_failed=retry_failed)
    ]


def mark_queue_completed(
    state: BatchRunState,
    path: Path,
    slug: str,
    *,
    duration_sec: float,
    export_meta: dict[str, Any],
) -> None:
    outputs = export_outputs(slug)
    outputs["gap_voxels"] = str(export_meta.get("gap_voxels", ""))
    mark_completed(state, slug, duration_sec=duration_sec, outputs=outputs)
    save_queue_run(path, state)


def mark_queue_running(state: BatchRunState, path: Path, slug: str) -> None:
    mark_running(state, slug)
    save_queue_run(path, state)


def finalize_queue_run(state: BatchRunState, path: Path) -> None:
    finalize_run(state)
    save_queue_run(path, state)
