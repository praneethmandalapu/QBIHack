"""Resumable batch-job checkpoint state for export_all_raw.

Twin copy: brain-cancer-sim/simulation-vinesh-philip-chandan/batch_job_state.py
"""

from __future__ import annotations

import hashlib
import json
import signal
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SCRIPT_NAME = "export_all_raw"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def config_fingerprint(config: dict[str, Any]) -> str:
    """Stable SHA-256 of normalized run configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BatchJob:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "outputs": dict(self.outputs),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchJob:
        return cls(
            job_id=str(data["job_id"]),
            status=JobStatus(str(data.get("status", JobStatus.PENDING.value))),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            duration_sec=data.get("duration_sec"),
            outputs=dict(data.get("outputs") or {}),
            error=data.get("error"),
        )


@dataclass
class BatchRunState:
    run_id: str
    run_status: RunStatus
    started_at: str
    updated_at: str
    config_fingerprint: str
    config: dict[str, Any]
    jobs: list[BatchJob]
    current_job_id: str | None = None
    current_job_started_at: str | None = None

    def summary(self) -> dict[str, int]:
        counts = {status.value: 0 for status in JobStatus}
        for job in self.jobs:
            counts[job.status.value] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "script": SCRIPT_NAME,
            "run_id": self.run_id,
            "run_status": self.run_status.value,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "config_fingerprint": self.config_fingerprint,
            "config": self.config,
            "current_job_id": self.current_job_id,
            "current_job_started_at": self.current_job_started_at,
            "jobs": [job.to_dict() for job in self.jobs],
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchRunState:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported checkpoint schema {data.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        jobs = [BatchJob.from_dict(entry) for entry in data.get("jobs", [])]
        return cls(
            run_id=str(data["run_id"]),
            run_status=RunStatus(str(data.get("run_status", RunStatus.RUNNING.value))),
            started_at=str(data["started_at"]),
            updated_at=str(data["updated_at"]),
            config_fingerprint=str(data["config_fingerprint"]),
            config=dict(data.get("config") or {}),
            jobs=jobs,
            current_job_id=data.get("current_job_id"),
            current_job_started_at=data.get("current_job_started_at"),
        )


def init_run(job_ids: list[str], config: dict[str, Any]) -> BatchRunState:
    now = _utc_now()
    fingerprint = config_fingerprint(config)
    return BatchRunState(
        run_id=str(uuid.uuid4()),
        run_status=RunStatus.RUNNING,
        started_at=now,
        updated_at=now,
        config_fingerprint=fingerprint,
        config=dict(config),
        jobs=[BatchJob(job_id=job_id) for job_id in job_ids],
    )


def load_run(path: Path) -> BatchRunState | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return BatchRunState.from_dict(data)


def save_run(path: Path, state: BatchRunState) -> None:
    state.updated_at = _utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _job_by_id(state: BatchRunState, job_id: str) -> BatchJob:
    for job in state.jobs:
        if job.job_id == job_id:
            return job
    raise KeyError(job_id)


def should_run_job(
    job: BatchJob,
    *,
    force: bool = False,
    retry_failed: bool = False,
) -> bool:
    if retry_failed:
        return job.status == JobStatus.FAILED
    if force:
        return job.status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED)
    if job.status == JobStatus.COMPLETED:
        return False
    if job.status == JobStatus.SKIPPED:
        return False
    if job.status == JobStatus.FAILED:
        return False
    return True


def next_runnable_job(
    state: BatchRunState,
    *,
    force: bool = False,
    retry_failed: bool = False,
) -> BatchJob | None:
    for job in state.jobs:
        if should_run_job(job, force=force, retry_failed=retry_failed):
            return job
    return None


def mark_running(state: BatchRunState, job_id: str) -> None:
    job = _job_by_id(state, job_id)
    now = _utc_now()
    job.status = JobStatus.RUNNING
    job.started_at = now
    job.finished_at = None
    job.duration_sec = None
    job.outputs = {}
    job.error = None
    state.current_job_id = job_id
    state.current_job_started_at = now
    state.run_status = RunStatus.RUNNING


def mark_completed(
    state: BatchRunState,
    job_id: str,
    *,
    duration_sec: float,
    outputs: dict[str, str],
) -> None:
    job = _job_by_id(state, job_id)
    now = _utc_now()
    job.status = JobStatus.COMPLETED
    job.finished_at = now
    job.duration_sec = round(duration_sec, 3)
    job.outputs = dict(outputs)
    job.error = None
    state.current_job_id = None
    state.current_job_started_at = None


def mark_failed(state: BatchRunState, job_id: str, *, duration_sec: float, error: str) -> None:
    job = _job_by_id(state, job_id)
    now = _utc_now()
    job.status = JobStatus.FAILED
    job.finished_at = now
    job.duration_sec = round(duration_sec, 3)
    job.error = error
    state.current_job_id = None
    state.current_job_started_at = None


def mark_skipped(state: BatchRunState, job_id: str, *, reason: str) -> None:
    job = _job_by_id(state, job_id)
    job.status = JobStatus.SKIPPED
    job.finished_at = _utc_now()
    job.error = reason


def finalize_run(state: BatchRunState) -> None:
    summary = state.summary()
    if summary[JobStatus.FAILED.value] > 0:
        state.run_status = RunStatus.FAILED
    elif summary[JobStatus.PENDING.value] > 0 or summary[JobStatus.RUNNING.value] > 0:
        state.run_status = RunStatus.INTERRUPTED
    else:
        state.run_status = RunStatus.COMPLETED
    state.current_job_id = None
    state.current_job_started_at = None


def resume_or_init(
    path: Path,
    job_ids: list[str],
    config: dict[str, Any],
    *,
    fresh: bool,
    resume: bool,
) -> BatchRunState:
    fingerprint = config_fingerprint(config)
    if fresh and path.is_file():
        path.unlink()

    if resume and path.is_file():
        existing = load_run(path)
        if existing is not None:
            if existing.config_fingerprint != fingerprint:
                raise ValueError(
                    f"Checkpoint config mismatch at {path}. "
                    "Use --fresh to start a new run."
                )
            existing_ids = [job.job_id for job in existing.jobs]
            if existing_ids != job_ids:
                raise ValueError(
                    f"Checkpoint job list mismatch at {path}. "
                    "Use --fresh to start a new run."
                )
            existing.config = dict(config)
            return existing

    return init_run(job_ids, config)


class _InterruptHandler:
    def __init__(self, state: BatchRunState, path: Path) -> None:
        self._state = state
        self._path = path
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)
        self._installed = True

    def _handle(self, signum: int, _frame: object) -> None:
        self._state.run_status = RunStatus.INTERRUPTED
        try:
            save_run(self._path, self._state)
        except OSError:
            pass
        raise SystemExit(128 + signum)


def install_interrupt_handler(state: BatchRunState, path: Path) -> _InterruptHandler:
    handler = _InterruptHandler(state, path)
    handler.install()
    return handler
