"""Tests for batch_job_state checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from batch_job_state import (
    BatchJob,
    JobStatus,
    RunStatus,
    config_fingerprint,
    init_run,
    load_run,
    mark_completed,
    mark_failed,
    mark_running,
    next_runnable_job,
    resume_or_init,
    save_run,
    should_run_job,
)


def test_save_load_round_trip(tmp_path: Path) -> None:
    state = init_run(["a", "b"], {"cohort_version": "rev2", "job_ids": ["a", "b"]})
    path = tmp_path / "state.json"
    save_run(path, state)
    loaded = load_run(path)
    assert loaded is not None
    assert loaded.run_id == state.run_id
    assert [job.job_id for job in loaded.jobs] == ["a", "b"]
    assert loaded.config_fingerprint == state.config_fingerprint


def test_config_fingerprint_is_stable() -> None:
    config = {"b": 2, "a": 1, "job_ids": ["x"]}
    assert config_fingerprint(config) == config_fingerprint({"a": 1, "b": 2, "job_ids": ["x"]})


def test_resume_skips_completed(tmp_path: Path) -> None:
    config = {"cohort_version": "rev2", "job_ids": ["done", "next"]}
    path = tmp_path / "state.json"
    state = init_run(["done", "next"], config)
    mark_running(state, "done")
    mark_completed(state, "done", duration_sec=1.0, outputs={"npy": "/tmp/done.npy"})
    save_run(path, state)

    resumed = resume_or_init(path, ["done", "next"], config, fresh=False, resume=True)
    assert next_runnable_job(resumed) is not None
    assert next_runnable_job(resumed).job_id == "next"


def test_stale_running_job_is_retried() -> None:
    job = BatchJob(job_id="stale", status=JobStatus.RUNNING)
    assert should_run_job(job) is True


def test_force_reruns_completed() -> None:
    job = BatchJob(job_id="done", status=JobStatus.COMPLETED)
    assert should_run_job(job, force=True) is True
    assert should_run_job(job, force=False) is False


def test_retry_failed_only_selects_failed() -> None:
    failed = BatchJob(job_id="bad", status=JobStatus.FAILED)
    pending = BatchJob(job_id="wait", status=JobStatus.PENDING)
    assert should_run_job(failed, retry_failed=True) is True
    assert should_run_job(pending, retry_failed=True) is False


def test_resume_or_init_rejects_config_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = init_run(["a"], {"cohort_version": "rev1", "job_ids": ["a"]})
    save_run(path, state)

    with pytest.raises(ValueError, match="config mismatch"):
        resume_or_init(path, ["a"], {"cohort_version": "rev2", "job_ids": ["a"]}, fresh=False, resume=True)


def test_finalize_marks_failed_when_any_job_failed() -> None:
    from batch_job_state import finalize_run

    state = init_run(["a"], {"job_ids": ["a"]})
    mark_failed(state, "a", duration_sec=0.1, error="boom")
    finalize_run(state)
    assert state.run_status == RunStatus.FAILED
