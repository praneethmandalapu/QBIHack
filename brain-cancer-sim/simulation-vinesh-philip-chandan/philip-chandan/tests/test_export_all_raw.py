"""Tests for checkpointed export_all_raw batch loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from batch_job_state import JobStatus, load_run
from export_all_raw import ExportJob, export_batch


def _patient(patient_id: str = "100002") -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "dataset_key": "ucsf_longitudinal_glioma",
        "disease": "Glioma",
        "cohort_group": "primary",
        "timepoints": [],
    }


def _job(slug: str, label: str = "baseline") -> ExportJob:
    return ExportJob(
        patient=_patient(),
        timepoint={"label": label, "study_date": "time1"},
        slug=slug,
        mr_path=Path("/tmp/mr.nii.gz"),
        seg_path=Path("/tmp/seg.nii.gz"),
        study_date="time1",
    )


@pytest.fixture
def status_file(tmp_path: Path) -> Path:
    return tmp_path / "state.json"


def test_export_batch_writes_checkpoint_per_job(status_file: Path, tmp_path: Path) -> None:
    jobs = [
        _job("glioma_ucsf_100002_baseline", "baseline"),
        _job("glioma_ucsf_100002_followup", "followup"),
    ]

    def fake_export(mr_path: Path, seg_path: Path, *, slug: str, **kwargs: Any):
        npy = tmp_path / f"{slug}.npy"
        json_path = tmp_path / f"{slug}.json"
        npy.write_bytes(b"npy")
        json_path.write_text("{}", encoding="utf-8")
        return npy, json_path

    with (
        patch("export_all_raw.collect_export_jobs", return_value=jobs),
        patch("export_all_raw.load_cohort", return_value={"version": "rev-test"}),
        patch("export_all_raw.export_raw_extract", side_effect=fake_export),
        patch("export_all_raw._run_qc"),
    ):
        export_batch(
            [_patient()],
            timepoint_selection=None,
            skip_qc=True,
            status_file=status_file,
            fresh=True,
            all_primary=True,
            timepoints_arg="all",
        )

    state = load_run(status_file)
    assert state is not None
    assert len(state.jobs) == 2
    assert all(job.status == JobStatus.COMPLETED for job in state.jobs)


def test_export_batch_resumes_without_rerunning_completed(status_file: Path, tmp_path: Path) -> None:
    jobs = [_job("glioma_ucsf_100002_baseline")]
    calls: list[str] = []

    def fake_export(mr_path: Path, seg_path: Path, *, slug: str, **kwargs: Any):
        calls.append(slug)
        npy = tmp_path / f"{slug}.npy"
        json_path = tmp_path / f"{slug}.json"
        npy.write_bytes(b"npy")
        json_path.write_text("{}", encoding="utf-8")
        return npy, json_path

    with (
        patch("export_all_raw.collect_export_jobs", return_value=jobs),
        patch("export_all_raw.load_cohort", return_value={"version": "rev-test"}),
        patch("export_all_raw.export_raw_extract", side_effect=fake_export),
        patch("export_all_raw._run_qc"),
    ):
        export_batch(
            [_patient()],
            timepoint_selection=None,
            skip_qc=True,
            status_file=status_file,
            fresh=True,
            all_primary=True,
            timepoints_arg="all",
        )

    with (
        patch("export_all_raw.collect_export_jobs", return_value=jobs),
        patch("export_all_raw.load_cohort", return_value={"version": "rev-test"}),
        patch("export_all_raw.export_raw_extract", side_effect=fake_export),
        patch("export_all_raw._run_qc"),
    ):
        export_batch(
            [_patient()],
            timepoint_selection=None,
            skip_qc=True,
            status_file=status_file,
            fresh=False,
            resume=True,
            all_primary=True,
            timepoints_arg="all",
        )

    assert calls == ["glioma_ucsf_100002_baseline"]
