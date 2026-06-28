"""Tests for checkpointed export_all_raw batch loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from batch_job_state import JobStatus, load_run
from export_all_raw import ExportJob, export_batch


def _patient(tcga_id: str = "TCGA-TEST") -> dict[str, Any]:
    return {
        "subtype": "Luminal A",
        "tcga_id": tcga_id,
        "cohort_group": "primary",
        "imaging": {},
    }


def _timepoint(label: str = "baseline", study_date: str = "2002-01-01") -> dict[str, Any]:
    return {"label": label, "study_date": study_date}


@pytest.fixture
def status_file(tmp_path: Path) -> Path:
    return tmp_path / "state.json"


def test_export_batch_writes_checkpoint_per_job(status_file: Path, tmp_path: Path) -> None:
    jobs = [
        ExportJob(_patient(), _timepoint(), "luminal_a_TCGA-TEST_baseline"),
        ExportJob(_patient(), _timepoint("followup", "2003-01-01"), "luminal_a_TCGA-TEST_followup"),
    ]

    def fake_export(tcga_id: str, subtype: str, study_date: str, *, slug: str | None = None):
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
    assert state.jobs[0].duration_sec is not None


def test_export_batch_resumes_without_rerunning_completed(status_file: Path, tmp_path: Path) -> None:
    jobs = [ExportJob(_patient(), _timepoint(), "luminal_a_TCGA-TEST_baseline")]
    calls: list[str] = []

    def fake_export(tcga_id: str, subtype: str, study_date: str, *, slug: str | None = None):
        calls.append(slug or "")
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

    assert calls == ["luminal_a_TCGA-TEST_baseline"]
