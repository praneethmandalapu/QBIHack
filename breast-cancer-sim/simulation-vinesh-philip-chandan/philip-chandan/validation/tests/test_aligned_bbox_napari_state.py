"""Tests for aligned-bbox napari queue checkpoint helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

VALIDATION_DIR = Path(__file__).resolve().parent.parent
PHILIP_CHANDAN_DIR = VALIDATION_DIR.parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent

sys.path.insert(0, str(SPIKE_ROOT))
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(VALIDATION_DIR))

from aligned_bbox_napari_state import (  # noqa: E402
    METHOD_ID,
    is_export_complete,
    mask_json_path,
    mask_npy_path,
    sync_completed_from_disk,
)
from batch_job_state import init_run  # noqa: E402


def test_is_export_complete_requires_npy_and_voxels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import aligned_bbox_napari_state as state_mod

    monkeypatch.setattr(state_mod, "SEGMENTATION_OUT_DIR", tmp_path)
    slug = "luminal_a_TCGA-AR-A1AX_baseline"
    npy = mask_npy_path(slug)
    meta_path = mask_json_path(slug)
    tmp_path.mkdir(parents=True, exist_ok=True)

    assert not is_export_complete(slug)

    meta_path.write_text(json.dumps({"mask_voxels": 10, "mask_npy": str(npy)}) + "\n")
    assert not is_export_complete(slug)

    np.save(npy, np.zeros((4, 4, 4), dtype=np.uint8))
    assert is_export_complete(slug)


def test_sync_completed_from_disk_marks_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import aligned_bbox_napari_state as state_mod

    monkeypatch.setattr(state_mod, "SEGMENTATION_OUT_DIR", tmp_path)
    slug = "basal_TCGA-AR-A1AQ_baseline"
    npy = tmp_path / f"{slug}_{METHOD_ID}_mask.npy"
    meta_path = tmp_path / f"{slug}_{METHOD_ID}_mask.json"
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.save(npy, np.ones((2, 2, 2), dtype=np.uint8))
    meta_path.write_text(
        json.dumps(
            {
                "mask_npy": str(npy),
                "mask_voxels": 8,
                "export_source": "view_aligned_cuboid_napari",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = init_run([slug], {"workflow": "test"})
    assert sync_completed_from_disk(state) == 1
    assert state.jobs[0].status.value == "completed"
    assert sync_completed_from_disk(state) == 0
