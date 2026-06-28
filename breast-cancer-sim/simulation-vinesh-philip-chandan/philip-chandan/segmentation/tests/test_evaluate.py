"""Tests for segmentation benchmark metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SEGMENTATION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SEGMENTATION_DIR))

from evaluate import compare_to_reference  # noqa: E402


def test_compare_to_reference_perfect_overlap():
    ref = np.zeros((8, 8, 8), dtype=np.uint8)
    ref[2:5, 2:5, 2:5] = 1
    spacing = [1.0, 1.0, 1.0]
    metrics = compare_to_reference(ref.copy(), ref, spacing)
    assert metrics["dice"] == 1.0
    assert metrics["reference_voxels"] == metrics["predicted_voxels"] == 27
    assert metrics["area_fraction_pred_over_ref"] == 1.0


def test_compare_to_reference_no_overlap():
    ref = np.zeros((8, 8, 8), dtype=np.uint8)
    ref[0, 0, 0] = 1
    pred = np.zeros((8, 8, 8), dtype=np.uint8)
    pred[-1, -1, -1] = 1
    metrics = compare_to_reference(pred, ref, [1.0, 1.0, 1.0])
    assert metrics["dice"] == 0.0
