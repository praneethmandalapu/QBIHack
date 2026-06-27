"""Tests for dual-backend radiomics extraction."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

STRETCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRETCH_DIR))

from extract_radiomics import (  # noqa: E402
    BACKENDS,
    PARITY_TOLERANCE,
    compare_parity,
    extract_features,
)
from prep_volume import (  # noqa: E402
    normalize_volume,
    numpy_to_sitk,
    tumor_mask_largest_component,
)

fastrad = pytest.importorskip("fastrad")


@pytest.fixture
def synthetic_sitk_pair(synthetic_raw_sphere):
    raw, spacing = synthetic_raw_sphere
    norm = normalize_volume(raw)
    mask = tumor_mask_largest_component(norm)
    return numpy_to_sitk(norm, mask, spacing)


def test_extract_pyradiomics_returns_features(synthetic_sitk_pair):
    sitk_image, sitk_mask = synthetic_sitk_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = extract_features(sitk_image, sitk_mask, backend="pyradiomics")
    assert features
    assert any(key.startswith("original_firstorder_") for key in features)
    assert any(key.startswith("original_shape_") for key in features)
    assert any(key.startswith("original_glcm_") for key in features)


def test_extract_fastrad_returns_features(synthetic_sitk_pair):
    sitk_image, sitk_mask = synthetic_sitk_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = extract_features(sitk_image, sitk_mask, backend="fastrad", device="cpu")
    assert features
    assert any(key.startswith("firstorder:") for key in features)
    assert any(key.startswith("shape:") for key in features)
    assert any(key.startswith("glcm:") for key in features)


def test_pyradiomics_fastrad_parity(synthetic_sitk_pair):
    sitk_image, sitk_mask = synthetic_sitk_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = compare_parity(sitk_image, sitk_mask, tolerance=PARITY_TOLERANCE)
    assert len(rows) >= 4
    for _name, py_val, fr_val, diff in rows:
        assert diff <= PARITY_TOLERANCE, f"{_name}: {py_val} vs {fr_val} (diff={diff})"


def test_unknown_backend_raises(synthetic_sitk_pair):
    sitk_image, sitk_mask = synthetic_sitk_pair
    with pytest.raises(ValueError, match="Unknown backend"):
        extract_features(sitk_image, sitk_mask, backend="not-a-backend")


def test_backends_constant():
    assert "pyradiomics" in BACKENDS
    assert "fastrad" in BACKENDS
