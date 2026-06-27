"""Tests for TCIA .les mask parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

STRETCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRETCH_DIR))

from load_les_mask import (  # noqa: E402
    embed_les_mask,
    load_les_mask,
    parse_les_filename,
    read_les_cuboid,
    write_synthetic_les,
)


def test_parse_les_filename_with_dce_index() -> None:
    patient, dce, lesion = parse_les_filename("TCGA-AR-A1AX-S2-1.les")
    assert patient == "TCGA-AR-A1AX"
    assert dce == 2
    assert lesion == 1


def test_parse_les_filename_without_dce_index() -> None:
    patient, dce, lesion = parse_les_filename("TCGA-BH-A0BQ-1.les")
    assert patient == "TCGA-BH-A0BQ"
    assert dce == 1
    assert lesion == 1


def test_les_round_trip(tmp_path: Path) -> None:
    volume_shape = (40, 64, 64)
    cuboid = np.zeros((5, 7, 4), dtype=np.uint8)
    cuboid[1:4, 2:5, 1:3] = 1
    metadata = {
        "y_start": 10,
        "y_end": 14,
        "x_start": 20,
        "x_end": 26,
        "z_start": 8,
        "z_end": 11,
    }
    les_path = tmp_path / "TCGA-AB-C123-S1-1.les"
    write_synthetic_les(les_path, cuboid_yxz=cuboid, **metadata)

    loaded_cuboid, loaded_meta = read_les_cuboid(les_path)
    assert loaded_meta["lesion_voxels"] == int(cuboid.sum())
    np.testing.assert_array_equal(loaded_cuboid, cuboid)

    dense, dense_meta = load_les_mask(les_path, volume_shape)
    assert dense.shape == volume_shape
    assert dense_meta["mask_voxels"] == int(cuboid.sum())
    assert dense[8:12, 10:15, 20:27].sum() == int(cuboid.sum())
    assert dense.sum() == int(cuboid.sum())


def test_real_primary_les_files_fit_raw_shapes() -> None:
    lesions_root = (
        Path(__file__).resolve().parents[4]
        / "data"
        / "raw"
        / "tcia-radiogenomics"
        / "lesions"
        / "TCGA_Segmented_Lesions_UofC"
    )
    if not lesions_root.exists():
        pytest.skip("lesions ZIP not downloaded locally")

    cases = {
        "TCGA-AR-A1AX-S2-1.les": (352, 256, 256),
        "TCGA-AR-A1AQ-S2-1.les": (464, 256, 256),
    }
    for name, shape in cases.items():
        path = lesions_root / name
        if not path.exists():
            pytest.skip(f"missing {name}")
        mask, meta = load_les_mask(path, shape)
        assert mask.shape == shape
        assert meta["mask_voxels"] > 0
