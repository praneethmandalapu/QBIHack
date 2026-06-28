"""Imaging-side tests for prepare_pde_input (Vinesh-owned).

Run from brain-cancer-sim/:
    pytest simulation-vinesh-philip-chandan/vinesh/test_prepare_pde_input.py
"""

import sys
from pathlib import Path

import numpy as np

VINESH_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = VINESH_DIR.parent
sys.path.insert(0, str(SPIKE_ROOT))
sys.path.insert(0, str(VINESH_DIR))

from handoff_contract import max_shape_for_grid, pde_input_spec  # noqa: E402
from prepare_pde_input import prepare_pde_input  # noqa: E402


def _synthetic_raw_and_mask(shape=(60, 256, 256), spacing=(3.0, 0.8594, 0.8594)):
    """Bright Gaussian blob + matching expert mask, anisotropic like real MR."""
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = (s / 2 for s in shape)
    r2 = ((zz - cz) / 6) ** 2 + ((yy - cy) / 30) ** 2 + ((xx - cx) / 30) ** 2
    raw = 300.0 + 700.0 * np.exp(-r2)
    mask = (r2 < 1.0).astype(np.float32)
    return raw.astype(np.float32), list(spacing), mask


def test_output_shape_within_max():
    raw, spacing, mask = _synthetic_raw_and_mask()
    out, _ = prepare_pde_input(raw, spacing, mask)
    limit = pde_input_spec()["max_shape"]
    assert all(d <= m for d, m in zip(out.shape, limit))


def test_output_dtype_and_spacing():
    raw, spacing, mask = _synthetic_raw_and_mask()
    out, out_spacing = prepare_pde_input(raw, spacing, mask)
    assert out.dtype == np.float32
    assert out_spacing == pde_input_spec()["target_spacing_mm"]


def test_value_range_normalized():
    raw, spacing, mask = _synthetic_raw_and_mask()
    out, _ = prepare_pde_input(raw, spacing, mask)
    lo, hi = pde_input_spec()["value_range"]
    assert out.min() >= lo and out.max() <= hi


def test_tumor_present_and_background_zero():
    raw, spacing, mask = _synthetic_raw_and_mask()
    out, _ = prepare_pde_input(raw, spacing, mask)
    bg = pde_input_spec()["background_value"]
    assert (out > bg).any(), "tumor voxels must be > background inside expert mask"
    assert (out == bg).any(), "background voxels must exist outside mask"


def test_tumor_continuous_not_binary():
    raw, spacing, mask = _synthetic_raw_and_mask()
    out, _ = prepare_pde_input(raw, spacing, mask)
    tumor_vals = out[out > pde_input_spec()["background_value"]]
    assert np.unique(tumor_vals).size > 2, "tumor should hold a range of densities"


def test_empty_mask_does_not_crash():
    flat = np.full((40, 64, 64), 100.0, dtype=np.float32)
    mask = np.zeros_like(flat)
    out, _ = prepare_pde_input(flat, [3.0, 0.8594, 0.8594], mask)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_grid_size_128():
    raw, spacing, mask = _synthetic_raw_and_mask()
    out, _ = prepare_pde_input(raw, spacing, mask, grid_size=128)
    assert out.shape == max_shape_for_grid(128)


if __name__ == "__main__":
    for t in (
        test_output_shape_within_max,
        test_output_dtype_and_spacing,
        test_value_range_normalized,
        test_tumor_present_and_background_zero,
        test_tumor_continuous_not_binary,
        test_empty_mask_does_not_crash,
        test_grid_size_128,
    ):
        t()
        print(f"  PASS  {t.__name__}")
    print("\nAll prepare_pde_input checks passed.")
