"""Imaging-side tests for prepare_pde_input (Vinesh-owned).

Run from this directory:
    python test_prepare_pde_input.py
or with pytest from breast-cancer-sim/:
    pytest simulation-vinesh-philip-chandan/vinesh/test_prepare_pde_input.py

These isolate the *imaging* step from the *solver* step: if a real MRI looks
wrong on integration day, this tells you whether the bug is in resample/segment
(here) or in solve_growth (test_solver.py).
"""

import sys
from pathlib import Path

import numpy as np

VINESH_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = VINESH_DIR.parent
sys.path.insert(0, str(SPIKE_ROOT))
sys.path.insert(0, str(VINESH_DIR))

from handoff_contract import pde_input_spec  # noqa: E402
from prepare_pde_input import prepare_pde_input  # noqa: E402


def _synthetic_raw_extract(shape=(60, 256, 256), spacing=(3.0, 0.8594, 0.8594)):
    """A bright Gaussian blob on a dim background, anisotropic like real MR.

    Matches Philip-Chandan's RAW contract: (Z,Y,X) float32, un-normalized
    intensities (NOT scaled to [0,1]).
    """
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = (s / 2 for s in shape)
    r2 = ((zz - cz) / 6) ** 2 + ((yy - cy) / 30) ** 2 + ((xx - cx) / 30) ** 2
    raw = 300.0 + 700.0 * np.exp(-r2)  # background ~300, tumor up to ~1000
    return raw.astype(np.float32), list(spacing)


def test_output_shape_within_max():
    raw, spacing = _synthetic_raw_extract()
    out, _ = prepare_pde_input(raw, spacing)
    limit = pde_input_spec()["max_shape"]
    assert all(d <= m for d, m in zip(out.shape, limit)), (
        f"shape {out.shape} exceeds contract max_shape {limit}"
    )


def test_output_dtype_and_spacing():
    raw, spacing = _synthetic_raw_extract()
    out, out_spacing = prepare_pde_input(raw, spacing)
    assert out.dtype == np.float32
    assert out_spacing == pde_input_spec()["target_spacing_mm"]


def test_value_range_normalized():
    raw, spacing = _synthetic_raw_extract()
    out, _ = prepare_pde_input(raw, spacing)
    lo, hi = pde_input_spec()["value_range"]
    assert out.min() >= lo and out.max() <= hi, "values must stay in contract range"


def test_tumor_present_and_background_zero():
    raw, spacing = _synthetic_raw_extract()
    out, _ = prepare_pde_input(raw, spacing)
    bg = pde_input_spec()["background_value"]
    assert (out > bg).any(), "tumor voxels must be > background after Otsu"
    assert (out == bg).any(), "background voxels must exist"


def test_tumor_continuous_not_binary():
    """Tumor density must be continuous (not a flat binary mask) so the solver's
    logistic term rho*u*(1-u) can actually drive growth."""
    raw, spacing = _synthetic_raw_extract()
    out, _ = prepare_pde_input(raw, spacing)
    tumor_vals = out[out > pde_input_spec()["background_value"]]
    assert np.unique(tumor_vals).size > 2, "tumor should hold a range of densities"


def test_empty_input_does_not_crash():
    """A flat volume (no tumor) should return a valid all-background array."""
    flat = np.full((40, 64, 64), 100.0, dtype=np.float32)
    out, _ = prepare_pde_input(flat, [3.0, 0.8594, 0.8594])
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


if __name__ == "__main__":
    tests = [
        test_output_shape_within_max,
        test_output_dtype_and_spacing,
        test_value_range_normalized,
        test_tumor_present_and_background_zero,
        test_tumor_continuous_not_binary,
        test_empty_input_does_not_crash,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nAll prepare_pde_input checks passed.")
