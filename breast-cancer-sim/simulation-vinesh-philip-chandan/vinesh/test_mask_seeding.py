"""Tests for breast mask-to-seed conversion."""

import sys
from pathlib import Path

import numpy as np

VINESH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VINESH_DIR))

from mask_seeding import seed_from_mask  # noqa: E402
from tumor_pde_solver import solve_growth, total_volume  # noqa: E402


def _sphere(shape=(48, 48, 48), radius=7):
    zz, yy, xx = np.indices(shape)
    center = [s / 2 for s in shape]
    mask = (
        (zz - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (xx - center[2]) ** 2
    ) < radius**2
    return mask.astype(np.float32)


def test_seed_subcapacity_and_in_range():
    seed = seed_from_mask(_sphere())
    assert seed.dtype == np.float32
    assert seed.min() >= 0.0
    assert seed.max() < 1.0
    assert (seed > 0).any()


def test_seed_does_not_expand_mask():
    mask = _sphere()
    seed = seed_from_mask(mask)
    assert np.all((seed > 0) <= (mask > 0))


def test_seed_growth_responds_to_risk():
    seed = seed_from_mask(_sphere())
    low = solve_growth(seed, 30, 0.1, params={"risk_multiplier": 0.7})
    high = solve_growth(seed, 30, 0.1, params={"risk_multiplier": 1.6})
    assert total_volume(high[-1]) > total_volume(low[-1])


def test_empty_mask_is_safe():
    seed = seed_from_mask(np.zeros((24, 24, 24), dtype=np.float32))
    frames = solve_growth(seed, 5, 0.1)
    assert np.isfinite(frames[-1]).all()
    assert not (frames[-1] > 0).any()


if __name__ == "__main__":
    for test in (
        test_seed_subcapacity_and_in_range,
        test_seed_does_not_expand_mask,
        test_seed_growth_responds_to_risk,
        test_empty_mask_is_safe,
    ):
        test()
        print(f"  PASS  {test.__name__}")
    print("\nAll mask-seeding checks passed.")
