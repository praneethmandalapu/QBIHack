"""Tests for mask -> PDE-seed conversion and growth responsiveness.

These exist to prevent two specific failure modes:
  (1) the "binary saturation" trap: a raw mask sits at u=1 where logistic growth
      is ~0, so the risk multiplier is muted -> the model ignores its own input;
  (2) any "Otsu-like" situation where the simulation does not respond
      proportionally and meaningfully to the parameter that is supposed to
      drive it (the risk multiplier).

Run:
    python test_mask_seeding.py
"""

import sys
from pathlib import Path

import numpy as np

VINESH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VINESH_DIR))

from mask_seeding import BRATS_LABEL_DENSITY, seed_from_mask  # noqa: E402
from tumor_pde_solver import solve_growth, total_volume  # noqa: E402


def _sphere(shape=(64, 64, 64), r=8, value=1.0):
    zz, yy, xx = np.indices(shape)
    c = [s / 2 for s in shape]
    m = (((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) < r * r)
    return (m.astype(np.float32) * value)


def _final_volume(seed, rm):
    frames = solve_growth(seed, timesteps=50, dt=0.1, params={"risk_multiplier": rm})
    return total_volume(frames[-1])


# --- basic contract -------------------------------------------------------

def test_seed_subcapacity_and_in_range():
    seed = seed_from_mask(_sphere())
    assert seed.dtype == np.float32
    assert seed.min() >= 0.0 and seed.max() <= 1.0, "must stay in [0,1]"
    assert seed.max() < 1.0, "must be strictly sub-capacity so logistic growth is active"
    assert (seed > 0).any(), "tumor must be present"


def test_extent_stays_within_mask():
    mask = _sphere()
    seed = seed_from_mask(mask, profile="ramp")
    assert np.all((seed > 0) <= (mask > 0)), "seed must not invent tumor outside the mask"


def test_multilabel_in_range_and_distinct():
    # BraTS-style nested labels: edema(2) > enhancing(4) > necrotic(1) shells.
    mask = np.zeros((64, 64, 64), np.float32)
    mask[_sphere(r=12) > 0] = 2  # edema
    mask[_sphere(r=8) > 0] = 4   # enhancing
    mask[_sphere(r=4) > 0] = 1   # necrotic
    seed = seed_from_mask(mask, profile="labels")
    assert seed.max() <= 1.0, "multi-label values must be normalized into range (contract)"
    levels = np.unique(seed[seed > 0])
    assert levels.size >= 3, "each tissue class should map to a distinct density"
    # necrotic densest, edema sparsest (biological ordering)
    assert BRATS_LABEL_DENSITY[1] > BRATS_LABEL_DENSITY[4] > BRATS_LABEL_DENSITY[2]


# --- the anti-Otsu guards -------------------------------------------------

def test_risk_multiplier_monotonic():
    """Final tumor size must STRICTLY increase with the risk multiplier.

    This is the core guard: the model must respond proportionally to the input
    that drives it, with no flat/saturated region."""
    seed = seed_from_mask(_sphere())
    vols = [_final_volume(seed, rm) for rm in (0.5, 1.0, 1.5, 2.0)]
    assert all(b > a for a, b in zip(vols, vols[1:])), f"not monotonic in risk: {vols}"


def test_risk_response_is_meaningful_and_beats_raw_binary():
    """Seeded growth must respond MORE strongly to risk than a raw binary mask,
    and the response must be non-trivial (>15%)."""
    binary = _sphere(value=1.0)            # raw mask straight into the solver
    seed = seed_from_mask(binary)          # our continuous seed

    def spread(vol):
        lo, hi = _final_volume(vol, 0.7), _final_volume(vol, 1.6)
        return (hi - lo) / lo

    raw_spread = spread(binary)
    seed_spread = spread(seed)
    assert seed_spread > 0.15, f"risk response too weak: {seed_spread:.2%}"
    assert seed_spread > raw_spread, (
        f"seed ({seed_spread:.2%}) must beat raw binary ({raw_spread:.2%})"
    )


def test_untreated_seed_grows():
    seed = seed_from_mask(_sphere())
    frames = solve_growth(seed, 50, 0.1, params={"risk_multiplier": 1.2})
    assert total_volume(frames[-1]) > total_volume(frames[0]), "untreated tumor should grow"


def test_empty_mask_is_safe():
    seed = seed_from_mask(np.zeros((32, 32, 32), np.float32))
    assert seed.dtype == np.float32 and not (seed > 0).any()
    frames = solve_growth(seed, 10, 0.1)  # must not crash
    assert np.isfinite(frames[-1]).all()


if __name__ == "__main__":
    tests = [
        test_seed_subcapacity_and_in_range,
        test_extent_stays_within_mask,
        test_multilabel_in_range_and_distinct,
        test_risk_multiplier_monotonic,
        test_risk_response_is_meaningful_and_beats_raw_binary,
        test_untreated_seed_grows,
        test_empty_mask_is_safe,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nAll mask-seeding checks passed.")
