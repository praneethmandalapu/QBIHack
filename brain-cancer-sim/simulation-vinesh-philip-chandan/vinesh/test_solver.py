"""Sanity checks for the PDE solver and glioma drug interventions."""

import numpy as np

from growth_interventions import apply_drug
from tumor_pde_solver import dummy_volume, solve_growth, total_volume


def test_shapes_and_count():
    vol = dummy_volume(shape=(30, 30, 30))
    frames = solve_growth(vol, timesteps=20, dt=0.1)
    assert len(frames) == 21
    assert all(f.shape == vol.shape for f in frames)
    assert all(f.dtype == np.float32 for f in frames)


def test_bounded_and_stable():
    vol = dummy_volume()
    frames = solve_growth(vol, timesteps=50, dt=0.1)
    last = frames[-1]
    assert np.isfinite(last).all()
    assert last.min() >= 0.0 and last.max() <= 1.0


def test_untreated_grows():
    vol = dummy_volume()
    frames = solve_growth(vol, timesteps=40, dt=0.1)
    assert total_volume(frames[-1]) > total_volume(frames[0])


def test_risk_multiplier_speeds_growth():
    vol = dummy_volume()
    low = solve_growth(vol, 40, 0.1, {"risk_multiplier": 0.7})
    high = solve_growth(vol, 40, 0.1, {"risk_multiplier": 1.6})
    assert total_volume(high[-1]) > total_volume(low[-1])


def test_temozolomide_slows_growth():
    vol = dummy_volume()
    untreated = solve_growth(vol, 40, 0.1)
    p = apply_drug(vol, "temozolomide", dose=0.9)
    treated = solve_growth(vol, 40, 0.1, params=p)
    assert total_volume(treated[-1]) < total_volume(untreated[-1])


if __name__ == "__main__":
    tests = [
        test_shapes_and_count,
        test_bounded_and_stable,
        test_untreated_grows,
        test_risk_multiplier_speeds_growth,
        test_temozolomide_slows_growth,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nAll checks passed.")
