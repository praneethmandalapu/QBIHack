"""Quick sanity checks + demo for the PDE solver and drug interventions.

Run from this directory:
    python test_solver.py

No real data needed — uses dummy_volume(). Once Philip/Chandan ship
extract_volume(), swap dummy_volume() for it and these checks still apply.
"""

import numpy as np

from tumor_pde_solver import solve_growth, dummy_volume, total_volume
from growth_interventions import apply_drug


def test_shapes_and_count():
    vol = dummy_volume(shape=(30, 30, 30))
    frames = solve_growth(vol, timesteps=20, dt=0.1)
    assert len(frames) == 21, "should return timesteps + 1 frames"
    assert all(f.shape == vol.shape for f in frames), "shape must be preserved"
    assert all(f.dtype == np.float32 for f in frames), "dtype must be float32"


def test_bounded_and_stable():
    vol = dummy_volume()
    frames = solve_growth(vol, timesteps=50, dt=0.1)
    last = frames[-1]
    assert np.isfinite(last).all(), "no NaN/inf -> stable"
    assert last.min() >= 0.0 and last.max() <= 1.0, "density bounded to [0, 1]"


def test_untreated_grows():
    vol = dummy_volume()
    frames = solve_growth(vol, timesteps=40, dt=0.1)
    v0 = total_volume(frames[0])
    v1 = total_volume(frames[-1])
    assert v1 > v0, "untreated tumor should grow"


def test_risk_multiplier_speeds_growth():
    vol = dummy_volume()
    low = solve_growth(vol, 40, 0.1, {"risk_multiplier": 0.7})   # Luminal A-like
    high = solve_growth(vol, 40, 0.1, {"risk_multiplier": 1.6})  # Basal-like
    assert total_volume(high[-1]) > total_volume(low[-1]), (
        "higher risk multiplier should grow faster"
    )


def test_drug_slows_growth():
    vol = dummy_volume()
    untreated = solve_growth(vol, 40, 0.1)
    p = apply_drug(vol, "chemo", dose=0.9)
    treated = solve_growth(vol, 40, 0.1, params=p)
    assert total_volume(treated[-1]) < total_volume(untreated[-1]), (
        "chemo should yield a smaller tumor than untreated"
    )


if __name__ == "__main__":
    tests = [
        test_shapes_and_count,
        test_bounded_and_stable,
        test_untreated_grows,
        test_risk_multiplier_speeds_growth,
        test_drug_slows_growth,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")

    # Demo: print a volume-over-time curve for untreated vs chemo.
    vol = dummy_volume()
    untreated = solve_growth(vol, 30, 0.1)
    treated = solve_growth(vol, 30, 0.1, params=apply_drug(vol, "chemo", 0.9))
    print("\n  step | untreated mm^3 | chemo mm^3")
    for i in range(0, 31, 5):
        print(f"  {i:4d} | {total_volume(untreated[i]):13.0f}  | {total_volume(treated[i]):9.0f}")
    print("\nAll checks passed.")
