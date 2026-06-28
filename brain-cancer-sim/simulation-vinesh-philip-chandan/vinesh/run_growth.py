"""Run PDE growth using defaults from handoff_contract.json.

Two entry points:
  run_growth(volume, ...)         - volume is already a [0,1] density field
  run_growth_from_mask(mask, ...) - expert segmentation mask -> seed -> growth

The brain pipeline hands us *expert masks* (binary or multi-label BraTS), so
run_growth_from_mask is the real entry point: it converts the mask to a
sub-carrying-capacity density via mask_seeding.seed_from_mask (keeping the
logistic term and risk multiplier active) before simulating.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

VINESH_DIR = Path(__file__).resolve().parent
HANDOFF_ROOT = VINESH_DIR.parent
sys.path.insert(0, str(HANDOFF_ROOT))
sys.path.insert(0, str(VINESH_DIR))

from handoff_contract import solver_spec  # noqa: E402
from mask_seeding import seed_from_mask  # noqa: E402
from tumor_pde_solver import dummy_volume, solve_growth, total_volume  # noqa: E402


def run_growth(
    initial_volume,
    contract_path: str | None = None,
    params: dict | None = None,
    timesteps: int | None = None,
    dt: float | None = None,
) -> list:
    """Thin wrapper: contract timesteps/dt/defaults -> solve_growth()."""
    spec = solver_spec(contract_path)
    merged = {**spec.get("default_params", {}), **(params or {})}
    return solve_growth(
        initial_volume,
        timesteps=int(timesteps if timesteps is not None else spec["timesteps"]),
        dt=float(dt if dt is not None else spec["dt"]),
        params=merged,
    )


def run_growth_from_mask(
    mask,
    contract_path: str | None = None,
    params: dict | None = None,
    timesteps: int | None = None,
    dt: float | None = None,
    **seed_kwargs,
) -> list:
    """Expert segmentation mask -> continuous seed -> contract-configured growth.

    `seed_kwargs` (peak, profile, label_density) are forwarded to
    seed_from_mask. Default profile="auto": multi-label masks map per class,
    binary masks get a distance-ramp density.
    """
    seed = seed_from_mask(np.asarray(mask), **seed_kwargs)
    return run_growth(
        seed, contract_path=contract_path, params=params, timesteps=timesteps, dt=dt
    )


def load_mask(path) -> np.ndarray:
    """Load a saved expert mask (.npy) for run_growth_from_mask."""
    return np.load(Path(path))


def _demo_multilabel_mask(shape=(64, 64, 64)) -> np.ndarray:
    """Synthetic BraTS-style nested mask (necrotic/enhancing/edema) for the demo."""
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = (s / 2 for s in shape)
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    mask = np.zeros(shape, np.float32)
    mask[r2 < 12 ** 2] = 2  # edema
    mask[r2 < 8 ** 2] = 4   # enhancing
    mask[r2 < 4 ** 2] = 1   # necrotic
    return mask


if __name__ == "__main__":
    # 1) plain density field (dummy) — unchanged path
    frames = run_growth(dummy_volume())
    print(f"contract run (dummy): {len(frames)} frames, shape={frames[0].shape}")
    print(f"  volume mm^3: {total_volume(frames[0]):.0f} -> {total_volume(frames[-1]):.0f}")

    # 2) expert-mask path: synthetic multi-label mask -> seed -> growth
    mframes = run_growth_from_mask(_demo_multilabel_mask(), params={"risk_multiplier": 1.5})
    print(f"contract run (mask):  {len(mframes)} frames, shape={mframes[0].shape}")
    print(f"  volume mm^3: {total_volume(mframes[0]):.0f} -> {total_volume(mframes[-1]):.0f}")
