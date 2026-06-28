"""Run breast tumor growth using the shared handoff contract.

Main entry points:
  run_growth(volume, ...)         - volume is already a [0, 1] density field
  run_growth_from_mask(mask, ...) - expert mask -> density seed -> growth
  run_growth_from_path(path, ...) - load a .npy baseline and grow it
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
from tumor_pde_solver import solve_growth, total_volume  # noqa: E402


def run_growth(
    initial_volume: np.ndarray,
    contract_path: str | None = None,
    params: dict | None = None,
    timesteps: int | None = None,
    dt: float | None = None,
) -> list[np.ndarray]:
    """Contract-aware wrapper around ``solve_growth``."""
    spec = solver_spec(contract_path)
    merged = {**spec.get("default_params", {}), **(params or {})}
    return solve_growth(
        np.asarray(initial_volume, dtype=np.float32),
        timesteps=int(timesteps if timesteps is not None else spec["timesteps"]),
        dt=float(dt if dt is not None else spec["dt"]),
        params=merged,
    )


def run_growth_from_mask(
    mask: np.ndarray,
    contract_path: str | None = None,
    params: dict | None = None,
    timesteps: int | None = None,
    dt: float | None = None,
    **seed_kwargs,
) -> list[np.ndarray]:
    """Expert tumor mask -> continuous density seed -> growth frames."""
    seed = seed_from_mask(np.asarray(mask), **seed_kwargs)
    return run_growth(
        seed,
        contract_path=contract_path,
        params=params,
        timesteps=timesteps,
        dt=dt,
    )


def run_growth_from_path(
    baseline_path: str | Path,
    *,
    is_mask: bool = False,
    contract_path: str | None = None,
    params: dict | None = None,
    timesteps: int | None = None,
    dt: float | None = None,
    **seed_kwargs,
) -> list[np.ndarray]:
    """Load a saved breast baseline ``.npy`` and return growth frames."""
    volume = np.load(Path(baseline_path))
    if is_mask:
        return run_growth_from_mask(
            volume,
            contract_path=contract_path,
            params=params,
            timesteps=timesteps,
            dt=dt,
            **seed_kwargs,
        )
    return run_growth(
        volume,
        contract_path=contract_path,
        params=params,
        timesteps=timesteps,
        dt=dt,
    )


if __name__ == "__main__":
    from tumor_pde_solver import dummy_volume

    frames = run_growth(dummy_volume(), params={"risk_multiplier": 1.4})
    print(f"frames={len(frames)} shape={frames[0].shape}")
    print(f"volume mm^3: {total_volume(frames[0]):.0f} -> {total_volume(frames[-1]):.0f}")
