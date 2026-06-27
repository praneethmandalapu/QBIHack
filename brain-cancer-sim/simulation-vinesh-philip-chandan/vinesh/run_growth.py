"""Run PDE growth using defaults from handoff_contract.json."""

from __future__ import annotations

import sys
from pathlib import Path

VINESH_DIR = Path(__file__).resolve().parent
HANDOFF_ROOT = VINESH_DIR.parent
sys.path.insert(0, str(HANDOFF_ROOT))
sys.path.insert(0, str(VINESH_DIR))

from handoff_contract import solver_spec  # noqa: E402
from tumor_pde_solver import dummy_volume, solve_growth, total_volume  # noqa: E402


def run_growth(
    initial_volume,
    contract_path: str | None = None,
    params: dict | None = None,
    timesteps: int | None = None,
    dt: float | None = None,
) -> list:
    """Thin wrapper: contract timesteps/dt/defaults → solve_growth()."""
    spec = solver_spec(contract_path)
    merged = {**spec.get("default_params", {}), **(params or {})}
    return solve_growth(
        initial_volume,
        timesteps=int(timesteps if timesteps is not None else spec["timesteps"]),
        dt=float(dt if dt is not None else spec["dt"]),
        params=merged,
    )


if __name__ == "__main__":
    vol = dummy_volume()
    frames = run_growth(vol)
    print(f"contract run: {len(frames)} frames, shape={frames[0].shape}")
    print(f"  volume mm³: {total_volume(frames[0]):.0f} → {total_volume(frames[-1]):.0f}")
