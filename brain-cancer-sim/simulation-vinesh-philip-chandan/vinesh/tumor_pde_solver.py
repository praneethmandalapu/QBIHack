"""Core PDE engine for tumor growth simulation.

Ported from breast-cancer-sim/simulation-vinesh-philip-chandan/vinesh/tumor_pde_solver.py
@ QBIHack 5119ad6 — disease-agnostic; array contract unchanged.

Implements a 3D Fisher-Kolmogorov (reaction-diffusion) model of tumor growth:

    du/dt = D * laplacian(u) + rho * u * (1 - u) - delta * u

ARRAY CONTRACT (imaging + visualization):
    shape  = (Z, Y, X)
    dtype  = float32
    values = tumor cell density in [0, 1]
    voxel spacing (mm) passed via params["spacing"]
"""

import numpy as np
from scipy.ndimage import laplace


DEFAULT_PARAMS: dict = {
    "D": 0.15,
    "rho": 0.25,
    "delta": 0.0,
    "risk_multiplier": 1.0,  # molecular risk / grade scalar (genomics TBD)
    "spacing": (1.0, 1.0, 1.0),
}


def _check_cfl(D: float, dt: float, spacing) -> None:
    if D <= 0:
        return
    dx = min(spacing)
    dt_max = dx * dx / (6.0 * D)
    assert dt <= dt_max, (
        f"Unstable: dt={dt} exceeds CFL limit {dt_max:.4f} "
        f"(D={D}, dx={dx}). Reduce dt or D."
    )


def solve_growth(
    initial_volume: np.ndarray,
    timesteps: int,
    dt: float,
    params: dict | None = None,
) -> list[np.ndarray]:
    """Advance tumor volume over time using a reaction-diffusion PDE."""
    p = {**DEFAULT_PARAMS, **(params or {})}

    D = float(p["D"])
    rho = float(p["rho"]) * float(p["risk_multiplier"])
    delta = float(p["delta"])
    spacing = p["spacing"]

    _check_cfl(D, dt, spacing)

    u = np.clip(initial_volume.astype(np.float32), 0.0, 1.0)
    dx2 = float(np.mean(np.square(spacing)))

    frames: list[np.ndarray] = [u.copy()]

    for _ in range(timesteps):
        diffusion = D * laplace(u, mode="nearest") / dx2
        reaction = rho * u * (1.0 - u)
        death = delta * u

        u = u + dt * (diffusion + reaction - death)
        u = np.clip(u, 0.0, 1.0).astype(np.float32)
        frames.append(u.copy())

    return frames


def total_volume(frame: np.ndarray, spacing=(1.0, 1.0, 1.0), threshold: float = 0.5) -> float:
    """Tumor volume (mm^3) of a single frame."""
    voxel_vol = float(np.prod(spacing))
    return float(np.count_nonzero(frame >= threshold)) * voxel_vol


def dummy_volume(shape=(40, 40, 40), radius: float = 8.0, seed: int = 0) -> np.ndarray:
    """Synthetic starting tumor for development before real imaging data exists."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = (s / 2 for s in shape)
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    blob = np.exp(-r2 / (2.0 * radius**2))
    blob = blob + 0.02 * rng.standard_normal(shape)
    return np.clip(blob, 0.0, 1.0).astype(np.float32)
