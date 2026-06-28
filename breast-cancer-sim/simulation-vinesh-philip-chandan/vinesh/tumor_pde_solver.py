"""Core PDE engine for tumor growth simulation.

Implements a 3D Fisher-Kolmogorov (reaction-diffusion) model of tumor growth:

    du/dt = D * laplacian(u) + rho * u * (1 - u) - delta * u

where
    u      = tumor cell density field, normalized to [0, 1] (1 = carrying capacity)
    D      = diffusion coefficient (invasion / spread, mm^2 per timestep unit)
    rho    = proliferation rate (net growth), scaled by Praneeth's risk multiplier
    delta  = death rate (0 for plain growth; raised by drug interventions)

The logistic term rho*u*(1-u) caps density at carrying capacity so the tumor
saturates instead of growing forever. The death term -delta*u is how
growth_interventions.apply_drug() carves out a necrotic core.

ARRAY CONTRACT (agree with Philip/Chandan + Jasim before swapping in real data):
    shape  = (Z, Y, X)
    dtype  = float32
    values = tumor cell density in [0, 1]
    voxel spacing (mm) passed via params["spacing"]
"""

import numpy as np
from scipy.ndimage import laplace


# Default physical parameters. These are placeholders tuned to look reasonable
# on the synthetic dummy volume; retune in Phase 3 against real TCIA volumes.
DEFAULT_PARAMS: dict = {
    "D": 0.15,              # diffusion coefficient (invasion speed)
    "rho": 0.25,            # base proliferation rate
    "delta": 0.0,           # death rate (drugs raise this via apply_drug)
    "risk_multiplier": 1.0, # <-- SWAP IN PRANEETH'S VALUE: scalar from XGBoost
                            #     (Luminal A ~ low e.g. 0.7, Basal ~ high e.g. 1.6).
                            #     Multiplies rho to make aggressive subtypes grow faster.
    "spacing": (1.0, 1.0, 1.0),  # voxel spacing (mm) in (Z, Y, X)
                                  # <-- SWAP IN PHILIP'S VALUE: read from DICOM
                                  #     PixelSpacing + SliceThickness in extract_volume().
}


def _check_cfl(D: float, dt: float, spacing) -> None:
    """Guard the explicit-Euler stability (CFL) condition.

    For a 3D explicit finite-difference diffusion step to stay stable:
        dt <= dx^2 / (6 * D)
    Using the smallest voxel dimension is the conservative choice. If this
    trips, lower dt (or D) instead of letting the simulation blow up to NaN.
    """
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
    """Advance tumor volume over time using a reaction-diffusion PDE.

    Args:
        initial_volume: 3D array (Z, Y, X) of tumor density in [0, 1].
            In Phase 1-2 this comes from dummy_volume(); in Phase 3 swap in
            philip-chandan/tcia_extractor.extract_volume().
        timesteps: number of frames to simulate (each is one returned array).
        dt: time increment per step. Keep small enough to satisfy the CFL guard.
        params: overrides for DEFAULT_PARAMS (D, rho, delta, risk_multiplier,
            spacing). Vinesh/Philip's sliders and Praneeth's risk model feed in here.

    Returns:
        List of `timesteps + 1` arrays (including the initial frame), one 3D
        density field per timestep. Hand this straight to Jasim's
        render_3d.render_volume() for the animated 3D view.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    D = float(p["D"])
    # Praneeth's risk multiplier scales how aggressively the tumor proliferates.
    rho = float(p["rho"]) * float(p["risk_multiplier"])
    delta = float(p["delta"])
    spacing = p["spacing"]

    _check_cfl(D, dt, spacing)

    # Work in float32 to keep memory low when passing heavy 3D arrays around
    # (Vinesh/Philip's main concern for the Streamlit app).
    u = np.clip(initial_volume.astype(np.float32), 0.0, 1.0)

    # Mean voxel spacing squared, used to scale the discrete Laplacian into
    # real physical units. scipy's laplace() assumes unit grid spacing.
    dx2 = float(np.mean(np.square(spacing)))

    frames: list[np.ndarray] = [u.copy()]

    for _ in range(timesteps):
        # Diffusion term: D * laplacian(u), corrected for voxel spacing.
        diffusion = D * laplace(u, mode="nearest") / dx2
        # Logistic reaction term: growth that saturates at carrying capacity 1.
        reaction = rho * u * (1.0 - u)
        # Death term: drug-induced cell kill (delta = 0 when no drug applied).
        death = delta * u

        u = u + dt * (diffusion + reaction - death)
        # Density is physically bounded to [0, 1].
        u = np.clip(u, 0.0, 1.0).astype(np.float32)

        frames.append(u.copy())

    return frames


def total_volume(frame: np.ndarray, spacing=(1.0, 1.0, 1.0), threshold: float = 0.5) -> float:
    """Tumor volume (mm^3) of a single frame: voxels above `threshold` x voxel size.

    Use this to build the volume-over-time curve that Vinesh/Philip's LLM tab
    summarizes ("tumor shrank 32% under hormone therapy", etc.).
    """
    voxel_vol = float(np.prod(spacing))
    return float(np.count_nonzero(frame >= threshold)) * voxel_vol


def dummy_volume(shape=(40, 40, 40), radius: float = 8.0, seed: int = 0) -> np.ndarray:
    """Synthetic starting tumor so Vinesh can build before real data exists.

    Produces a smooth Gaussian blob centered in the grid — a stand-in for a
    real segmented tumor. Drop this the moment Philip/Chandan deliver
    extract_volume(); the array contract (shape, dtype, [0,1] range) is
    identical, so solve_growth() needs no changes.

    Args:
        shape: (Z, Y, X) grid size.
        radius: characteristic blob radius in voxels.
        seed: RNG seed for the small noise term (kept for reproducibility).
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = (s / 2 for s in shape)
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    blob = np.exp(-r2 / (2.0 * radius**2))
    # A touch of noise so it isn't perfectly symmetric (more lifelike).
    blob = blob + 0.02 * rng.standard_normal(shape)
    return np.clip(blob, 0.0, 1.0).astype(np.float32)
