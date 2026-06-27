"""Core PDE engine for tumor growth simulation."""

import numpy as np


def solve_growth(
    initial_volume: np.ndarray,
    timesteps: int,
    dt: float,
    params: dict | None = None,
) -> list[np.ndarray]:
    """Advance tumor volume over time using reaction-diffusion PDE."""
    raise NotImplementedError
