"""Apply drug and intervention logic to simulated tumor volumes."""

import numpy as np


def apply_drug(
    volume: np.ndarray,
    drug: str,
    dose: float,
    params: dict | None = None,
) -> np.ndarray:
    """Modify growth parameters or volume based on drug intervention."""
    raise NotImplementedError
