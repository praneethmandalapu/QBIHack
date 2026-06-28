"""Convert breast tumor masks into PDE initial-condition density fields.

The breast PDE handoff may already provide continuous density volumes in
``data/processed/pde-input-vinesh``. Use those directly. This helper is for the
other common case: a binary expert tumor mask that needs to become a
sub-capacity density field before Fisher-KPP growth.

The spatial extent comes from the mask. Interior density is a modeling proxy,
not a measured cell density.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt
except Exception:  # pragma: no cover - exercised when scipy is unavailable
    _distance_transform_edt = None


_MAX_DENSITY = 0.999


def _centroid_ramp(positive: np.ndarray, peak: float) -> np.ndarray:
    """Cheap fallback ramp for environments without SciPy."""
    out = np.zeros(positive.shape, dtype=np.float32)
    coords = np.argwhere(positive)
    if coords.size == 0:
        return out

    center = coords.mean(axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    max_distance = float(distances.max())
    if max_distance <= 0:
        out[positive] = peak
        return out

    values = peak * (1.0 - distances / max_distance)
    values = np.clip(values, peak * 0.15, peak)
    out[tuple(coords.T)] = values.astype(np.float32)
    return out


def seed_from_mask(
    mask: np.ndarray,
    *,
    peak: float = 0.5,
    profile: str = "ramp",
) -> np.ndarray:
    """Map an expert segmentation mask to a continuous PDE seed in [0, 1).

    Args:
        mask: 3D array in ``(Z, Y, X)`` order. Any value greater than zero is
            treated as tumor.
        peak: maximum tumor density assigned inside the mask. Values below one
            keep the logistic growth term active.
        profile: ``"ramp"`` for dense core plus lower-density rim, or
            ``"flat"`` for a uniform density inside the mask.
    """
    m = np.asarray(mask)
    positive = m > 0
    out = np.zeros(m.shape, dtype=np.float32)
    if not positive.any():
        return out

    peak = float(np.clip(peak, 1e-3, _MAX_DENSITY))
    if profile == "flat":
        out[positive] = peak
    elif profile == "ramp":
        if _distance_transform_edt is not None:
            dist = _distance_transform_edt(positive).astype(np.float32)
            dmax = float(dist.max())
            if dmax > 0:
                out = peak * (dist / dmax)
            else:
                out[positive] = peak
        else:
            out = _centroid_ramp(positive, peak)
    else:
        raise ValueError(f"Unknown profile {profile!r}")

    return np.clip(out, 0.0, _MAX_DENSITY).astype(np.float32)
