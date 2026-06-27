"""Convert expert segmentation masks into PDE initial-condition density fields.

The brain-cancer imaging team supplies *expert* tumor masks (binary, or
multi-label BraTS-style: 1=necrotic, 2=edema, 4=enhancing). The Fisher-KPP
solver needs a *continuous* density: a raw binary/label mask sits at or above
carrying capacity (u>=1), where the logistic term rho*u*(1-u) is ~0 — so the
tumor barely proliferates and the risk multiplier is muted (see test_mask_seeding
for the guard against exactly this). This module maps a mask to a
sub-carrying-capacity density so growth dynamics stay active and responsive.

MODELING HONESTY (the lesson from the Otsu episode):
  * The tumor's *spatial extent* comes from the real expert mask — ground truth,
    not a brightness proxy. That part is trustworthy.
  * The *density values* assigned inside the tumor are a modeling assumption —
    MRI does not measure cell density. We pick a principled profile (sub-capacity,
    with a low-density infiltrative margin: the standard glioma reaction-diffusion
    initialization), NOT a measurement. Do not present interior density as data.

Why the chosen label densities (maximize logistic growth at u=0.5):
  * enhancing  -> ~0.5  : actively proliferating (fastest logistic growth)
  * edema      -> ~0.2  : infiltrative low-density margin
  * necrotic   -> ~0.95 : dense but effectively non-proliferating (u near cap)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

# Absolute initial densities for BraTS-style integer labels (all < 1.0).
BRATS_LABEL_DENSITY: dict[int, float] = {
    1: 0.95,  # necrotic core: near carrying capacity -> ~no proliferation
    2: 0.20,  # peritumoral edema: sparse infiltrative cells
    4: 0.50,  # enhancing tumor: most actively proliferating
}

_MAX_DENSITY = 0.999  # strictly below carrying capacity so rho*u*(1-u) > 0


def seed_from_mask(
    mask: np.ndarray,
    *,
    peak: float = 0.5,
    profile: str = "auto",
    label_density: dict[int, float] | None = None,
) -> np.ndarray:
    """Map an expert segmentation mask to a continuous PDE seed density in [0, 1).

    Args:
        mask: 3D array (Z,Y,X). Background = 0. Either binary (any >0 = tumor) or
            multi-label integers (e.g. BraTS 1/2/4).
        peak: core density for binary/ramp/flat profiles. Kept < 1 so the logistic
            growth term is active. 0.5 maximizes proliferation rate.
        profile: "auto" (labels if multi-label, else ramp), "ramp"
            (distance-from-edge taper -> dense core, infiltrative margin),
            "flat" (uniform `peak`), or "labels" (per-class density).
        label_density: override BRATS_LABEL_DENSITY for "labels" profile.

    Returns:
        float32 array, same shape as `mask`, values in [0, _MAX_DENSITY],
        nonzero only inside the mask. Feed straight to solve_growth().
    """
    m = np.asarray(mask)
    out = np.zeros(m.shape, dtype=np.float32)
    positive = m > 0
    if not positive.any():
        return out  # empty mask -> all background, never crashes the solver

    peak = float(np.clip(peak, 1e-3, _MAX_DENSITY))
    pos_labels = np.unique(np.rint(m[positive]).astype(int))

    chosen = profile
    if profile == "auto":
        chosen = "labels" if pos_labels.size > 1 else "ramp"

    if chosen == "labels":
        density = label_density or BRATS_LABEL_DENSITY
        rounded = np.rint(m).astype(int)
        for lab in pos_labels:
            val = float(density.get(int(lab), peak))
            out[rounded == lab] = np.clip(val, 0.0, _MAX_DENSITY)
    elif chosen == "flat":
        out[positive] = peak
    elif chosen == "ramp":
        # Distance to nearest background: dense core, tapering infiltrative rim.
        dist = distance_transform_edt(positive).astype(np.float32)
        dmax = float(dist.max())
        if dmax > 0:
            out = (peak * (dist / dmax)).astype(np.float32)
        else:
            out[positive] = peak
    else:
        raise ValueError(f"Unknown profile {profile!r}")

    return np.clip(out, 0.0, _MAX_DENSITY).astype(np.float32)
