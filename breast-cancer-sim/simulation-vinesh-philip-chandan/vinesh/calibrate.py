"""Calibrate the PDE solver against a real second timepoint (Vinesh-owned).

`solve_growth` forecasts forward from a single baseline. When Philip-Chandan
supply a *followup* scan for the same patient, we can tune the growth so the
simulation reproduces the observed baseline -> followup change, then animate a
trajectory that actually passes through the real follow-up burden.

This does NOT change solve_growth's single-timepoint contract — it wraps it.

Honesty note: with only two timepoints this is a *calibration* (a fit to the
observed change), not an out-of-sample prediction. Its scientific value is that
the calibrated growth rate comes out biologically consistent (aggressive
subtypes need net growth, indolent/treated ones need net regression).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from scipy.optimize import brentq
from skimage.filters import threshold_otsu

from tumor_pde_solver import solve_growth


def isolate_tumor(volume: np.ndarray) -> np.ndarray:
    """Return the largest connected tumor region as a continuous density field.

    Philip-Chandan's PDE inputs are ~94% nonzero (whole-breast normalized
    intensity), so we Otsu-threshold and keep the largest connected component to
    get a *localized* tumor the PDE can grow. Density values inside the tumor
    are preserved (continuous); everything else is set to 0.
    """
    vol = np.asarray(volume, dtype=np.float32)
    nonzero = vol[vol > 0]
    if nonzero.size == 0 or nonzero.max() <= nonzero.min():
        return np.zeros_like(vol)
    thresh = threshold_otsu(nonzero)
    mask = vol > thresh
    labels, n = label(mask)
    if n == 0:
        return np.zeros_like(vol)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # ignore background
    largest = sizes.argmax()
    component = labels == largest
    return np.where(component, vol, 0.0).astype(np.float32)


def tumor_burden(volume: np.ndarray) -> float:
    """Total tumor burden = integral of density (sum of voxel values).

    Used as the calibration target: it is monotonic in the growth rate and
    (unlike a thresholded voxel count) is conserved by diffusion, so the
    root-finder is well behaved.
    """
    return float(np.asarray(volume, dtype=np.float64).sum())


def tumor_volume(volume: np.ndarray, threshold: float, spacing=(1.0, 1.0, 1.0)) -> float:
    """Physical tumor volume (mm^3): voxels above `threshold` times voxel size."""
    voxel = float(np.prod(spacing))
    return float(np.count_nonzero(np.asarray(volume) >= threshold)) * voxel


def calibrate_growth(
    baseline: np.ndarray,
    followup: np.ndarray,
    timesteps: int,
    dt: float,
    base_params: dict | None = None,
    *,
    max_multiplier: float = 25.0,
    max_delta: float = 8.0,
) -> dict:
    """Tune one growth knob so simulated final burden matches the real followup.

    Growth case  (followup bigger): solve for `risk_multiplier` (death off).
    Regression case (followup smaller): solve for the death rate `delta`.

    Returns a dict with the calibrated `params` (ready for solve_growth) plus
    diagnostics: target/baseline/achieved burden, the fitted knob, and the
    regime ("growth" or "regression").
    """
    base = {**(base_params or {})}
    base_iso = isolate_tumor(baseline)
    fu_iso = isolate_tumor(followup)

    b0 = tumor_burden(base_iso)
    target = tumor_burden(fu_iso)
    if b0 <= 0:
        raise ValueError("Baseline tumor is empty after isolation; cannot calibrate.")

    def sim_burden(params: dict) -> float:
        frames = solve_growth(base_iso, timesteps, dt, params)
        return tumor_burden(frames[-1])

    if target >= b0:
        regime = "growth"

        def err(m: float) -> float:
            p = {**base, "risk_multiplier": m, "delta": 0.0}
            return sim_burden(p) - target

        knob = brentq(err, 0.0, max_multiplier, xtol=1e-3)
        params = {**base, "risk_multiplier": knob, "delta": 0.0}
        knob_name = "risk_multiplier"
    else:
        regime = "regression"
        rm = float(base.get("risk_multiplier", 1.0))

        def err(d: float) -> float:
            p = {**base, "risk_multiplier": rm, "delta": d}
            return sim_burden(p) - target

        knob = brentq(err, 0.0, max_delta, xtol=1e-3)
        params = {**base, "risk_multiplier": rm, "delta": knob}
        knob_name = "delta"

    achieved = sim_burden(params)
    return {
        "params": params,
        "regime": regime,
        "knob_name": knob_name,
        "knob_value": float(knob),
        "baseline_burden": b0,
        "target_burden": target,
        "achieved_burden": achieved,
        "burden_error_pct": 100.0 * (achieved - target) / target if target else 0.0,
        "baseline_iso": base_iso,
        "followup_iso": fu_iso,
    }


def predict_trajectory(
    baseline_iso: np.ndarray,
    params: dict,
    timesteps: int,
    dt: float,
) -> list[np.ndarray]:
    """Run the calibrated forward simulation from the isolated baseline tumor."""
    return solve_growth(baseline_iso, timesteps, dt, params)
