"""Apply drug and intervention logic to simulated tumor growth.

Drugs do NOT edit the density array directly. Instead they return a modified
`params` dict that solve_growth() consumes, so the physics stays in one place
(the PDE). This keeps the model honest: a drug changes *rates*, and the PDE
plays those rates forward over time.

Mapping (Vihari's sliders -> these levers):
    hormone   (e.g. tamoxifen)  -> lowers proliferation rho
    chemo                       -> lowers rho AND adds cell death (necrotic core)
    radiation                   -> strong, mostly cell death
    none                        -> untreated baseline

The death term `delta` is what produces the necrotic core that Jasim's
color_maps render in a different color from healthy/active tumor tissue.
"""

import numpy as np

from tumor_pde_solver import DEFAULT_PARAMS


# How strongly each drug pulls the growth/death levers, per unit dose.
# Dose is expected normalized to [0, 1] from Vihari's slider.
# These are hackathon placeholders — retune against literature / real runs.
DRUG_EFFECTS: dict[str, dict] = {
    "none":      {"rho_factor": 1.00, "delta_add": 0.00},
    "hormone":   {"rho_factor": 0.50, "delta_add": 0.05},  # cytostatic: slows growth
    "chemo":     {"rho_factor": 0.40, "delta_add": 0.20},  # slows + kills
    "radiation": {"rho_factor": 0.70, "delta_add": 0.35},  # mostly kills
}


def apply_drug(
    volume: np.ndarray,
    drug: str,
    dose: float,
    params: dict | None = None,
) -> dict:
    """Return growth parameters adjusted for a drug intervention.

    Args:
        volume: current 3D tumor density (Z, Y, X). Accepted for interface
            symmetry / future spatially-targeted drugs (e.g. radiation only
            hitting a sub-region); the baseline model uses it only for shape.
        drug: one of DRUG_EFFECTS keys ("none", "hormone", "chemo", "radiation").
        dose: normalized dose in [0, 1] from Vihari's slider. 0 = no effect.
        params: base params to modify (defaults to DEFAULT_PARAMS).

    Returns:
        A new params dict (rho/delta adjusted) to pass straight into
        solve_growth(). Example:
            p = apply_drug(vol, "chemo", 0.8)
            frames = solve_growth(vol, timesteps=30, dt=0.1, params=p)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    drug = (drug or "none").lower()
    if drug not in DRUG_EFFECTS:
        raise ValueError(
            f"Unknown drug {drug!r}; expected one of {list(DRUG_EFFECTS)}"
        )

    dose = float(np.clip(dose, 0.0, 1.0))
    effect = DRUG_EFFECTS[drug]

    # Interpolate the effect by dose: dose=0 -> no change, dose=1 -> full effect.
    rho_factor = 1.0 + (effect["rho_factor"] - 1.0) * dose
    delta_add = effect["delta_add"] * dose

    out = dict(p)
    out["rho"] = float(p["rho"]) * rho_factor
    out["delta"] = float(p["delta"]) + delta_add
    return out
