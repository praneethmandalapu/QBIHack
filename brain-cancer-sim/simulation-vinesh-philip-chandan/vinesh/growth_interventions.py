"""Apply drug and intervention logic to simulated tumor growth.

Ported from breast-cancer-sim/simulation-vinesh-philip-chandan/vinesh/growth_interventions.py
@ QBIHack 5119ad6 — drug names adapted for glioma; physics unchanged.

Drugs return a modified `params` dict for solve_growth() (rates, not direct array edits).
"""

import numpy as np

from tumor_pde_solver import DEFAULT_PARAMS


DRUG_EFFECTS: dict[str, dict] = {
    "none": {"rho_factor": 1.00, "delta_add": 0.00},
    "temozolomide": {"rho_factor": 0.45, "delta_add": 0.18},
    "radiation": {"rho_factor": 0.70, "delta_add": 0.35},
    "bevacizumab": {"rho_factor": 0.55, "delta_add": 0.08},
}


def apply_drug(
    volume: np.ndarray,
    drug: str,
    dose: float,
    params: dict | None = None,
) -> dict:
    """Return growth parameters adjusted for a drug intervention."""
    p = {**DEFAULT_PARAMS, **(params or {})}

    drug = (drug or "none").lower()
    if drug not in DRUG_EFFECTS:
        raise ValueError(
            f"Unknown drug {drug!r}; expected one of {list(DRUG_EFFECTS)}"
        )

    dose = float(np.clip(dose, 0.0, 1.0))
    effect = DRUG_EFFECTS[drug]

    rho_factor = 1.0 + (effect["rho_factor"] - 1.0) * dose
    delta_add = effect["delta_add"] * dose

    out = dict(p)
    out["rho"] = float(p["rho"]) * rho_factor
    out["delta"] = float(p["delta"]) + delta_add
    return out
