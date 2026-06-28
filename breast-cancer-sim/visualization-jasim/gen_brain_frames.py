"""Generate brain glioma growth sequences for the OncoPulse viewer (Person 3).

Two honestly-distinct scenarios, both on the *real* UCSF-LPTDG baseline
(patient 100002), both in real units (dt = 1 day, 183-day interval), both
tracked by integrated mass (the quantity Vinesh's PDE was calibrated against):

  measured : exact calibration params for 100002 (IDH-mutant) -> the real
             +2.8% diffusion-dominated course. Faithful, subtle.
  model    : Fisher-Kolmogorov growth at a rho tuned so total mass growth lands
             in the cohort's aggressive IDH-WT range (handoff sec.5: +182..+609%).
             Labeled illustrative -- a model on a real geometry, not patient
             100002's measured outcome.

Writes <slug>_frames.npy (T,Z,Y,X float32) + <slug>_frames.json (mass curve,
days, growth %). Output is gitignored (.npy); the viewer bakes values into HTML.

    ../.venv/Scripts/python.exe gen_brain_frames.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BRAIN = REPO / "brain-cancer-sim"
sys.path.insert(0, str(BRAIN / "simulation-vinesh-philip-chandan/vinesh"))
from tumor_pde_solver import solve_growth  # noqa: E402

OUT = REPO / "breast-cancer-sim/data/processed/brain-frames-jasim"
OUT.mkdir(parents=True, exist_ok=True)
BASELINE = BRAIN / "data/processed/pde-input-vinesh/glioma_ucsf_100002_baseline.npy"

INTERVAL_DAYS = 183          # real baseline->follow-up gap for 100002
TIMESTEPS = 183              # dt = 1.0 day  (CFL-stable: dt_max = 1.11 at D=0.15)
N_KEEP = 24

SCENARIOS = {
    # exact calibration for patient 100002 (calibration-philip-chandan/100002.json)
    "measured": dict(risk=0.0067, rho=0.025, D=0.15,
                     kind="measured", idh="IDH-mutant", grade="2",
                     note="Calibrated to the real 183-day follow-up (+2.8% mass)."),
    # rho tuned so mass growth matches the aggressive IDH-WT cohort range
    "model": dict(risk=1.0, rho=0.013, D=0.15,
                  kind="model", idh="high-grade (model)", grade="3–4",
                  note="Illustrative Fisher–Kolmogorov growth on a real glioma geometry."),
}


def main() -> None:
    base = np.load(BASELINE).astype(np.float32)
    voxel = 1.0
    for slug, cfg in SCENARIOS.items():
        frames = solve_growth(base, TIMESTEPS, 1.0,
                              params={"risk_multiplier": cfg["risk"], "rho": cfg["rho"],
                                      "D": cfg["D"], "delta": 0.0})
        idx = np.linspace(0, len(frames) - 1, N_KEEP).round().astype(int)
        kept = [frames[i] for i in idx]
        mass = [float(f.sum()) * voxel for f in kept]
        days = [int(i) for i in idx]                      # dt = 1 day
        m0 = mass[0] or 1.0
        meta = {
            "slug": f"glioma_100002_{slug}",
            "kind": cfg["kind"], "idh": cfg["idh"], "grade": cfg["grade"],
            "patient": "100002", "dataset": "UCSF-LPTDG",
            "interval_days": INTERVAL_DAYS, "n_frames": len(kept),
            "days": days,
            "mass": [round(m, 1) for m in mass],
            "mass_index": [round(100 * m / m0, 1) for m in mass],   # baseline = 100
            "growth_pct": round(100 * (mass[-1] - m0) / m0, 1),
            "params": {"risk_multiplier": cfg["risk"], "rho": cfg["rho"], "D": cfg["D"], "dt_days": 1.0},
            "note": cfg["note"],
        }
        np.save(OUT / f"{meta['slug']}_frames.npy", np.stack(kept).astype(np.float32))
        (OUT / f"{meta['slug']}_frames.json").write_text(json.dumps(meta, indent=2))
        print(f"{slug:9s} growth {meta['growth_pct']:+.0f}% mass  "
              f"endmax {kept[-1].max():.2f}  -> {meta['slug']}_frames.npy")


if __name__ == "__main__":
    main()
