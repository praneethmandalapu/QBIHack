"""Generate the full set of PDE growth sequences + manifest for Jasim's website.

For each labeled patient in the cohort (pde_burden_compare.json), calibrate the
PDE so simulated burden matches the real followup, export a frame stack with
true-scale metadata (spacing, days_per_step, IDH/grade/real-growth), and write a
manifest.json indexing every case so the site can populate a case picker.

Usage:
    python make_jasim_sequences.py --data-dir D:/pde-input-vinesh --out D:/pde-input-vinesh/frames-for-jasim
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

VINESH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VINESH_DIR))

from export_frames import export_frames  # noqa: E402
from tumor_pde_solver import cfl_max_dt, solve_growth  # noqa: E402

DT, N_KEEP = 0.1, 21

# Horizon is tied to each patient's real follow-up interval so the invasive front
# travels a physically meaningful distance (Fisher-KPP front speed c~2*sqrt(D*rho)
# acts over the modeled time). 1 step ~ 1 day, clamped to keep runtimes sane.
TS_PER_DAY = 1.0
TS_MIN, TS_MAX = 60, 360
RHO_MULT = 1.0          # fixed baseline proliferation; invasion is the fitted knob
D_FLOOR, D_CEIL = 0.02, 1.4   # D_CEIL stays under the CFL limit at dt=0.1, 1mm


def _timesteps_for(interval_days: float | None) -> int:
    if not interval_days:
        return 180
    return int(np.clip(round(float(interval_days) * TS_PER_DAY), TS_MIN, TS_MAX))


def _find(data_dir: Path, pid: str, tp: str) -> Path | None:
    hits = [f for f in glob.glob(str(data_dir / f"*{pid}_{tp}.npy"))
            if not os.path.basename(f).startswith("._")]
    return Path(hits[0]) if hits else None


def _burden(v: np.ndarray) -> float:
    return float(v.sum())


EXTENT_FLOOR = 0.2   # density floor for spatial extent: captures the invasive rim


def _extent(v: np.ndarray, thr: float = EXTENT_FLOOR) -> float:
    """Tumor spatial EXTENT = voxels above a low density floor (incl. invasive
    margin). This is the growth calibration target: it is monotonic in the
    invasion coefficient D (more diffusion -> larger extent), so 'aggressive'
    means the front genuinely advances. A clinical-density VOLUME (>0.5) is NOT
    used here because it is non-monotonic in D (high D over-diffuses and thins
    density below 0.5), which a Sigma-u/volume target could not calibrate."""
    return float((v > thr).sum())


def calibrate(baseline: np.ndarray, followup: np.ndarray, timesteps: int) -> tuple[str, dict]:
    """Fit the PDE so the simulated tumor matches the real follow-up.

    Growth: tune the invasion coefficient D (Fisher-KPP front speed) so the
    simulated tumor EXTENT hits the follow-up extent. Regression: tune the death
    rate delta against Sigma-u. Returns (regime, params ready for solve_growth)."""
    assert DT <= cfl_max_dt(D_CEIL, (1.0, 1.0, 1.0)), "D_CEIL violates CFL at this dt"
    base_ext, target_ext = _extent(baseline), _extent(followup)
    target_sum = _burden(followup)

    if target_ext >= base_ext:
        def err(D: float) -> float:
            fr = solve_growth(baseline, timesteps, DT,
                              {"D": D, "risk_multiplier": RHO_MULT, "delta": 0.0})
            return _extent(fr[-1]) - target_ext
        lo, hi = err(D_FLOOR), err(D_CEIL)
        if lo > 0:            # already at/over target with minimal invasion
            D = D_FLOOR
        elif hi < 0:          # cannot reach target even at max invasion -> clamp
            D = D_CEIL
        else:
            D = brentq(err, D_FLOOR, D_CEIL, xtol=1e-4)
        return "growth", {"D": D, "risk_multiplier": RHO_MULT, "delta": 0.0}

    # regression: shrink Sigma-u toward the follow-up via a uniform death term
    sim_sum = lambda x: _burden(solve_growth(  # noqa: E731
        baseline, timesteps, DT, {"risk_multiplier": 1.0, "delta": x})[-1])
    d = brentq(lambda x: sim_sum(x) - target_sum, 0.0, 8.0, xtol=1e-3)
    return "regression", {"risk_multiplier": 1.0, "delta": d}


def main() -> None:
    default_data = Path(__file__).resolve().parents[2] / "data" / "processed" / "pde-input-vinesh"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=default_data)
    ap.add_argument("--out", type=Path, default=default_data / "frames-for-jasim")
    args = ap.parse_args()

    cohort = json.loads((args.data_dir / "pde_burden_compare.json").read_text())
    manifest = {"generated_for": "jasim-visualization", "cases": []}

    for p in cohort["patients"]:
        pid = p["patient_id"]
        bp, fp = _find(args.data_dir, pid, "baseline"), _find(args.data_dir, pid, "followup")
        if not bp or not fp:
            print(f"SKIP {pid}: missing baseline/followup")
            continue
        baseline, followup = np.load(bp), np.load(fp)
        interval_days = p.get("interval_days")
        timesteps = _timesteps_for(interval_days)
        regime, params = calibrate(baseline, followup, timesteps)
        idh = str(p.get("idh_status"))
        grade = str(p.get("grade"))
        slug = f"glioma_{pid}_idh{idh}_grade{grade}".replace(".0", "")
        npy, js = export_frames(
            baseline, slug, args.out,
            risk_multiplier=params["risk_multiplier"],
            params={k: v for k, v in params.items() if k != "risk_multiplier"},
            timesteps=timesteps, dt=DT, n_keep=N_KEEP,
            interval_days=interval_days,
            meta_extra={
                "patient_id": pid, "disease": "glioma",
                "idh_status": idh, "grade": grade, "regime": regime,
                "fitted_D": round(float(params.get("D", 0.15)), 4),
                "fitted_delta": round(float(params.get("delta", 0.0)), 4),
                "real_growth_pct": round(p.get("wt_growth_pct", 0.0), 1),
                "baseline_mm3": p["baseline"]["wt_mm3"],
                "followup_mm3": p["followup"]["wt_mm3"],
            },
        )
        manifest["cases"].append({
            "slug": slug, "patient_id": pid, "idh_status": idh, "grade": grade,
            "regime": regime, "real_growth_pct": round(p.get("wt_growth_pct", 0.0), 1),
            "frames_file": npy.name, "meta_file": js.name,
        })
        print(f"{pid}: {regime:10s} idh={idh:3s} grade={grade:3s} -> {npy.name}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"\nwrote manifest with {len(manifest['cases'])} cases -> {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
