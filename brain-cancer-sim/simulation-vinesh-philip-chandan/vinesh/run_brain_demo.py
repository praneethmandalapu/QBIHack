"""One-command brain growth demo -> interactive 3D web page (physically calibrated).

Loads a real UCSF glioma baseline (expert-mask tumor cell density), runs the
Fisher-KPP PDE on a REAL-TIME, day-scaled schedule, calibrates the growth rate to
that patient's real follow-up, and renders the sequence with Jasim's render_3d.

Design choices that keep it honest (fixes for the known demo issues):
  * Real time axis    : dt is in DAYS, total steps = real baseline->followup
                        interval, so the x-axis is real days (not arbitrary
                        "sim time"). D is mm^2/day, rho is 1/day, spacing is mm.
  * Calibrated        : rho (growth) or delta (regression) is fit by root-find so
                        the simulated tumor matches the real follow-up burden.
  * Honest seed       : the initial condition is the expert-mask NORMALIZED tumor
                        cell density (a modeling proxy from MR intensity inside the
                        radiologist mask), values in [0,1] -- not a measured cell
                        count. Labeled as such.
  * Robust metric     : on-screen growth is integrated burden (threshold-FREE),
                        plus detectable volume at a detection iso that is validated
                        against the expert-mask volume -- NOT the unstable iso=0.5.

Usage (from .../simulation-vinesh-philip-chandan/vinesh):
    python run_brain_demo.py --data-dir "C:/Users/Vinesh B/Downloads/pde-handoff-vinesh/pde-input-vinesh" --patient 100118
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import webbrowser
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

VINESH_DIR = Path(__file__).resolve().parent
ROOT = VINESH_DIR.parents[1]  # brain-cancer-sim/
sys.path.insert(0, str(VINESH_DIR))
sys.path.insert(0, str(ROOT / "visualization-jasim"))

from tumor_pde_solver import solve_growth  # noqa: E402
import render_3d as r  # noqa: E402

# Diffusion coefficient (mm^2/day). Literature-scale for glioma; held fixed while
# the proliferation rate is calibrated per patient. dt=1 day satisfies CFL
# (dt <= dx^2/(6D) = 1/(6*0.1) = 1.67 day at 1 mm spacing).
D_MM2_PER_DAY = 0.10
DT_DAYS = 1.0
# Detection iso for the *visible* tumor edge. Chosen/validated so volume at this
# iso matches the expert-mask volume (see printout); avoids the iso=0.5 artifact.
DETECT_ISO = 0.15


def _find(data_dir: Path, pid: str, tp: str) -> Path:
    hits = [Path(x) for x in glob.glob(str(data_dir / f"*{pid}_{tp}.npy"))
            if not os.path.basename(x).startswith("._")]
    if not hits:
        raise FileNotFoundError(f"No {tp} .npy for patient {pid} in {data_dir}")
    return sorted(hits, key=lambda p: len(p.name))[0]


def _cohort_info(data_dir: Path, pid: str) -> dict:
    bc = data_dir / "pde_burden_compare.json"
    if bc.exists():
        for p in json.loads(bc.read_text()).get("patients", []):
            if str(p["patient_id"]) == str(pid):
                return p
    return {}


def _burden(u: np.ndarray, voxel_mm3: float) -> float:
    return float(u.sum()) * voxel_mm3  # threshold-free integrated burden


def _volume_at(u: np.ndarray, iso: float, voxel_mm3: float) -> float:
    return float((u >= iso).sum()) * voxel_mm3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--patient", default="100118")
    ap.add_argument("--D", type=float, default=D_MM2_PER_DAY)
    ap.add_argument("--interval-days", type=float, default=None,
                    help="override real baseline->followup gap (else read from cohort)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    baseline = np.load(_find(args.data_dir, args.patient, "baseline"))
    try:
        followup = np.load(_find(args.data_dir, args.patient, "followup"))
    except FileNotFoundError:
        followup = None

    info = _cohort_info(args.data_dir, args.patient)
    interval = args.interval_days or info.get("interval_days") or 60.0
    voxel_mm3 = 1.0  # data is 1 mm isotropic per contract
    steps = max(int(round(interval / DT_DAYS)), 1)

    # --- calibrate to the real follow-up DETECTABLE VOLUME ---------------------
    # We calibrate to volume at DETECT_ISO (not burden) because that iso is
    # validated against the expert-mask volume at BOTH real timepoints (see
    # printout), so the on-screen metric is the one being matched -- and it is
    # the clinically meaningful quantity (tumor extent, mm^3).
    v0 = _volume_at(baseline, DETECT_ISO, voxel_mm3)
    base_params = {"D": args.D, "risk_multiplier": 1.0, "spacing": (1, 1, 1)}

    # Single signed net-rate knob g (1/day): g>0 -> proliferation (rho=g),
    # g<0 -> net cell death (delta=-g). Monotonic in g, so one bracketed
    # root-find spans both growth and regression regimes robustly.
    def sim_vol(g: float) -> float:
        p = {**base_params, "rho": max(g, 0.0), "delta": max(-g, 0.0)}
        return _volume_at(solve_growth(baseline, steps, DT_DAYS, p)[-1], DETECT_ISO, voxel_mm3)

    if followup is not None:
        target = _volume_at(followup, DETECT_ISO, voxel_mm3)
        g = brentq(lambda x: sim_vol(x) - target, -1.0, 1.0, xtol=1e-4)
    else:
        g = 0.05
    rho, delta = max(g, 0.0), max(-g, 0.0)
    net_rate = float(g)  # signed net proliferation (1/day): + growth, - death
    # Regime reflects the real VOLUME change, not the sign of the rate (diffusion
    # can shrink detectable volume even at a small positive proliferation).
    if followup is not None:
        regime = "growth" if _volume_at(followup, DETECT_ISO, voxel_mm3) >= v0 else "regression"
    else:
        regime = "growth"
    base_params.update(rho=rho, delta=delta)

    frames = solve_growth(baseline, steps, DT_DAYS, base_params)
    b0 = _burden(baseline, voxel_mm3)

    # --- report (honest, day-scaled, volume calibrated to expert mask) --------
    sim_v = _volume_at(frames[-1], DETECT_ISO, voxel_mm3)
    real_v = _volume_at(followup, DETECT_ISO, voxel_mm3) if followup is not None else None
    # Observed VOLUME doubling time from the real data (not from rho, which is a
    # local proliferation rate -- Fisher-KPP volume grows by front propagation).
    obs_dbl = None
    if real_v and real_v > v0:
        obs_dbl = interval * np.log(2) / np.log(real_v / v0)

    print(f"Patient {args.patient}  (IDH={info.get('idh_status','?')}, grade={info.get('grade','?')})")
    print(f"  regime={regime}  D={args.D} mm^2/day  dt={DT_DAYS} day  steps={steps} days (real time)")
    print(f"  calibrated net rate={net_rate:+.4f} /day  (+ proliferation / - death; local rate)")
    print(f"  TUMOR VOLUME @iso={DETECT_ISO} (mm^3, calibrated metric): "
          f"baseline {v0:.0f} -> sim {sim_v:.0f}"
          + (f"  [real follow-up {real_v:.0f}]" if real_v is not None else ""))
    if obs_dbl:
        print(f"  observed volume doubling ~{obs_dbl:.0f} days")
    if info:
        print(f"  (iso validated vs expert mask: baseline {info['baseline']['wt_mm3']:.0f}, "
              f"follow-up {info['followup']['wt_mm3']:.0f} mm^3)")
    print(f"  integrated burden (secondary, threshold-free): {b0:.0f} -> {_burden(frames[-1], voxel_mm3):.0f}")

    # --- render at the detection iso so growth is visible from day 0 ----------
    ds = [r.downsample(f, 2) for f in frames]
    fig = r.render_sequence(ds, iso=DETECT_ISO)
    match = f", sim {sim_v:.0f} vs real {real_v:.0f} mm^3" if real_v is not None else ""
    fig.update_layout(title=(
        f"Glioma {args.patient} ({regime}) | net rate {net_rate:+.3f}/day, D={args.D} mm^2/day, "
        f"{int(interval)} real days | volume@iso{DETECT_ISO}{match} "
        f"(seed = expert-mask tumor density, modeling proxy)"
    ))

    out = args.out or (args.data_dir.parent / f"brain_demo_{args.patient}.html")
    fig.write_html(str(out))
    print(f"\nwrote {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
