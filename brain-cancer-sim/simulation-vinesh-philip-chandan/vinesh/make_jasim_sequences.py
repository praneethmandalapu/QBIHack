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
from tumor_pde_solver import solve_growth  # noqa: E402

TIMESTEPS, DT, N_KEEP = 50, 0.1, 26


def _find(data_dir: Path, pid: str, tp: str) -> Path | None:
    hits = [f for f in glob.glob(str(data_dir / f"*{pid}_{tp}.npy"))
            if not os.path.basename(f).startswith("._")]
    return Path(hits[0]) if hits else None


def _burden(v: np.ndarray) -> float:
    return float(v.sum())


def calibrate(baseline: np.ndarray, followup: np.ndarray) -> tuple[str, dict]:
    """Return (regime, params) so simulated burden matches the followup."""
    b0, target = _burden(baseline), _burden(followup)
    sim = lambda p: _burden(solve_growth(baseline, TIMESTEPS, DT, p)[-1])  # noqa: E731
    if target >= b0:
        rm = brentq(lambda m: sim({"risk_multiplier": m, "delta": 0.0}) - target,
                    0.0, 60.0, xtol=1e-3)
        return "growth", {"risk_multiplier": rm, "delta": 0.0}
    d = brentq(lambda x: sim({"risk_multiplier": 1.0, "delta": x}) - target,
               0.0, 8.0, xtol=1e-3)
    return "regression", {"risk_multiplier": 1.0, "delta": d}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("D:/pde-input-vinesh"))
    ap.add_argument("--out", type=Path, default=Path("D:/pde-input-vinesh/frames-for-jasim"))
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
        regime, params = calibrate(baseline, followup)
        idh = str(p.get("idh_status"))
        grade = str(p.get("grade"))
        slug = f"glioma_{pid}_idh{idh}_grade{grade}".replace(".0", "")
        npy, js = export_frames(
            baseline, slug, args.out,
            risk_multiplier=params["risk_multiplier"],
            params={"delta": params["delta"]},
            timesteps=TIMESTEPS, dt=DT, n_keep=N_KEEP,
            interval_days=p.get("interval_days"),
            meta_extra={
                "patient_id": pid, "disease": "glioma",
                "idh_status": idh, "grade": grade, "regime": regime,
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
