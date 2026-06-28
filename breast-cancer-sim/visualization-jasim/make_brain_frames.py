"""Generate the BRAIN demo frame stacks the OncoPulse site renders (Person 3 / jasim).

Uses Philip-Chandan's locked demo pair from brain cohort rev2 / manifest v1.0.0:

  • aggressive — patient **100118** (IDH-WT GBM, measured +609% WT growth)
  • indolent   — patient **100002** (IDH-mut oligodendroglioma, measured +3%)

Each case loads Vinesh's PDE baseline + follow-up cubes, calibrates invasion/death
(`make_jasim_sequences.calibrate`), and forward-simulates with `solve_growth`.
This matches Vinesh's HANDOFF_JASIM.md contract rather than reusing one geometry
with two parameter sets.

Outputs (gitignored under breast-cancer-sim/data/processed/brain-frames-jasim/):

  glioma_<patient_id>_<regime>_frames.npy
  glioma_<patient_id>_<regime>_frames.json   # interval_days, real_growth_pct, …

`build_site.py` calls `ensure_frames()` to regenerate on demand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BREAST = HERE.parent                      # breast-cancer-sim/
REPO = BREAST.parent                      # qbihack/
BRAIN_REPO = REPO / "brain-cancer-sim"
PDE_INPUT = BRAIN_REPO / "data/processed/pde-input-vinesh"
BURDEN_JSON = PDE_INPUT / "pde_burden_compare.json"
OUT_DIR = BREAST / "data/processed/brain-frames-jasim"
BURDEN_THR = 0.2   # matches isosurface floor + Vinesh extent calibration

VINESH = BRAIN_REPO / "simulation-vinesh-philip-chandan/vinesh"
sys.path.insert(0, str(VINESH))

from scipy.optimize import brentq  # noqa: E402

from make_jasim_sequences import N_KEEP, DT, _find, _timesteps_for, calibrate  # noqa: E402
from tumor_pde_solver import solve_growth  # noqa: E402

# Philip-Chandan cohort.json primary pair + manifest v1.0.0 demo contrast.
SCENARIOS = {
    "aggressive": {"patient_id": "100118"},
    "indolent": {"patient_id": "100002"},
}
DEMO_VERSION = "rev2-100118-100002-extent-cal"
D_STABLE = 0.02   # minimal invasion — Vinesh D_FLOOR


def _extent(v: np.ndarray, thr: float = BURDEN_THR) -> float:
    return float((v > thr).sum())


def _calibrate_stable_extent(
    baseline: np.ndarray, followup: np.ndarray, timesteps: int,
) -> tuple[str, dict]:
    """Fit risk_multiplier at low D so lesion extent (voxels > BURDEN_THR) matches follow-up.

    Philip's sum-based calibration can raise total mass while diffusing peak density
    below the viewer threshold — the shell vanishes even for +3% real WT growth.
    """
    target = _extent(followup)
    if target < _extent(baseline):

        def err_delta(delta: float) -> float:
            fr = solve_growth(baseline, timesteps, DT,
                              {"D": D_STABLE, "risk_multiplier": 1.0, "delta": delta})
            return _burden_sum(fr[-1]) - _burden_sum(followup)

        delta = brentq(err_delta, 0.0, 8.0, xtol=1e-3)
        return "regression", {"D": D_STABLE, "risk_multiplier": 1.0, "delta": float(delta)}

    def err(risk_multiplier: float) -> float:
        fr = solve_growth(baseline, timesteps, DT,
                          {"D": D_STABLE, "risk_multiplier": risk_multiplier, "delta": 0.0})
        return _extent(fr[-1]) - target

    rm = brentq(err, 0.001, 2.0, xtol=1e-3)
    return "growth", {"D": D_STABLE, "risk_multiplier": float(rm), "delta": 0.0}


def _burden_sum(v: np.ndarray) -> float:
    return float(v.sum())


def _burden_row(patient_id: str) -> dict:
    rows = json.loads(BURDEN_JSON.read_text())["patients"]
    for row in rows:
        if row["patient_id"] == patient_id:
            return row
    raise KeyError(f"patient {patient_id} not in {BURDEN_JSON}")


def frame_path(out_dir: Path, key: str) -> Path:
    pid = SCENARIOS[key]["patient_id"]
    return out_dir / f"glioma_{pid}_{key}_frames.npy"


def meta_path(out_dir: Path, key: str) -> Path:
    pid = SCENARIOS[key]["patient_id"]
    return out_dir / f"glioma_{pid}_{key}_frames.json"


def _simulate(key: str) -> tuple[np.ndarray, dict]:
    pid = SCENARIOS[key]["patient_id"]
    row = _burden_row(pid)
    bp, fp = _find(PDE_INPUT, pid, "baseline"), _find(PDE_INPUT, pid, "followup")
    if not bp or not fp:
        raise FileNotFoundError(f"Missing PDE baseline/followup for patient {pid}")
    baseline = np.load(bp)
    followup = np.load(fp)
    interval_days = float(row["interval_days"])
    timesteps = _timesteps_for(interval_days)
    dt = DT

    if key == "indolent":
        regime, params = _calibrate_stable_extent(baseline, followup, timesteps)
        cal_source = "extent-at-threshold"
    else:
        regime, params = calibrate(baseline, followup, timesteps)
        cal_source = "vinesh-extent"

    frames = solve_growth(baseline, timesteps, dt, params)
    idx = np.linspace(0, len(frames) - 1, N_KEEP).round().astype(int)
    stack = np.stack([frames[i] for i in idx]).astype(np.float32)
    meta = {
        "patient_id": pid,
        "regime_key": key,
        "idh_status": row["idh_status"],
        "grade": row["grade"],
        "pde_regime": regime,
        "calibration_source": cal_source,
        "interval_days": round(interval_days, 1),
        "real_growth_pct": round(float(row["wt_growth_pct"]), 1),
        "sim_params": params,
        "sim_dt": dt,
        "sim_timesteps": timesteps,
        "n_frames": int(stack.shape[0]),
    }
    return stack, meta


def generate(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for key in SCENARIOS:
        stack, meta = _simulate(key)
        npy = frame_path(out_dir, key)
        js = meta_path(out_dir, key)
        np.save(npy, stack)
        js.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        vol = np.array([(stack[i] > BURDEN_THR).sum() for i in range(len(stack))], float)
        written[key] = npy
        print(f"{key:10s} patient {meta['patient_id']}  {stack.shape}  "
              f"real WT {meta['real_growth_pct']:+.0f}%  burden idx 100 -> "
              f"{100 * vol[-1] / max(vol[0], 1):.0f}  -> {npy.name}")
    (out_dir / ".demo_version").write_text(DEMO_VERSION + "\n", encoding="utf-8")
    return written


def ensure_frames(out_dir: Path = OUT_DIR) -> None:
    """Regenerate demo stacks when missing or when the demo patient pairing changes."""
    marker = out_dir / ".demo_version"
    needed = [frame_path(out_dir, k) for k in SCENARIOS]
    stale = not marker.is_file() or marker.read_text(encoding="utf-8").strip() != DEMO_VERSION
    if stale or not all(p.exists() for p in needed):
        print("brain frames missing or stale -> regenerating from Philip/Vinesh demo pair")
        generate(out_dir)


if __name__ == "__main__":
    generate()
