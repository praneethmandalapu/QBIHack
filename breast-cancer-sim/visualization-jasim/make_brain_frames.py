"""Generate the BRAIN demo frame stacks the OncoPulse site renders (Person 3 / jasim).

Builds one forward-simulation stack per UCSF patient in each IDH regime:

  • aggressive (IDH-WT) — Vinesh extent calibration
  • indolent (IDH-mut)  — low-D extent calibration

Outputs (gitignored under breast-cancer-sim/data/processed/brain-frames-jasim/):

  glioma_<patient_id>_<regime>_frames.npy
  glioma_<patient_id>_<regime>_frames.json

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

DEMO_VERSION = "rev2-cohort-picker-v1"
D_STABLE = 0.02   # minimal invasion — Vinesh D_FLOOR

DEFAULTS = {"aggressive": "100118", "indolent": "100002"}


def regime_config() -> dict[str, dict]:
    """Regime -> {default, patients} from pde_burden_compare (14 no-RC cohort)."""
    rows = json.loads(BURDEN_JSON.read_text())["patients"]
    wt = sorted(
        [r for r in rows if (r.get("idh_status") or "").upper() == "WT"],
        key=lambda r: -abs(float(r.get("wt_growth_pct") or 0)),
    )
    mut = sorted(
        [r for r in rows if (r.get("idh_status") or "").upper() != "WT"],
        key=lambda r: abs(float(r.get("wt_growth_pct") or 0)),
    )
    return {
        "aggressive": {
            "default": DEFAULTS["aggressive"],
            "patients": [str(r["patient_id"]) for r in wt],
        },
        "indolent": {
            "default": DEFAULTS["indolent"],
            "patients": [str(r["patient_id"]) for r in mut],
        },
    }


# Backward-compatible demo pair for scripts that import SCENARIOS.
SCENARIOS = {k: {"patient_id": v} for k, v in DEFAULTS.items()}


def _brentq_or_best(err_fn, lo: float, hi: float) -> float:
    try:
        return float(brentq(err_fn, lo, hi, xtol=1e-3))
    except ValueError:
        grid = np.linspace(lo, hi, 40)
        return float(grid[int(np.argmin([abs(err_fn(x)) for x in grid]))])


def _extent(v: np.ndarray, thr: float = BURDEN_THR) -> float:
    return float((v > thr).sum())


def _calibrate_stable_extent(
    baseline: np.ndarray, followup: np.ndarray, timesteps: int,
) -> tuple[str, dict]:
    target = _extent(followup)
    if target < _extent(baseline):

        def err_delta(delta: float) -> float:
            fr = solve_growth(baseline, timesteps, DT,
                              {"D": D_STABLE, "risk_multiplier": 1.0, "delta": delta})
            return _burden_sum(fr[-1]) - _burden_sum(followup)

        delta = _brentq_or_best(err_delta, 0.0, 8.0)
        return "regression", {"D": D_STABLE, "risk_multiplier": 1.0, "delta": float(delta)}

    def err(risk_multiplier: float) -> float:
        fr = solve_growth(baseline, timesteps, DT,
                          {"D": D_STABLE, "risk_multiplier": risk_multiplier, "delta": 0.0})
        return _extent(fr[-1]) - target

    rm = _brentq_or_best(err, 0.001, 2.0)
    return "growth", {"D": D_STABLE, "risk_multiplier": float(rm), "delta": 0.0}


def _burden_sum(v: np.ndarray) -> float:
    return float(v.sum())


def _burden_row(patient_id: str) -> dict:
    rows = json.loads(BURDEN_JSON.read_text())["patients"]
    for row in rows:
        if str(row["patient_id"]) == str(patient_id):
            return row
    raise KeyError(f"patient {patient_id} not in {BURDEN_JSON}")


def frame_path(out_dir: Path, regime: str, patient_id: str) -> Path:
    return out_dir / f"glioma_{patient_id}_{regime}_frames.npy"


def meta_path(out_dir: Path, regime: str, patient_id: str) -> Path:
    return out_dir / f"glioma_{patient_id}_{regime}_frames.json"


def _simulate(regime: str, patient_id: str) -> tuple[np.ndarray, dict]:
    row = _burden_row(patient_id)
    bp, fp = _find(PDE_INPUT, patient_id, "baseline"), _find(PDE_INPUT, patient_id, "followup")
    if not bp or not fp:
        raise FileNotFoundError(f"Missing PDE baseline/followup for patient {patient_id}")
    baseline = np.load(bp)
    followup = np.load(fp)
    interval_days = float(row["interval_days"])
    timesteps = _timesteps_for(interval_days)
    dt = DT

    if regime == "indolent":
        try:
            pde_regime, params = _calibrate_stable_extent(baseline, followup, timesteps)
            cal_source = "extent-at-threshold"
        except ValueError:
            pde_regime, params = calibrate(baseline, followup, timesteps)
            cal_source = "vinesh-extent-fallback"
    else:
        pde_regime, params = calibrate(baseline, followup, timesteps)
        cal_source = "vinesh-extent"

    frames = solve_growth(baseline, timesteps, dt, params)
    idx = np.linspace(0, len(frames) - 1, N_KEEP).round().astype(int)
    stack = np.stack([frames[i] for i in idx]).astype(np.float32)
    meta = {
        "patient_id": patient_id,
        "regime_key": regime,
        "idh_status": row["idh_status"],
        "grade": row["grade"],
        "pde_regime": pde_regime,
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
    written: dict[str, Path] = {}
    cfg = regime_config()
    for regime, block in cfg.items():
        for patient_id in block["patients"]:
            stack, meta = _simulate(regime, patient_id)
            npy = frame_path(out_dir, regime, patient_id)
            js = meta_path(out_dir, regime, patient_id)
            np.save(npy, stack)
            js.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            vol = np.array([(stack[i] > BURDEN_THR).sum() for i in range(len(stack))], float)
            written[f"{regime}/{patient_id}"] = npy
            print(f"{regime:10s} patient {patient_id}  {stack.shape}  "
                  f"real WT {meta['real_growth_pct']:+.0f}%  burden idx 100 -> "
                  f"{100 * vol[-1] / max(vol[0], 1):.0f}  -> {npy.name}")
    (out_dir / ".demo_version").write_text(DEMO_VERSION + "\n", encoding="utf-8")
    return written


def ensure_frames(out_dir: Path = OUT_DIR) -> None:
    """Regenerate stacks when missing or when the cohort pairing version changes."""
    marker = out_dir / ".demo_version"
    cfg = regime_config()
    needed = [
        frame_path(out_dir, regime, pid)
        for regime, block in cfg.items()
        for pid in block["patients"]
    ]
    stale = not marker.is_file() or marker.read_text(encoding="utf-8").strip() != DEMO_VERSION
    if stale or not all(p.exists() for p in needed):
        print("brain frames missing or stale -> regenerating cohort picker stacks")
        generate(out_dir)


if __name__ == "__main__":
    generate()
