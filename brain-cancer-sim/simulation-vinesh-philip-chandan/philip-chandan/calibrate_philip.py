"""Calibrate the PDE solver against a real second timepoint (brain port).

Philip-Chandan prototype of breast `vinesh/calibrate.py`. Brain v1 uses
expert-mask PDE cubes from `prepare_pde_input.py` — not breast Otsu isolation.

Calibration fits one solver knob so forward simulation from the baseline PDE
seed matches an observed follow-up burden. Default target scales baseline PDE
burden by expert whole-tumor (WT) volume change from `wt_volume_report.json`,
because independent per-timepoint crops make raw follow-up PDE burden unreliable.

Honesty note: with only two timepoints this is a *calibration* (a fit to the
observed change), not an out-of-sample prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SIM_ROOT = PHILIP_CHANDAN_DIR.parent
REPO_ROOT = SIM_ROOT.parent
VINESH_DIR = SIM_ROOT / "vinesh"

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SIM_ROOT))
sys.path.insert(0, str(VINESH_DIR))

from handoff_contract import pde_input_spec, solver_spec  # noqa: E402
from spike_paths import resolve_pde_input_npy  # noqa: E402
from tumor_pde_solver import cfl_max_dt, solve_growth  # noqa: E402

WT_VOLUME_REPORT_PATH = REPO_ROOT / "data/processed/raw-extract-philip-chandan/wt_volume_report.json"
MANIFEST_PATH = REPO_ROOT / "data/processed/raw-extract-philip-chandan/manifest.json"
CALIBRATION_OUT_DIR = REPO_ROOT / "data/processed/calibration-philip-chandan"

TargetMode = Literal["wt_scaled", "pde_followup"]


def prepare_initial_condition(
    pde_volume: np.ndarray,
    *,
    background: float | None = None,
) -> np.ndarray:
    """Use a Philip-Chandan PDE cube as the solver initial condition."""
    spec = pde_input_spec()
    bg = float(background if background is not None else spec["background_value"])
    vol = np.clip(np.asarray(pde_volume, dtype=np.float32), 0.0, 1.0)
    return np.where(vol > bg, vol, 0.0).astype(np.float32)


def tumor_burden(volume: np.ndarray) -> float:
    """Total tumor burden = integral of density (sum of voxel values)."""
    return float(np.asarray(volume, dtype=np.float64).sum())


def tumor_volume(volume: np.ndarray, threshold: float, spacing=(1.0, 1.0, 1.0)) -> float:
    """Physical tumor volume (mm^3): voxels above `threshold` times voxel size."""
    voxel = float(np.prod(spacing))
    return float(np.count_nonzero(np.asarray(volume) >= threshold)) * voxel


def load_wt_volume_report(path: Path | None = None) -> dict[str, Any]:
    report_path = path or WT_VOLUME_REPORT_PATH
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing WT volume report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or MANIFEST_PATH
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _manifest_patient(manifest: dict[str, Any], patient_id: str) -> dict[str, Any] | None:
    for row in manifest.get("patients", []):
        if str(row.get("patient_id")) == str(patient_id):
            return row
    return None


def load_observed_change(
    patient_id: str,
    *,
    wt_report_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Expert-mask WT volumes and interval between baseline and follow-up."""
    report = load_wt_volume_report(wt_report_path)
    row = next(
        (p for p in report.get("patients", []) if str(p.get("patient_id")) == str(patient_id)),
        None,
    )
    if row is None:
        raise KeyError(f"Patient {patient_id!r} not in WT volume report")

    baseline = row.get("baseline") or {}
    followup = row.get("followup") or {}
    wt_baseline = float(baseline.get("computed_mm3", 0.0))
    wt_followup = float(followup.get("computed_mm3", 0.0))

    interval_days = row.get("interval_days")
    manifest_row = None
    if manifest_path is not None or MANIFEST_PATH.is_file():
        manifest_row = _manifest_patient(load_manifest(manifest_path), patient_id)
        if interval_days is None and manifest_row is not None:
            interval_days = manifest_row.get("interval_days")

    return {
        "patient_id": str(patient_id),
        "wt_baseline_mm3": wt_baseline,
        "wt_followup_mm3": wt_followup,
        "wt_delta_mm3": float(row.get("computed_delta_mm3", wt_followup - wt_baseline)),
        "wt_growth_pct": float(row.get("computed_growth_pct", 0.0)),
        "interval_days": float(interval_days) if interval_days is not None else None,
        "baseline_slug": (manifest_row or {}).get("baseline_slug"),
        "followup_slug": (manifest_row or {}).get("followup_slug"),
    }


def calibration_target_burden(
    baseline_pde_burden: float,
    *,
    pde_followup_burden: float,
    observed: dict[str, Any] | None = None,
    mode: TargetMode = "wt_scaled",
) -> tuple[float, TargetMode]:
    """Choose the scalar burden target for the root-finder."""
    if mode == "pde_followup":
        return float(pde_followup_burden), mode

    if observed is None:
        return float(pde_followup_burden), "pde_followup"

    wt_baseline = float(observed.get("wt_baseline_mm3") or 0.0)
    wt_followup = float(observed.get("wt_followup_mm3") or 0.0)
    if wt_baseline <= 0:
        return float(pde_followup_burden), "pde_followup"

    scaled = baseline_pde_burden * (wt_followup / wt_baseline)
    return float(scaled), "wt_scaled"


def solver_schedule(
    *,
    interval_days: float | None = None,
    timesteps: int | None = None,
    dt: float | None = None,
    contract_path: str | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float | int | str | None]:
    """Build CFL-safe (timesteps, dt). When `interval_days` is set, 1 step ~= 1 day.

    Diffusion on the g64 / 1 mm grid caps dt at ~1.11 days (D=0.15). We use dt=1 day
  and timesteps ~= interval_days so the forward sim spans the real scan interval.
    The breast contract (50 x dt=0.1) is used only when no interval is provided.
    """
    spec = solver_spec(contract_path)
    default_params = spec.get("default_params", {})
    D = float(default_params.get("D", 0.15))

    if interval_days is not None and interval_days > 0:
        dt_max = cfl_max_dt(D, spacing)
        step_dt = min(1.0, dt_max * 0.99)
        steps = max(1, int(round(float(interval_days) / step_dt)))
        simulated_days = steps * step_dt
        return {
            "timesteps": steps,
            "dt": step_dt,
            "dt_unit": "days",
            "interval_days": float(interval_days),
            "simulated_days": simulated_days,
            "cfl_max_dt": dt_max,
        }

    steps = int(timesteps if timesteps is not None else spec["timesteps"])
    step_dt = float(dt if dt is not None else spec["dt"])
    return {
        "timesteps": steps,
        "dt": step_dt,
        "dt_unit": "abstract",
        "interval_days": None,
        "simulated_days": None,
        "cfl_max_dt": cfl_max_dt(D, spacing),
    }


def _calibrate_seed(
    baseline_seed: np.ndarray,
    target_burden: float,
    timesteps: int,
    dt: float,
    base_params: dict[str, Any],
    *,
    max_multiplier: float = 25.0,
    max_delta: float = 8.0,
) -> dict[str, Any]:
    b0 = tumor_burden(baseline_seed)
    if b0 <= 0:
        raise ValueError("Baseline PDE seed is empty; cannot calibrate.")

    def sim_burden(params: dict) -> float:
        frames = solve_growth(baseline_seed, timesteps, dt, params)
        return tumor_burden(frames[-1])

    if target_burden >= b0:
        regime = "growth"

        def err(m: float) -> float:
            p = {**base_params, "risk_multiplier": m, "delta": 0.0}
            return sim_burden(p) - target_burden

        knob = brentq(err, 0.0, max_multiplier, xtol=1e-3)
        params = {**base_params, "risk_multiplier": knob, "delta": 0.0}
        knob_name = "risk_multiplier"
    else:
        regime = "regression"
        rm = float(base_params.get("risk_multiplier", 1.0))

        def err(d: float) -> float:
            p = {**base_params, "risk_multiplier": rm, "delta": d}
            return sim_burden(p) - target_burden

        knob = brentq(err, 0.0, max_delta, xtol=1e-3)
        params = {**base_params, "risk_multiplier": rm, "delta": knob}
        knob_name = "delta"

    achieved = sim_burden(params)
    return {
        "params": params,
        "regime": regime,
        "knob_name": knob_name,
        "knob_value": float(knob),
        "baseline_burden": b0,
        "target_burden": float(target_burden),
        "achieved_burden": achieved,
        "burden_error_pct": 100.0 * (achieved - target_burden) / target_burden if target_burden else 0.0,
    }


def calibrate_growth(
    baseline: np.ndarray,
    followup: np.ndarray,
    timesteps: int,
    dt: float,
    base_params: dict | None = None,
    *,
    observed: dict[str, Any] | None = None,
    target_mode: TargetMode | None = None,
    interval_days: float | None = None,
    background: float | None = None,
    max_multiplier: float = 25.0,
    max_delta: float = 8.0,
) -> dict[str, Any]:
    """Tune one growth knob so simulated final burden matches the follow-up target."""
    spec = solver_spec()
    base = {**spec.get("default_params", {}), **(base_params or {}), "spacing": (1.0, 1.0, 1.0)}

    effective_interval = interval_days
    if effective_interval is None and observed:
        effective_interval = observed.get("interval_days")

    schedule = solver_schedule(
        interval_days=float(effective_interval) if effective_interval is not None else None,
        timesteps=timesteps if effective_interval is None else None,
        dt=dt if effective_interval is None else None,
        spacing=tuple(base["spacing"]),
    )
    steps = int(schedule["timesteps"])
    step_dt = float(schedule["dt"])
    if schedule.get("dt_unit") == "days":
        contract_dt = float(spec["dt"])
        base["rho"] = float(base["rho"]) * (contract_dt / step_dt)

    baseline_seed = prepare_initial_condition(baseline, background=background)
    followup_ref = prepare_initial_condition(followup, background=background)
    pde_baseline = tumor_burden(baseline_seed)
    pde_followup = tumor_burden(followup_ref)

    mode: TargetMode = target_mode or ("wt_scaled" if observed else "pde_followup")
    target, resolved_mode = calibration_target_burden(
        pde_baseline,
        pde_followup_burden=pde_followup,
        observed=observed,
        mode=mode,
    )

    fit = _calibrate_seed(
        baseline_seed,
        target,
        steps,
        step_dt,
        base,
        max_multiplier=max_multiplier,
        max_delta=max_delta,
    )

    observed_block: dict[str, Any] = {}
    if observed:
        observed_block = {
            "wt_baseline_mm3": observed.get("wt_baseline_mm3"),
            "wt_followup_mm3": observed.get("wt_followup_mm3"),
            "wt_delta_mm3": observed.get("wt_delta_mm3"),
            "wt_growth_pct": observed.get("wt_growth_pct"),
        }

    return {
        **fit,
        "target_mode": resolved_mode,
        "pde_burden_baseline": pde_baseline,
        "pde_burden_followup": pde_followup,
        "timesteps": steps,
        "dt": step_dt,
        "dt_unit": schedule.get("dt_unit"),
        "interval_days": schedule["interval_days"],
        "simulated_days": schedule.get("simulated_days"),
        "cfl_max_dt": schedule.get("cfl_max_dt"),
        "observed": observed_block or None,
        "baseline_seed": baseline_seed,
        "followup_ref": followup_ref,
        "baseline_iso": baseline_seed,
        "followup_iso": followup_ref,
    }


def load_patient_pde_cubes(
    patient_id: str,
    *,
    manifest_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    row = _manifest_patient(manifest, patient_id)
    if row is None:
        raise KeyError(f"Patient {patient_id!r} not in manifest")

    baseline_slug = row.get("baseline_slug")
    followup_slug = row.get("followup_slug")
    if not baseline_slug or not followup_slug:
        raise ValueError(f"Patient {patient_id!r} missing baseline/followup slugs in manifest")

    baseline_path = resolve_pde_input_npy(str(baseline_slug))
    followup_path = resolve_pde_input_npy(str(followup_slug))
    if not baseline_path.is_file() or not followup_path.is_file():
        raise FileNotFoundError(
            f"Missing PDE cubes for {patient_id}: {baseline_path} / {followup_path}"
        )

    return np.load(baseline_path), np.load(followup_path), row


def calibrate_patient(
    patient_id: str,
    *,
    target_mode: TargetMode | None = None,
    max_multiplier: float = 25.0,
    write_json: bool = True,
) -> dict[str, Any]:
    """Load on-disk PDE cubes + WT report and calibrate one patient."""
    baseline, followup, manifest_row = load_patient_pde_cubes(patient_id)
    observed = load_observed_change(patient_id)
    interval_days = observed.get("interval_days") or manifest_row.get("interval_days")

    result = calibrate_growth(
        baseline,
        followup,
        timesteps=solver_spec()["timesteps"],
        dt=solver_spec()["dt"],
        observed=observed,
        target_mode=target_mode,
        interval_days=float(interval_days) if interval_days is not None else None,
        max_multiplier=max_multiplier,
    )
    payload = {
        "patient_id": str(patient_id),
        "baseline_slug": manifest_row.get("baseline_slug"),
        "followup_slug": manifest_row.get("followup_slug"),
        **{k: v for k, v in result.items() if k not in ("baseline_seed", "followup_ref", "baseline_iso", "followup_iso")},
        "params": result["params"],
    }

    if write_json:
        CALIBRATION_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = CALIBRATION_OUT_DIR / f"{patient_id}.json"

        def _json_default(obj: object) -> object:
            if isinstance(obj, np.ndarray):
                return None
            raise TypeError(type(obj))

        out_path.write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        try:
            payload["output_path"] = str(out_path.relative_to(REPO_ROOT))
        except ValueError:
            payload["output_path"] = str(out_path)

    return payload


def predict_trajectory(
    baseline_seed: np.ndarray,
    params: dict,
    timesteps: int,
    dt: float,
) -> list[np.ndarray]:
    """Run the calibrated forward simulation from the baseline PDE seed."""
    return solve_growth(baseline_seed, timesteps, dt, params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate PDE growth for a UCSF patient pair.")
    parser.add_argument("--patient-id", required=True, help="e.g. 100002")
    parser.add_argument(
        "--target-mode",
        choices=("wt_scaled", "pde_followup"),
        default=None,
        help="wt_scaled uses expert WT mm3 ratio (default when WT report exists)",
    )
    parser.add_argument("--max-multiplier", type=float, default=25.0)
    parser.add_argument("--no-write", action="store_true", help="Skip calibration JSON output")
    args = parser.parse_args()

    result = calibrate_patient(
        args.patient_id,
        target_mode=args.target_mode,
        max_multiplier=args.max_multiplier,
        write_json=not args.no_write,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "params"}, indent=2))
    print("params:", json.dumps(result["params"], indent=2))
    if "output_path" in result:
        print(f"Wrote {result['output_path']}")


if __name__ == "__main__":
    main()
