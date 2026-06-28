"""Tests for brain PDE calibration (expert-mask PDE cubes, not breast Otsu)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parents[1]
SIM_ROOT = PHILIP_CHANDAN_DIR.parent
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SIM_ROOT))
sys.path.insert(0, str(SIM_ROOT / "vinesh"))

from calibrate_philip import (  # noqa: E402
    calibration_target_burden,
    calibrate_growth,
    calibrate_patient,
    load_observed_change,
    prepare_initial_condition,
    solver_schedule,
    tumor_burden,
)
from spike_paths import resolve_pde_input_npy  # noqa: E402


def test_prepare_initial_condition_zeros_background():
    vol = np.array([0.0, 0.1, 0.5, 0.9], dtype=np.float32)
    seed = prepare_initial_condition(vol, background=0.0)
    assert seed.tolist() == pytest.approx([0.0, 0.1, 0.5, 0.9])


def test_prepare_initial_condition_preserves_tumor_density():
    cube = np.zeros((8, 8, 8), dtype=np.float32)
    cube[2:6, 2:6, 2:6] = 0.42
    seed = prepare_initial_condition(cube)
    assert seed[3, 3, 3] == pytest.approx(0.42)
    assert seed[0, 0, 0] == 0.0


def test_load_observed_change_spike_patient():
    try:
        observed = load_observed_change("100002")
    except FileNotFoundError:
        pytest.skip("wt_volume_report.json not on disk")
    assert observed["wt_baseline_mm3"] == pytest.approx(8137.0)
    assert observed["wt_followup_mm3"] == pytest.approx(8365.0)
    assert observed["wt_growth_pct"] == pytest.approx(2.8, rel=0.01)
    assert observed["interval_days"] == pytest.approx(183, rel=0.01)


def test_calibration_target_wt_scaled():
    observed = {"wt_baseline_mm3": 100.0, "wt_followup_mm3": 110.0}
    target, mode = calibration_target_burden(
        1000.0,
        pde_followup_burden=900.0,
        observed=observed,
        mode="wt_scaled",
    )
    assert mode == "wt_scaled"
    assert target == pytest.approx(1100.0)


def test_solver_schedule_spans_interval_in_days():
    schedule = solver_schedule(interval_days=183.0)
    assert schedule["dt_unit"] == "days"
    assert schedule["dt"] == pytest.approx(1.0)
    assert schedule["timesteps"] == 183
    assert schedule["simulated_days"] == pytest.approx(183.0)
    assert schedule["cfl_max_dt"] == pytest.approx(1.1111, rel=0.01)


def test_calibrate_growth_pde_followup_mode():
    baseline_path = resolve_pde_input_npy("glioma_ucsf_100002_baseline")
    followup_path = resolve_pde_input_npy("glioma_ucsf_100002_followup")
    if not baseline_path.is_file() or not followup_path.is_file():
        pytest.skip("spike PDE cubes not on disk")

    baseline = np.load(baseline_path)
    followup = np.load(followup_path)
    result = calibrate_growth(
        baseline,
        followup,
        timesteps=50,
        dt=0.1,
        target_mode="pde_followup",
    )

    assert result["target_mode"] == "pde_followup"
    assert result["baseline_burden"] > 0
    assert abs(result["burden_error_pct"]) < 5.0


def test_calibrate_growth_wt_scaled_spike_patient():
    baseline_path = resolve_pde_input_npy("glioma_ucsf_100002_baseline")
    followup_path = resolve_pde_input_npy("glioma_ucsf_100002_followup")
    if not baseline_path.is_file() or not followup_path.is_file():
        pytest.skip("spike PDE cubes not on disk")
    try:
        observed = load_observed_change("100002")
    except FileNotFoundError:
        pytest.skip("wt_volume_report.json not on disk")

    baseline = np.load(baseline_path)
    followup = np.load(followup_path)
    result = calibrate_growth(
        baseline,
        followup,
        timesteps=50,
        dt=0.1,
        observed=observed,
        target_mode="wt_scaled",
    )

    assert result["target_mode"] == "wt_scaled"
    assert result["regime"] == "growth"
    assert result["knob_name"] == "risk_multiplier"
    assert result["target_burden"] > result["baseline_burden"]
    assert result["pde_burden_followup"] < result["pde_burden_baseline"]
    assert result["interval_days"] == pytest.approx(183, rel=0.01)
    assert result["dt_unit"] == "days"
    assert result["dt"] == pytest.approx(1.0)
    assert result["timesteps"] == pytest.approx(183, rel=0.02)
    assert result["simulated_days"] == pytest.approx(183, rel=0.02)
    assert abs(result["burden_error_pct"]) < 5.0


def test_calibrate_patient_writes_json(tmp_path, monkeypatch):
    baseline_path = resolve_pde_input_npy("glioma_ucsf_100002_baseline")
    if not baseline_path.is_file():
        pytest.skip("spike PDE cubes not on disk")
    try:
        load_observed_change("100002")
    except FileNotFoundError:
        pytest.skip("wt_volume_report.json not on disk")

    import calibrate_philip as mod

    out_dir = tmp_path / "calibration"
    monkeypatch.setattr(mod, "CALIBRATION_OUT_DIR", out_dir)
    result = calibrate_patient("100002", write_json=True)
    assert (out_dir / "100002.json").is_file()
    assert result["patient_id"] == "100002"
    assert result["regime"] == "growth"
