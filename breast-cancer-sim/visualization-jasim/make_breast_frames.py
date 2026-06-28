"""Generate breast PDE growth frame stacks for the OncoPulse viewer (Person 3 / jasim).

Runs Vinesh's Fisher-KPP engine on Philip-Chandan's PDE baseline cubes with
subtype-default risk multipliers (same as ``app/tabs/simulate_tab.py``). This
matches the prototype breast growth player that shipped briefly on gh-pages.

Outputs (gitignored under breast-cancer-sim/data/processed/breast-frames-jasim/):

  breast_<case_id>_frames.npy
  breast_<case_id>_frames.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BREAST = HERE.parent
VINESH = BREAST / "simulation-vinesh-philip-chandan/vinesh"
PDE_ROOT = BREAST / "data/processed/pde-input-vinesh"
MANIFEST = BREAST / "data/processed/raw-extract-philip-chandan/manifest.json"
OUT_DIR = BREAST / "data/processed/breast-frames-jasim"

sys.path.insert(0, str(VINESH))
from run_growth import run_growth  # noqa: E402

N_KEEP = 26
ISO = 0.15
TIMESTEPS = 50
DT = 0.1
DEMO_VERSION = "pde-baseline-risk-v1"

SCENARIOS = {
    "luminal_a": {
        "tcga": "TCGA-AR-A1AX",
        "label": "Luminal A",
        "risk": 0.7,
        "color": "#46e0b0",
    },
    "basal": {
        "tcga": "TCGA-AR-A1AQ",
        "label": "Basal-like",
        "risk": 1.6,
        "color": "#ff3b54",
    },
}


def _manifest_row(tcga: str) -> dict:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in data.get("patients", []):
        if row.get("tcga_id") == tcga:
            return row
    raise KeyError(f"{tcga} not in manifest patients")


def _baseline_path(tcga: str) -> Path:
    return PDE_ROOT / tcga / "g64" / "baseline.npy"


def frame_path(out_dir: Path, key: str) -> Path:
    return out_dir / f"breast_{key}_frames.npy"


def meta_path(out_dir: Path, key: str) -> Path:
    return out_dir / f"breast_{key}_frames.json"


def _simulate(key: str) -> tuple[np.ndarray, dict]:
    sc = SCENARIOS[key]
    tcga = sc["tcga"]
    baseline = np.load(_baseline_path(tcga)).astype(np.float32)
    row = _manifest_row(tcga)
    interval_days = float(row["interval_days"])
    frames = run_growth(
        baseline,
        params={"risk_multiplier": sc["risk"]},
        timesteps=TIMESTEPS,
        dt=DT,
    )
    idx = np.linspace(0, len(frames) - 1, N_KEEP).round().astype(int)
    stack = np.stack([frames[i] for i in idx]).astype(np.float32)
    meta = {
        "case_id": key,
        "tcga": tcga,
        "label": sc["label"],
        "slug": row["baseline_slug"],
        "baseline_date": row.get("baseline_study_date"),
        "interval_days": interval_days,
        "risk_multiplier": sc["risk"],
        "iso": ISO,
        "n_frames": int(stack.shape[0]),
        "sim_timesteps": TIMESTEPS,
        "sim_dt": DT,
    }
    return stack, meta


def generate(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for key in SCENARIOS:
        stack, meta = _simulate(key)
        npy = frame_path(out_dir, key)
        js = meta_path(out_dir, key)
        np.save(npy, stack)
        js.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        written[key] = npy
        b0 = float(stack[0].sum()) or 1.0
        peak = round(100 * float(stack[-1].sum()) / b0)
        print(f"{key:10s} {meta['tcga']}  {stack.shape}  risk ×{meta['risk_multiplier']:.1f}  "
              f"peak idx {peak}  -> {npy.name}")
    (out_dir / ".demo_version").write_text(DEMO_VERSION + "\n", encoding="utf-8")
    return written


def ensure_frames(out_dir: Path = OUT_DIR) -> None:
    marker = out_dir / ".demo_version"
    needed = [frame_path(out_dir, k) for k in SCENARIOS]
    stale = not marker.is_file() or marker.read_text(encoding="utf-8").strip() != DEMO_VERSION
    if stale or not all(p.exists() for p in needed):
        print("breast frames missing or stale -> regenerating from PDE baselines")
        generate(out_dir)


if __name__ == "__main__":
    generate()
