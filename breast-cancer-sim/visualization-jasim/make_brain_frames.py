"""Generate the BRAIN demo frame stacks the site renders (Person 3 / jasim).

Seeds the *real* UCSF glioma baseline (patient 100002) and forward-simulates two
biologically distinct scenarios with Vinesh's reaction-diffusion solver:

  • aggressive (IDH-wildtype): high diffusion D -> the tumor margin actually
    INVADES outward (a travelling front), not just brightens in place.
  • indolent  (IDH-mutant): low D + a death term -> the lesion stays ~stable /
    slightly regresses, matching this patient's real +2.8% follow-up.

Why this exists (the bug it fixes):
  The old demo seeded a flat ~0.30 density plateau and grew it with a small
  proliferation multiplier over a very short horizon. Fisher-KPP front speed is
  c ~ 2*sqrt(D*rho), so over T=5 the front moved <2 voxels: total burden rose
  but the lesion did not expand — it densified in place. Philip (biology) flagged
  it as unrealistic. Two changes make growth read as growth:
    1. Seed the real lesion near carrying capacity (u~0.9), so the logistic term
       has little in-place headroom and growth must come from the invading front.
    2. Raise D (invasion) and simulate a real horizon, so the front travels.

The .npy stacks are gitignored (too large for git); build_site.py calls
ensure_frames() to regenerate them on demand from the (drive-synced) seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BREAST = HERE.parent                      # breast-cancer-sim/
REPO = BREAST.parent                      # qbihack/
BRAIN_REPO = REPO / "brain-cancer-sim"
SEED_PATH = (BRAIN_REPO / "data/processed/pde-input-vinesh"
             / "glioma_ucsf_100002_baseline.npy")
OUT_DIR = BREAST / "data/processed/brain-frames-jasim"

# Vinesh's solver (disease-agnostic; identical contract in both repos).
sys.path.insert(0, str(BRAIN_REPO / "simulation-vinesh-philip-chandan/vinesh"))
from tumor_pde_solver import solve_growth  # noqa: E402

N_KEEP, DT = 21, 0.1

# Per-scenario PDE parameters. D drives spatial invasion (the missing ingredient
# before); risk_multiplier scales proliferation; delta is net cell death.
SCENARIOS = {
    # IDH-wildtype GBM: fast outward invasion, volume ~+200% over the window.
    "aggressive": {"D": 0.40, "risk_multiplier": 1.0, "delta": 0.0, "timesteps": 170},
    # IDH-mutant: minimal invasion + attrition -> roughly stable / slight regress.
    "indolent": {"D": 0.10, "risk_multiplier": 0.5, "delta": 0.05, "timesteps": 160},
}


def load_seed() -> np.ndarray:
    """Real 100002 lesion, rescaled so the tumor sits near carrying capacity.

    The prepared input is a low (~0.30) intensity plateau. Rescaling its 99th
    percentile to ~0.92 (shape/texture preserved, background untouched) makes the
    isosurface at 0.5 meaningful from frame 0 and forces growth to come from the
    front rather than from filling in headroom below carrying capacity.
    """
    raw = np.load(SEED_PATH).astype(np.float32)
    nz = raw[raw > 0]
    if nz.size == 0:
        raise ValueError(f"Seed {SEED_PATH} has no tumor voxels")
    hi = float(np.percentile(nz, 99.0))
    seed = np.clip(raw / hi * 0.92, 0.0, 1.0).astype(np.float32)
    seed[raw <= 0] = 0.0
    return seed


def _stack(seed: np.ndarray, p: dict) -> np.ndarray:
    frames = solve_growth(
        seed, p["timesteps"], DT,
        {"D": p["D"], "risk_multiplier": p["risk_multiplier"], "delta": p["delta"]},
    )
    idx = np.linspace(0, len(frames) - 1, N_KEEP).round().astype(int)
    return np.stack([frames[i] for i in idx]).astype(np.float32)  # (T, Z, Y, X)


def generate(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = load_seed()
    written = {}
    for key, p in SCENARIOS.items():
        stack = _stack(seed, p)
        path = out_dir / f"glioma_100002_{key}_frames.npy"
        np.save(path, stack)
        vol = np.array([(stack[i] > 0.5).sum() for i in range(len(stack))], float)
        written[key] = path
        print(f"{key:10s} {stack.shape}  tumor-volume idx 100 -> "
              f"{100 * vol[-1] / max(vol[0], 1):.0f}  peak {stack[-1].max():.2f}  -> {path.name}")
    return written


def ensure_frames(out_dir: Path = OUT_DIR) -> None:
    """Generate the demo stacks if they are missing (gitignored on fresh clones)."""
    needed = [out_dir / f"glioma_100002_{k}_frames.npy" for k in SCENARIOS]
    if all(p.exists() for p in needed):
        return
    print("brain frames missing -> regenerating from real seed")
    generate(out_dir)


if __name__ == "__main__":
    generate()
