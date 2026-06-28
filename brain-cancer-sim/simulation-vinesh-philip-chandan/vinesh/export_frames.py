"""Export a PDE growth sequence as a frame stack for the visualizer (Jasim).

Turns a PDE-ready baseline volume into an animated growth sequence and writes it
in the shape Jasim's render_3d expects (float32 (Z,Y,X) density in [0,1]).

Outputs (per case):
  <out>/<slug>_frames.npy   float32 (T, Z, Y, X)  -- the sequence
  <out>/<slug>_frames.json  metadata: risk_multiplier, dt, spacing, per-frame mm^3

Jasim loads it with:
    import numpy as np
    frames = list(np.load("<slug>_frames.npy"))   # each frame is (Z,Y,X) in [0,1]
    render_volume(frames[i])                       # or animate over frames

Usage:
    python export_frames.py --baseline <path.npy> --slug demo --risk 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VINESH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VINESH_DIR))

from tumor_pde_solver import solve_growth, total_volume  # noqa: E402

try:
    from mask_seeding import seed_from_mask  # noqa: E402
except Exception:  # mask seeding optional if baseline is already a density field
    seed_from_mask = None


def export_frames(
    baseline: np.ndarray,
    slug: str,
    out_dir: Path,
    *,
    risk_multiplier: float = 1.5,
    timesteps: int = 50,
    dt: float = 0.1,
    n_keep: int = 26,
    is_mask: bool = False,
    spacing=(1.0, 1.0, 1.0),
) -> tuple[Path, Path]:
    """Simulate growth and write a frame stack + metadata for the visualizer."""
    vol = np.asarray(baseline, dtype=np.float32)
    if is_mask:
        if seed_from_mask is None:
            raise RuntimeError("mask_seeding unavailable; pass a density field instead")
        vol = seed_from_mask(vol)

    frames = solve_growth(vol, timesteps, dt, params={"risk_multiplier": risk_multiplier})

    # Keep ~n_keep evenly spaced frames so the web payload stays light.
    idx = np.linspace(0, len(frames) - 1, min(n_keep, len(frames))).round().astype(int)
    kept = [frames[i] for i in idx]
    stack = np.stack(kept).astype(np.float32)  # (T, Z, Y, X)

    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{slug}_frames.npy"
    json_path = out_dir / f"{slug}_frames.json"
    np.save(npy_path, stack)

    meta = {
        "slug": slug,
        "n_frames": int(stack.shape[0]),
        "shape_per_frame": list(stack.shape[1:]),
        "dtype": "float32",
        "axis_order": ["T", "Z", "Y", "X"],
        "value_range": [0.0, 1.0],
        "spacing_mm": list(spacing),
        "risk_multiplier": float(risk_multiplier),
        "timesteps": int(timesteps),
        "dt": float(dt),
        "sim_time_index": [int(i) for i in idx],
        "volume_mm3_per_frame": [round(total_volume(f, spacing), 1) for f in kept],
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return npy_path, json_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True, help="PDE-ready baseline .npy")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", type=Path, default=VINESH_DIR / "frames-for-jasim")
    ap.add_argument("--risk", type=float, default=1.5)
    ap.add_argument("--timesteps", type=int, default=50)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--n-keep", type=int, default=26)
    ap.add_argument("--is-mask", action="store_true", help="baseline is a raw mask -> seed it")
    args = ap.parse_args()

    vol = np.load(args.baseline)
    npy, js = export_frames(
        vol, args.slug, args.out,
        risk_multiplier=args.risk, timesteps=args.timesteps, dt=args.dt,
        n_keep=args.n_keep, is_mask=args.is_mask,
    )
    print(f"wrote {npy}")
    print(f"wrote {js}")


if __name__ == "__main__":
    main()
