"""Export a breast PDE growth sequence as a frame stack for visualization.

Outputs:
  <out>/<slug>_frames.npy   float32 (T, Z, Y, X)
  <out>/<slug>_frames.json  metadata and per-frame tumor volumes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VINESH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VINESH_DIR))

from mask_seeding import seed_from_mask  # noqa: E402
from run_growth import run_growth  # noqa: E402
from tumor_pde_solver import total_volume  # noqa: E402


def export_frames(
    baseline: np.ndarray,
    slug: str,
    out_dir: Path,
    *,
    risk_multiplier: float = 1.2,
    params: dict | None = None,
    timesteps: int = 50,
    dt: float = 0.1,
    n_keep: int = 26,
    is_mask: bool = False,
    spacing=(1.0, 1.0, 1.0),
    volume_threshold: float = 0.15,
    interval_days: float | None = None,
    meta_extra: dict | None = None,
) -> tuple[Path, Path]:
    """Simulate growth and write a frame stack plus metadata."""
    vol = np.asarray(baseline, dtype=np.float32)
    if is_mask:
        vol = seed_from_mask(vol)

    sim_params = {"risk_multiplier": risk_multiplier, "spacing": spacing, **(params or {})}
    frames = run_growth(vol, params=sim_params, timesteps=timesteps, dt=dt)

    keep_count = min(max(int(n_keep), 2), len(frames))
    idx = np.linspace(0, len(frames) - 1, keep_count).round().astype(int)
    kept = [frames[i] for i in idx]
    stack = np.stack(kept).astype(np.float32)

    days_per_step = None
    if interval_days is not None and len(kept) > 1:
        days_per_step = float(interval_days) / (len(kept) - 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{slug}_frames.npy"
    json_path = out_dir / f"{slug}_frames.json"
    np.save(npy_path, stack)

    meta = {
        "slug": slug,
        "frames_file": npy_path.name,
        "n_frames": int(stack.shape[0]),
        "shape_per_frame": list(stack.shape[1:]),
        "dtype": "float32",
        "axis_order": ["T", "Z", "Y", "X"],
        "value_range": [0.0, 1.0],
        "spacing_mm": list(spacing),
        "interval_days": interval_days,
        "days_per_step": days_per_step,
        "risk_multiplier": float(risk_multiplier),
        "volume_threshold": float(volume_threshold),
        "sim_params": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in sim_params.items()
        },
        "timesteps": int(timesteps),
        "dt": float(dt),
        "volume_mm3_per_frame": [
            round(total_volume(frame, spacing, threshold=volume_threshold), 1)
            for frame in kept
        ],
        "integrated_burden_per_frame": [round(float(frame.sum()), 4) for frame in kept],
    }
    if meta_extra:
        meta.update(meta_extra)
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return npy_path, json_path


def load_sequence(path_or_slug) -> tuple[list[np.ndarray], dict]:
    """Load a sequence written by ``export_frames``."""
    p = Path(path_or_slug)
    if p.suffix == ".npy":
        npy_path = p
    elif p.name.endswith("_frames"):
        npy_path = p.with_suffix(".npy")
    else:
        npy_path = p.with_name(p.name + "_frames.npy")
    json_path = npy_path.with_name(npy_path.name.replace(".npy", ".json"))
    stack = np.load(npy_path)
    frames = [np.asarray(frame, dtype=np.float32) for frame in stack]
    meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    return frames, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="PDE-ready baseline .npy")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--out", type=Path, default=VINESH_DIR / "frames-for-jasim")
    parser.add_argument("--risk", type=float, default=1.2)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--n-keep", type=int, default=26)
    parser.add_argument("--volume-threshold", type=float, default=0.15)
    parser.add_argument("--interval-days", type=float, default=None)
    parser.add_argument("--is-mask", action="store_true", help="baseline is a mask, not a density field")
    args = parser.parse_args()

    baseline = np.load(args.baseline)
    npy_path, json_path = export_frames(
        baseline,
        args.slug,
        args.out,
        risk_multiplier=args.risk,
        params={"delta": args.delta},
        timesteps=args.timesteps,
        dt=args.dt,
        n_keep=args.n_keep,
        is_mask=args.is_mask,
        volume_threshold=args.volume_threshold,
        interval_days=args.interval_days,
        meta_extra={"source_baseline": str(args.baseline)},
    )
    print(f"wrote {npy_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
