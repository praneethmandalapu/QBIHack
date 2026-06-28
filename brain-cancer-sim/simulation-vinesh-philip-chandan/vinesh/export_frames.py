"""Export a PDE growth sequence as a frame stack for the visualizer (Jasim).

Turns a PDE-ready baseline volume into an animated growth sequence and writes it
in the shape Jasim's render_3d expects (float32 (Z,Y,X) density in [0,1]).

Outputs (per case):
  <out>/<slug>_frames.npy   float32 (T, Z, Y, X)  -- the sequence
  <out>/<slug>_frames.json  metadata (see schema below)

Jasim loads and renders with:
    from export_frames import load_sequence
    frames, meta = load_sequence("<out>/<slug>")          # frames: list of (Z,Y,X)
    render_sequence(frames)                                # animated 3D player
    growth_analytics(frames, spacing=meta["spacing_mm"],
                     days_per_step=meta["days_per_step"])  # true-scale KPIs/curve

CLI:
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
    params: dict | None = None,
    timesteps: int = 50,
    dt: float = 0.1,
    n_keep: int = 26,
    is_mask: bool = False,
    spacing=(1.0, 1.0, 1.0),
    interval_days: float | None = None,
    meta_extra: dict | None = None,
) -> tuple[Path, Path]:
    """Simulate growth and write a frame stack + metadata for the visualizer.

    `params` is merged over {"risk_multiplier": risk_multiplier}, so a regression
    case can pass {"delta": d}. `interval_days` (real baseline->followup gap)
    sets days_per_step so Jasim's analytics show true time.
    """
    vol = np.asarray(baseline, dtype=np.float32)
    if is_mask:
        if seed_from_mask is None:
            raise RuntimeError("mask_seeding unavailable; pass a density field instead")
        vol = seed_from_mask(vol)

    sim_params = {"risk_multiplier": risk_multiplier, **(params or {})}
    frames = solve_growth(vol, timesteps, dt, params=sim_params)

    # Keep ~n_keep evenly spaced frames so the web payload stays light.
    idx = np.linspace(0, len(frames) - 1, min(n_keep, len(frames))).round().astype(int)
    kept = [frames[i] for i in idx]
    stack = np.stack(kept).astype(np.float32)  # (T, Z, Y, X)

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
        "sim_params": {k: float(v) for k, v in sim_params.items()},
        "timesteps": int(timesteps),
        "dt": float(dt),
        "volume_mm3_per_frame": [round(total_volume(f, spacing), 1) for f in kept],
        # Robust burden = integral of density (Sigma-u). Unlike the thresholded
        # volume above (sensitive near the start), this is diffusion-stable —
        # prefer it for the burden-over-time curve. See HANDOFF_JASIM.md s7.
        "burden_sum_per_frame": [round(float(f.sum()), 2) for f in kept],
    }
    if meta_extra:
        meta.update(meta_extra)
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return npy_path, json_path


def load_sequence(path_or_slug) -> tuple[list[np.ndarray], dict]:
    """Load a sequence written by export_frames.

    Accepts either the stem (".../slug") or the full "..._frames.npy" path.
    Returns (frames, metadata): frames is a list of (Z,Y,X) float32 arrays ready
    for render_sequence / render_volume.
    """
    p = Path(path_or_slug)
    if p.suffix == ".npy":
        npy_path = p
    elif p.name.endswith("_frames"):
        npy_path = p.with_name(p.name + ".npy")
    else:
        npy_path = p.with_name(p.name + "_frames.npy")
    json_path = npy_path.with_name(npy_path.name.replace(".npy", ".json"))
    stack = np.load(npy_path)
    frames = [np.asarray(f, dtype=np.float32) for f in stack]
    meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    return frames, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True, help="PDE-ready baseline .npy")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", type=Path, default=VINESH_DIR / "frames-for-jasim")
    ap.add_argument("--risk", type=float, default=1.5)
    ap.add_argument("--delta", type=float, default=0.0, help="death rate for regression cases")
    ap.add_argument("--timesteps", type=int, default=50)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--n-keep", type=int, default=26)
    ap.add_argument("--interval-days", type=float, default=None)
    ap.add_argument("--is-mask", action="store_true", help="baseline is a raw mask -> seed it")
    args = ap.parse_args()

    vol = np.load(args.baseline)
    npy, js = export_frames(
        vol, args.slug, args.out,
        risk_multiplier=args.risk, params={"delta": args.delta},
        timesteps=args.timesteps, dt=args.dt, n_keep=args.n_keep,
        is_mask=args.is_mask, interval_days=args.interval_days,
    )
    print(f"wrote {npy}")
    print(f"wrote {js}")


if __name__ == "__main__":
    main()
