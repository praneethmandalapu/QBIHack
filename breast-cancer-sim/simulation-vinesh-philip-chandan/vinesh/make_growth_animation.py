"""Render calibrated tumor-growth animations from real two-timepoint data.

For each patient: isolate the baseline tumor, calibrate the PDE against the real
followup (calibrate.py), forward-simulate, and write a GIF showing
  (left)  the tumor density slice evolving over time, and
  (right) the simulated burden curve landing on the *real* followup burden.

This is a QC/demo artifact (Vinesh). The production 3D view is Jasim's
render_3d; this just proves the calibrated prediction visually.

Usage:
    python make_growth_animation.py --data-dir <pde-input dir> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from calibrate import calibrate_growth, predict_trajectory, tumor_burden  # noqa: E402

# Patient registry for the spike: (baseline slug, followup slug, days, subtype).
PATIENTS = [
    ("luminal_a_TCGA-AR-A1AX_baseline", "luminal_a_TCGA-AR-A1AX_followup", 377, "Luminal A (low-risk)"),
    ("basal_TCGA-AR-A1AQ_baseline", "basal_TCGA-AR-A1AQ_followup", 532, "Basal-like (high-risk)"),
]

TIMESTEPS = 50
DT = 0.1


def _best_slice(volume: np.ndarray) -> int:
    """Z index with the most tumor, for a representative axial view."""
    return int(volume.sum(axis=(1, 2)).argmax())


def animate_patient(base_path: Path, fu_path: Path, days: int, subtype: str, out_path: Path) -> dict:
    baseline = np.load(base_path)
    followup = np.load(fu_path)

    cal = calibrate_growth(baseline, followup, TIMESTEPS, DT)
    base_iso = cal["baseline_iso"]
    frames = predict_trajectory(base_iso, cal["params"], TIMESTEPS, DT)

    burdens = np.array([tumor_burden(f) for f in frames])
    day_axis = np.linspace(0, days, len(frames))
    target_burden = cal["target_burden"]
    z = _best_slice(base_iso)
    vmax = float(max(base_iso.max(), 1e-6))

    fig, (ax_img, ax_curve) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"{subtype}  —  {cal['regime']} calibrated to real followup", fontsize=12)

    im = ax_img.imshow(frames[0][z], cmap="inferno", vmin=0, vmax=vmax)
    ax_img.set_title("tumor density (axial slice)")
    ax_img.axis("off")
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    ax_curve.plot(day_axis, burdens, color="#888", lw=1, ls="--", label="simulated trajectory")
    ax_curve.scatter([0], [burdens[0]], color="#1f77b4", zorder=5, label="baseline (real)")
    ax_curve.scatter([days], [target_burden], marker="*", s=180, color="#d62728",
                     zorder=5, label="followup (real)")
    (moving,) = ax_curve.plot([], [], "o", color="#2ca02c", ms=9, zorder=6, label="prediction")
    ax_curve.set_xlabel("days from baseline")
    ax_curve.set_ylabel("tumor burden (sum of density)")
    ax_curve.set_title("predicted burden vs real followup")
    ax_curve.legend(loc="best", fontsize=8)
    ax_curve.grid(alpha=0.3)

    def update(i):
        im.set_data(frames[i][z])
        ax_img.set_xlabel(f"day {day_axis[i]:.0f}")
        moving.set_data([day_axis[i]], [burdens[i]])
        return im, moving

    anim = FuncAnimation(fig, update, frames=len(frames), interval=120, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=8))
    plt.close(fig)

    return {
        "out": out_path,
        "regime": cal["regime"],
        "knob": (cal["knob_name"], cal["knob_value"]),
        "burden_error_pct": cal["burden_error_pct"],
        "real_change_pct": 100.0 * (target_burden - cal["baseline_burden"]) / cal["baseline_burden"],
    }


def main() -> None:
    here = Path(__file__).resolve()
    default_data = here.parents[3] / "data" / "processed" / "pde-input-vinesh"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--out", type=Path, default=here.parent / "animations")
    args = parser.parse_args()

    for base_slug, fu_slug, days, subtype in PATIENTS:
        base_path = args.data_dir / f"{base_slug}.npy"
        fu_path = args.data_dir / f"{fu_slug}.npy"
        if not (base_path.exists() and fu_path.exists()):
            print(f"SKIP {subtype}: missing {base_path.name} or {fu_path.name}")
            continue
        out_path = args.out / f"{base_slug.rsplit('_', 1)[0]}_growth.gif"
        info = animate_patient(base_path, fu_path, days, subtype, out_path)
        knob_name, knob_val = info["knob"]
        print(f"{subtype}: {info['regime']}  {knob_name}={knob_val:.3f}  "
              f"real change {info['real_change_pct']:+.0f}%  "
              f"fit error {info['burden_error_pct']:+.2f}%  ->  {info['out']}")


if __name__ == "__main__":
    main()
