"""QC PNGs documenting Vinesh expert-mask PDE prep on Philip-Chandan raw extracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
VINESH_DIR = SPIKE_ROOT / "vinesh"
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SPIKE_ROOT))
sys.path.insert(0, str(VINESH_DIR))

from handoff_contract import default_grid_size, grid_size_options  # noqa: E402
from spike_paths import ensure_spike_dirs, qc_pde_prep_dir  # noqa: E402
from prepare_pde_input import (  # noqa: E402
    load_expert_mask,
    load_raw_extract,
    prepare_pde_stages,
)

MASK_CONTOUR_COLOR = "lime"
PDE_CMAP = "inferno"


def expert_seg_overlay_path(slug: str, *, grid_size: int | None = None) -> Path:
    size = grid_size or default_grid_size()
    return qc_pde_prep_dir(size) / f"{slug}_expert-seg-overlay.png"


def pde_input_slice_path(slug: str, *, grid_size: int | None = None) -> Path:
    size = grid_size or default_grid_size()
    return qc_pde_prep_dir(size) / f"{slug}_pde-input-mid-z.png"


def pde_input_longitudinal_path(patient_id: str, *, grid_size: int | None = None) -> Path:
    size = grid_size or default_grid_size()
    return qc_pde_prep_dir(size) / f"{patient_id}_pde-input-longitudinal-mid-z.png"


def pick_tumor_z_index(mask: np.ndarray) -> int:
    if mask.any():
        return int(mask.sum(axis=(1, 2)).argmax())
    return mask.shape[0] // 2


def _load_stages(slug: str, *, grid_size: int | None = None) -> dict:
    raw_volume, raw_metadata = load_raw_extract(slug)
    expert_mask, _ = load_expert_mask(raw_metadata, raw_volume.shape)
    stages = prepare_pde_stages(
        raw_volume,
        raw_metadata["spacing_mm"],
        expert_mask,
        grid_size=grid_size,
    )
    stages["raw_metadata"] = raw_metadata
    return stages


def save_expert_seg_overlay_plot(slug: str, *, grid_size: int | None = None) -> Path:
    """Normalized resampled slice with expert tumor contour (pre-crop)."""
    ensure_spike_dirs()
    stages = _load_stages(slug, grid_size=grid_size)
    norm = stages["normalized"]
    tumor_mask = stages["resampled_mask"]
    meta = stages["raw_metadata"]
    size = int(stages["grid_size"])

    z_idx = pick_tumor_z_index(tumor_mask)
    out_path = expert_seg_overlay_path(slug, grid_size=size)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(norm[z_idx], cmap="gray", vmin=0, vmax=1)
    if tumor_mask[z_idx].any():
        axis.contour(tumor_mask[z_idx], levels=[0.5], colors=MASK_CONTOUR_COLOR, linewidths=1.0)
    axis.set_title(
        f"{meta.get('patient_id', slug)} g{size} expert mask z={z_idx} "
        f"(mask vox={int(tumor_mask.sum())})"
    )
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_pde_input_slice_plot(slug: str, *, grid_size: int | None = None) -> Path:
    """Cropped PDE input slice showing continuous tumor density."""
    ensure_spike_dirs()
    stages = _load_stages(slug, grid_size=grid_size)
    pde_volume = stages["pde_volume"]
    meta = stages["raw_metadata"]
    background = float(stages["background_value"])
    size = int(stages["grid_size"])

    tumor_mask = pde_volume > background
    z_idx = pick_tumor_z_index(tumor_mask)
    out_path = pde_input_slice_path(slug, grid_size=size)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 6))
    slice_2d = pde_volume[z_idx]
    axis.imshow(slice_2d, cmap=PDE_CMAP, vmin=0, vmax=1)
    if tumor_mask[z_idx].any():
        axis.contour(tumor_mask[z_idx], levels=[0.5], colors="cyan", linewidths=0.8)
    axis.set_title(
        f"{meta.get('patient_id', slug)} g{size} PDE z={z_idx} shape={list(pde_volume.shape)}"
    )
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_pde_input_longitudinal_plot(
    patient_id: str,
    *,
    baseline_slug: str | None = None,
    followup_slug: str | None = None,
    grid_size: int | None = None,
) -> Path:
    """Side-by-side baseline vs follow-up cropped PDE input slices."""
    ensure_spike_dirs()
    size = grid_size or default_grid_size()
    baseline_slug = baseline_slug or f"glioma_ucsf_{patient_id}_baseline"
    followup_slug = followup_slug or f"glioma_ucsf_{patient_id}_followup"

    baseline_stages = _load_stages(baseline_slug, grid_size=size)
    followup_stages = _load_stages(followup_slug, grid_size=size)
    background = float(baseline_stages["background_value"])

    panels = [
        (baseline_stages["pde_volume"], "Baseline (t1)"),
        (followup_stages["pde_volume"], "Follow-up (t2)"),
    ]

    import matplotlib.pyplot as plt

    out_path = pde_input_longitudinal_path(patient_id, grid_size=size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, (pde_volume, label) in zip(axes, panels):
        tumor_mask = pde_volume > background
        z_idx = pick_tumor_z_index(tumor_mask)
        slice_2d = pde_volume[z_idx]
        axis.imshow(slice_2d, cmap=PDE_CMAP, vmin=0, vmax=1)
        if tumor_mask[z_idx].any():
            axis.contour(tumor_mask[z_idx], levels=[0.5], colors="cyan", linewidths=0.8)
        axis.set_title(
            f"{label}  z={z_idx}  shape={list(pde_volume.shape)}",
            fontsize=10,
        )
        axis.axis("off")

    fig.suptitle(
        f"Patient {patient_id} g{size} cropped PDE input (inferno, cyan contour > 0)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_pde_qc_plots(slug: str, *, grid_size: int | None = None) -> tuple[Path, Path]:
    return (
        save_expert_seg_overlay_plot(slug, grid_size=grid_size),
        save_pde_input_slice_plot(slug, grid_size=grid_size),
    )


def ensure_expert_seg_overlay(slug: str, *, grid_size: int | None = None) -> Path | None:
    out_path = expert_seg_overlay_path(slug, grid_size=grid_size)
    if out_path.is_file():
        return out_path
    try:
        return save_expert_seg_overlay_plot(slug, grid_size=grid_size)
    except FileNotFoundError:
        return None


def ensure_pde_input_slice(slug: str, *, grid_size: int | None = None) -> Path | None:
    out_path = pde_input_slice_path(slug, grid_size=grid_size)
    if out_path.is_file():
        return out_path
    try:
        return save_pde_input_slice_plot(slug, grid_size=grid_size)
    except FileNotFoundError:
        return None


def ensure_pde_input_longitudinal(
    patient_id: str,
    *,
    grid_size: int | None = None,
    force: bool = False,
) -> Path | None:
    out_path = pde_input_longitudinal_path(patient_id, grid_size=grid_size)
    if out_path.is_file() and not force:
        return out_path
    try:
        return save_pde_input_longitudinal_plot(patient_id, grid_size=grid_size)
    except FileNotFoundError:
        return None


def main() -> None:
    import argparse

    from handoff_contract import spike_patient

    parser = argparse.ArgumentParser(description="Save expert-mask PDE prep QC PNGs for a slug.")
    parser.add_argument("--slug", default=spike_patient()["slug"])
    parser.add_argument(
        "--grid-size",
        type=int,
        choices=grid_size_options(),
        default=None,
    )
    parser.add_argument("--all-grids", action="store_true")
    args = parser.parse_args()

    sizes = list(grid_size_options()) if args.all_grids else [args.grid_size or default_grid_size()]
    for size in sizes:
        seg_path, pde_path = save_pde_qc_plots(args.slug, grid_size=size)
        print(f"g{size}: wrote {seg_path}")
        print(f"g{size}: wrote {pde_path}")


if __name__ == "__main__":
    main()
