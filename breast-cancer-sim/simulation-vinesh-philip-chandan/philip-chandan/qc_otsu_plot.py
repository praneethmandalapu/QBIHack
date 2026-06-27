"""QC PNGs documenting Vinesh Otsu tumor segmentation on Philip-Chandan raw extracts."""

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

from spike_paths import QC_OTSU_PLOTS_VINESH, ensure_spike_dirs  # noqa: E402
from prepare_pde_input import load_raw_extract, prepare_pde_stages  # noqa: E402

OTSU_CONTOUR_COLOR = "magenta"
PDE_CMAP = "inferno"


def otsu_norm_overlay_path(slug: str) -> Path:
    return QC_OTSU_PLOTS_VINESH / f"{slug}_otsu-norm-overlay.png"


def pde_input_slice_path(slug: str) -> Path:
    return QC_OTSU_PLOTS_VINESH / f"{slug}_pde-input-mid-z.png"


def pick_tumor_z_index(mask: np.ndarray) -> int:
    if mask.any():
        return int(mask.sum(axis=(1, 2)).argmax())
    return mask.shape[0] // 2


def _load_stages(slug: str) -> dict:
    raw_volume, raw_metadata = load_raw_extract(slug)
    stages = prepare_pde_stages(raw_volume, raw_metadata["spacing_mm"])
    stages["raw_metadata"] = raw_metadata
    return stages


def save_otsu_norm_overlay_plot(slug: str) -> Path:
    """Normalized resampled slice with Otsu tumor contour (pre-crop)."""
    ensure_spike_dirs()
    stages = _load_stages(slug)
    norm = stages["normalized"]
    tumor_mask = stages["tumor_mask"]
    meta = stages["raw_metadata"]
    threshold = stages["otsu_threshold"]

    z_idx = pick_tumor_z_index(tumor_mask)
    out_path = otsu_norm_overlay_path(slug)

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(norm[z_idx], cmap="gray", vmin=0, vmax=1)
    if tumor_mask[z_idx].any():
        axis.contour(tumor_mask[z_idx], levels=[0.5], colors=OTSU_CONTOUR_COLOR, linewidths=1.0)
    thresh_text = f"{threshold:.3f}" if threshold is not None else "n/a"
    axis.set_title(
        f"{meta.get('tcga_id', slug)} Otsu z={z_idx} thresh={thresh_text} "
        f"(mask vox={int(tumor_mask.sum())})"
    )
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_pde_input_slice_plot(slug: str) -> Path:
    """Cropped 64^3 PDE input slice showing continuous tumor density after Otsu."""
    ensure_spike_dirs()
    stages = _load_stages(slug)
    pde_volume = stages["pde_volume"]
    meta = stages["raw_metadata"]
    background = float(stages["background_value"])

    tumor_mask = pde_volume > background
    z_idx = pick_tumor_z_index(tumor_mask)
    out_path = pde_input_slice_path(slug)

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 6))
    slice_2d = pde_volume[z_idx]
    axis.imshow(slice_2d, cmap=PDE_CMAP, vmin=0, vmax=1)
    if tumor_mask[z_idx].any():
        axis.contour(tumor_mask[z_idx], levels=[0.5], colors="cyan", linewidths=0.8)
    axis.set_title(
        f"{meta.get('tcga_id', slug)} PDE input z={z_idx} shape={list(pde_volume.shape)}"
    )
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_otsu_qc_plots(slug: str) -> tuple[Path, Path]:
    """Write both Otsu documentation PNGs for one slug."""
    return save_otsu_norm_overlay_plot(slug), save_pde_input_slice_plot(slug)


def ensure_otsu_norm_overlay(slug: str) -> Path | None:
    out_path = otsu_norm_overlay_path(slug)
    if out_path.is_file():
        return out_path
    try:
        return save_otsu_norm_overlay_plot(slug)
    except FileNotFoundError:
        return None


def ensure_pde_input_slice(slug: str) -> Path | None:
    out_path = pde_input_slice_path(slug)
    if out_path.is_file():
        return out_path
    try:
        return save_pde_input_slice_plot(slug)
    except FileNotFoundError:
        return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Save Otsu segmentation QC PNGs for a slug.")
    parser.add_argument("--slug", default="luminal_a_TCGA-AR-A1AX_baseline")
    args = parser.parse_args()
    norm_path, pde_path = save_otsu_qc_plots(args.slug)
    print(f"Wrote {norm_path}")
    print(f"Wrote {pde_path}")


if __name__ == "__main__":
    main()
