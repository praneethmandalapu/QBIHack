"""Mid-Z QC overlays for segmentation methods."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seg_paths import ensure_segmentation_dirs, qc_overlay_png


def _peak_z_index(mask: np.ndarray, fallback_shape_z: int) -> int:
    if mask.any():
        return int(mask.sum(axis=(1, 2)).argmax())
    return fallback_shape_z // 2


def save_mid_z_overlay(
    slug: str,
    method: str,
    norm: np.ndarray,
    mask: np.ndarray,
    *,
    title: str,
    fill_rgba: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.45),
    contour_color: str = "#00FF00",
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    ensure_segmentation_dirs()
    out_path = qc_overlay_png(slug, method)
    z_idx = _peak_z_index(mask, norm.shape[0])

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(norm[z_idx], cmap="gray", vmin=0, vmax=1)
    slice_mask = mask[z_idx]
    if slice_mask.any():
        overlay = colors.to_rgba(contour_color, alpha=fill_rgba[3])
        tint = np.zeros((*slice_mask.shape, 4), dtype=np.float32)
        tint[slice_mask > 0] = overlay
        axis.imshow(tint)
        axis.contour(slice_mask, levels=[0.5], colors=[contour_color], linewidths=2.0)
    axis.set_title(title, fontsize=10)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
