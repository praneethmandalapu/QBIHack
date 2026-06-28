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


def save_cuboid_enhancement_overlay(
    slug: str,
    norm: np.ndarray,
    expert_mask: np.ndarray,
    predicted_mask: np.ndarray,
    les_meta: dict,
    *,
    title: str,
) -> Path:
    """Mid-z overlay: predicted mask + .les dots + cuboid shell."""
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.patches import Rectangle

    ensure_segmentation_dirs()
    out_path = qc_overlay_png(slug, "cuboid_enhancement")

    combined = expert_mask.astype(bool) | predicted_mask.astype(bool)
    z_idx = _peak_z_index(combined, norm.shape[0])

    y0, y1 = les_meta["y_start"], les_meta["y_end"]
    x0, x1 = les_meta["x_start"], les_meta["x_end"]

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(norm[z_idx], cmap="gray", vmin=0, vmax=1)

    pred_slice = predicted_mask[z_idx]
    if pred_slice.any():
        tint = np.zeros((*pred_slice.shape, 4), dtype=np.float32)
        tint[pred_slice > 0] = colors.to_rgba("#FF8800", alpha=0.35)
        axis.imshow(tint)
        axis.contour(pred_slice, levels=[0.5], colors=["#FF8800"], linewidths=2.0)

    expert_slice = expert_mask[z_idx]
    if expert_slice.any():
        axis.contour(expert_slice, levels=[0.5], colors=["#00FF00"], linewidths=1.5)

    rect = Rectangle(
        (x0 - 0.5, y0 - 0.5),
        x1 - x0 + 1,
        y1 - y0 + 1,
        fill=False,
        edgecolor="#00BFFF",
        linewidth=1.0,
        linestyle="--",
    )
    axis.add_patch(rect)

    axis.set_title(title, fontsize=10)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
