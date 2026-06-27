"""Save middle-slice PNGs for visual QC of raw MR volumes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SPIKE_ROOT))

from spike_paths import (  # noqa: E402
    QC_SLICE_PLOTS_PHILIP_CHANDAN,
    SPIKE_PATIENT,
    ensure_spike_dirs,
    raw_extract_npy,
)
from tcia_extractor import extract_volume_for_timepoint  # noqa: E402

# Bright-voxel contour for QC overlays only — not clinical tumor segmentation.
DEFAULT_ENHANCEMENT_PERCENTILE = 90.0
OVERLAY_CONTOUR_COLOR = "lime"


def slice_plot_path(slug: str, *, overlay: bool = False) -> Path:
    suffix = "_mid-z-overlay.png" if overlay else "_mid-z.png"
    return QC_SLICE_PLOTS_PHILIP_CHANDAN / f"{slug}{suffix}"


def enhancement_contour_mask(
    slice_2d: np.ndarray,
    *,
    percentile: float = DEFAULT_ENHANCEMENT_PERCENTILE,
) -> np.ndarray:
    """Binary mask of high-intensity voxels on one slice (QC visualization only)."""
    positive = slice_2d[slice_2d > 0]
    if positive.size == 0:
        return np.zeros(slice_2d.shape, dtype=bool)
    threshold = float(np.percentile(positive, percentile))
    return slice_2d >= threshold


def pick_overlay_z_index(volume: np.ndarray) -> int:
    """Z index with the most high-intensity voxels (best slice for contrast overlay)."""
    best_z = volume.shape[0] // 2
    best_count = -1
    for z in range(volume.shape[0]):
        count = int(enhancement_contour_mask(volume[z]).sum())
        if count > best_count:
            best_z = z
            best_count = count
    return best_z


def _save_slice_figure(
    volume: np.ndarray,
    out_path: Path,
    *,
    title: str,
    z_index: int | None = None,
    overlay: bool = False,
) -> Path:
    ensure_spike_dirs()
    z_idx = volume.shape[0] // 2 if z_index is None else z_index
    z_idx = max(0, min(z_idx, volume.shape[0] - 1))
    slice_2d = volume[z_idx]

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(slice_2d, cmap="gray")
    if overlay:
        mask = enhancement_contour_mask(slice_2d)
        if mask.any():
            axis.contour(mask, levels=[0.5], colors=OVERLAY_CONTOUR_COLOR, linewidths=1.0)
    axis.set_title(title)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_middle_slice_plot(
    tcga_id: str,
    subtype: str,
    study_date: str,
    *,
    slug: str | None = None,
    volume: np.ndarray | None = None,
) -> Path:
    output_slug = slug or SPIKE_PATIENT["slug"]
    out_path = slice_plot_path(output_slug, overlay=False)

    if volume is None:
        volume = extract_volume_for_timepoint(tcga_id, subtype, study_date)
    mid_z = volume.shape[0] // 2
    return _save_slice_figure(
        volume,
        out_path,
        title=f"{tcga_id} {study_date} z={mid_z}",
        z_index=mid_z,
        overlay=False,
    )


def save_middle_slice_overlay_plot(
    tcga_id: str,
    subtype: str,
    study_date: str,
    *,
    slug: str | None = None,
    volume: np.ndarray | None = None,
) -> Path:
    """Save slice PNG with lime contour over high-intensity voxels (QC only)."""
    output_slug = slug or SPIKE_PATIENT["slug"]
    out_path = slice_plot_path(output_slug, overlay=True)

    if volume is None:
        volume = extract_volume_for_timepoint(tcga_id, subtype, study_date)
    z_idx = pick_overlay_z_index(volume)
    return _save_slice_figure(
        volume,
        out_path,
        title=(
            f"{tcga_id} {study_date} z={z_idx} "
            f"(>{DEFAULT_ENHANCEMENT_PERCENTILE:.0f}th pct contour)"
        ),
        z_index=z_idx,
        overlay=True,
    )


def save_overlay_plot_from_volume(
    volume: np.ndarray,
    slug: str,
    *,
    title: str | None = None,
) -> Path:
    """Write overlay QC PNG from an in-memory raw volume (e.g. loaded .npy)."""
    out_path = slice_plot_path(slug, overlay=True)
    z_idx = pick_overlay_z_index(volume)
    plot_title = title or f"{slug} z={z_idx} (>{DEFAULT_ENHANCEMENT_PERCENTILE:.0f}th pct contour)"
    return _save_slice_figure(
        volume,
        out_path,
        title=plot_title,
        z_index=z_idx,
        overlay=True,
    )


def ensure_overlay_plot(slug: str) -> Path | None:
    """Return overlay PNG path, generating from raw .npy on disk if needed."""
    out_path = slice_plot_path(slug, overlay=True)
    if out_path.is_file():
        return out_path

    npy_path = raw_extract_npy(slug)
    if not npy_path.is_file():
        return None

    volume = np.load(npy_path)
    return save_overlay_plot_from_volume(volume, slug)


def main() -> None:
    patient = SPIKE_PATIENT
    out_path = save_middle_slice_overlay_plot(
        patient["tcga_id"],
        patient["subtype"],
        patient["study_date"],
        slug=patient["slug"],
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
