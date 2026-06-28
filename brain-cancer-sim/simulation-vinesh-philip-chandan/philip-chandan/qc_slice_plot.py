"""Save middle-slice PNGs with expert mask overlay for visual QC."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SIM_ROOT = PHILIP_CHANDAN_DIR.parent

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SIM_ROOT))

from nifti_extractor import extract_volume, load_expert_mask  # noqa: E402
from spike_paths import SPIKE_PATIENT, ensure_spike_dirs, slice_plot_path  # noqa: E402

MASK_OVERLAY_COLOR = "lime"


def pick_mask_z_index(mask: np.ndarray) -> int:
    """Z index with the most tumor voxels."""
    counts = [int((mask[z] > 0).sum()) for z in range(mask.shape[0])]
    if max(counts, default=0) == 0:
        return mask.shape[0] // 2
    return int(np.argmax(counts))


def save_slice_plot(
    volume: np.ndarray,
    mask: np.ndarray,
    slug: str,
    *,
    title: str | None = None,
    overlay: bool = True,
) -> Path:
    ensure_spike_dirs()
    out_path = slice_plot_path(slug, overlay=overlay)
    z_idx = pick_mask_z_index(mask)
    mr_slice = volume[z_idx]
    mask_slice = mask[z_idx] > 0

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(mr_slice, cmap="gray")
    if overlay and mask_slice.any():
        axis.contour(mask_slice, levels=[0.5], colors=MASK_OVERLAY_COLOR, linewidths=1.0)
    plot_title = title or f"{slug} z={z_idx}"
    axis.set_title(plot_title)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_spike_slice_plot(
    mr_path: Path,
    seg_path: Path,
    *,
    slug: str | None = None,
) -> Path:
    output_slug = slug or SPIKE_PATIENT["slug"]
    volume = extract_volume(mr_path)
    mask = load_expert_mask(seg_path, volume.shape)
    return save_slice_plot(volume, mask, output_slug)


def ensure_overlay_plot(slug: str) -> Path | None:
    """Return expert-mask overlay PNG, generating from raw extract if needed."""
    out_path = slice_plot_path(slug, overlay=True)
    if out_path.exists():
        return out_path

    npy_path = SIM_ROOT.parent / "data" / "processed" / "raw-extract-philip-chandan" / f"{slug}.npy"
    json_path = npy_path.with_suffix(".json")
    if not npy_path.exists() or not json_path.exists():
        return None

    import json

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    volume = np.load(npy_path)
    seg_path = meta.get("segmentation_path")
    if not seg_path:
        return None
    mask_path = Path(seg_path)
    if not mask_path.is_absolute():
        mask_path = SIM_ROOT.parent / mask_path
    if not mask_path.exists():
        return None
    mask = load_expert_mask(mask_path, volume.shape)
    return save_slice_plot(volume, mask, slug)


def main() -> None:
    from nifti_extractor import resolve_ucsf_paths

    patient = SPIKE_PATIENT
    patient_dir = SIM_ROOT.parent / "data" / "raw" / "ucsf_alptdg" / patient["patient_id"]
    mr_path, seg_path = resolve_ucsf_paths(patient_dir, patient.get("timepoint", "baseline"))
    out_path = save_spike_slice_plot(mr_path, seg_path, slug=patient["slug"])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
