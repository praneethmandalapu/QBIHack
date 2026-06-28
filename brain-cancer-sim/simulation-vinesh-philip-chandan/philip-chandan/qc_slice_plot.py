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
from spike_paths import (  # noqa: E402
    REPO_ROOT,
    SPIKE_PATIENT,
    ensure_spike_dirs,
    resolve_raw_extract_metadata,
    resolve_raw_extract_npy,
    slice_plot_path,
)

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

    npy_path = resolve_raw_extract_npy(slug)
    json_path = resolve_raw_extract_metadata(slug)
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


def longitudinal_plot_path(patient_id: str) -> Path:
    return slice_plot_path(f"patient_{patient_id}", overlay=True).with_name(
        f"{patient_id}_longitudinal_mid-z-overlay.png"
    )


def save_longitudinal_overlay_plot(
    patient_id: str,
    *,
    baseline_slug: str | None = None,
    followup_slug: str | None = None,
    baseline_wt_mm3: float | None = None,
    followup_wt_mm3: float | None = None,
    interval_days: float | None = None,
) -> Path:
    """Side-by-side baseline vs follow-up T1ce with expert mask contours."""
    ensure_spike_dirs()
    out_path = longitudinal_plot_path(patient_id)

    baseline_npy = resolve_raw_extract_npy(baseline_slug or f"glioma_ucsf_{patient_id}_baseline")
    followup_npy = resolve_raw_extract_npy(followup_slug or f"glioma_ucsf_{patient_id}_followup")
    if not baseline_npy.is_file() or not followup_npy.is_file():
        raise FileNotFoundError(
            f"Missing raw extract for patient {patient_id}: {baseline_npy} / {followup_npy}"
        )

    import json

    baseline_vol = np.load(baseline_npy)
    followup_vol = np.load(followup_npy)
    baseline_meta = json.loads(baseline_npy.with_suffix(".json").read_text(encoding="utf-8"))
    followup_meta = json.loads(followup_npy.with_suffix(".json").read_text(encoding="utf-8"))

    def _mask_from_meta(volume: np.ndarray, meta: dict) -> np.ndarray:
        seg_path = Path(meta["segmentation_path"])
        if not seg_path.is_absolute():
            seg_path = REPO_ROOT / seg_path
        return load_expert_mask(seg_path, volume.shape)

    baseline_mask = _mask_from_meta(baseline_vol, baseline_meta)
    followup_mask = _mask_from_meta(followup_vol, followup_meta)
    z_idx = pick_mask_z_index(baseline_mask)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    panels = [
        (baseline_vol, baseline_mask, "Baseline (t1)", baseline_wt_mm3),
        (followup_vol, followup_mask, "Follow-up (t2)", followup_wt_mm3),
    ]
    for axis, (volume, mask, label, wt_mm3) in zip(axes, panels):
        slice_2d = volume[z_idx]
        mask_slice = mask[z_idx] > 0
        axis.imshow(slice_2d, cmap="gray")
        if mask_slice.any():
            axis.contour(mask_slice, levels=[0.5], colors=MASK_OVERLAY_COLOR, linewidths=1.0)
        title = f"{label}  z={z_idx}"
        if wt_mm3 is not None:
            title += f"\nWT {wt_mm3:,.0f} mm³"
        axis.set_title(title, fontsize=10)
        axis.axis("off")

    growth_note = ""
    if baseline_wt_mm3 is not None and followup_wt_mm3 is not None and baseline_wt_mm3 > 0:
        pct = 100.0 * (followup_wt_mm3 - baseline_wt_mm3) / baseline_wt_mm3
        growth_note = f"  |  ΔWT {followup_wt_mm3 - baseline_wt_mm3:+,.0f} mm³ ({pct:+.1f}%)"
    if interval_days is not None:
        growth_note += f"  |  {interval_days:.0f} d"

    fig.suptitle(f"Patient {patient_id}{growth_note}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def ensure_longitudinal_overlay_plot(
    patient_id: str,
    *,
    baseline_wt_mm3: float | None = None,
    followup_wt_mm3: float | None = None,
    interval_days: float | None = None,
    force: bool = False,
) -> Path | None:
    out_path = longitudinal_plot_path(patient_id)
    if out_path.exists() and not force:
        return out_path
    try:
        return save_longitudinal_overlay_plot(
            patient_id,
            baseline_wt_mm3=baseline_wt_mm3,
            followup_wt_mm3=followup_wt_mm3,
            interval_days=interval_days,
        )
    except (FileNotFoundError, ValueError, OSError):
        return None


def main() -> None:
    from nifti_extractor import resolve_ucsf_paths

    patient = SPIKE_PATIENT
    patient_dir = SIM_ROOT.parent / "data" / "raw" / "ucsf_alptdg" / patient["patient_id"]
    mr_path, seg_path = resolve_ucsf_paths(patient_dir, patient.get("timepoint", "baseline"))
    out_path = save_spike_slice_plot(mr_path, seg_path, slug=patient["slug"])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
