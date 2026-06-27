"""Load Philip-Chandan raw extract and prepare PDE-ready input (Vinesh-owned)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, zoom
from skimage.filters import threshold_otsu

VINESH_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = VINESH_DIR.parent
sys.path.insert(0, str(SPIKE_ROOT))

from handoff_contract import (  # noqa: E402
    contract_version,
    max_shape,
    pde_input_spec,
    target_spacing_mm,
)
from spike_paths import (  # noqa: E402
    SPIKE_PATIENT,
    ensure_spike_dirs,
    pde_input_metadata,
    pde_input_npy,
    raw_extract_metadata,
    raw_extract_npy,
)


def load_raw_extract(slug: str | None = None) -> tuple[np.ndarray, dict]:
    name = slug or SPIKE_PATIENT["slug"]
    npy_path = raw_extract_npy(name)
    json_path = raw_extract_metadata(name)
    if not npy_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing raw extract. Ask Philip-Chandan to run export_raw_extract.py "
            f"(expected {npy_path} and {json_path})"
        )
    volume = np.load(npy_path)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    return volume, metadata


def _crop_or_pad_centered(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    center: tuple[float, ...],
    pad_value: float,
) -> np.ndarray:
    """Return a `target_shape` box of `volume` centered on `center` (voxel coords).

    Crops where the volume is larger than the target and pads with `pad_value`
    where it is smaller, so the tumor stays roughly centered regardless of how
    resampling changed the dimensions.
    """
    out = np.full(target_shape, pad_value, dtype=volume.dtype)
    for_slices_src: list[slice] = []
    for_slices_dst: list[slice] = []
    for axis, (size, tgt, c) in enumerate(zip(volume.shape, target_shape, center)):
        # Start of the source box so that `center` lands in the middle of target.
        start = int(round(c - tgt / 2))
        end = start + tgt
        # Clamp the source window to the array, tracking the matching dst window.
        src_start = max(start, 0)
        src_end = min(end, size)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        for_slices_src.append(slice(src_start, src_end))
        for_slices_dst.append(slice(dst_start, dst_end))
    out[tuple(for_slices_dst)] = volume[tuple(for_slices_src)]
    return out


def prepare_pde_input(
    volume: np.ndarray,
    spacing_mm: list[float],
    *,
    max_shape_xyz: tuple[int, int, int] | None = None,
    target_spacing: list[float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Resample, crop, and normalize a raw extract into solve_growth input.

    Pipeline (all targets read from handoff_contract.json, nothing hardcoded):
      1. Resample to isotropic `target_spacing` mm via scipy.ndimage.zoom.
      2. Normalize intensities into the contract `value_range` (min-max).
      3. Otsu-threshold (skimage) to separate tumor from background; keep the
         *continuous* normalized intensity inside the tumor and set background
         to `background_value`. Continuous (not binary) values matter: a binary
         field would sit at u=1 where the solver's logistic term rho*u*(1-u)=0,
         so the tumor would only diffuse, not grow.
      4. Crop/pad to `max_shape`, tumor roughly centered on its center of mass.

    Returns the float32 PDE volume and its new (isotropic) spacing.
    """
    pde_spec = pde_input_spec()
    shape_limit = tuple(max_shape_xyz or max_shape())
    spacing_target = target_spacing or target_spacing_mm()
    vmin_out, vmax_out = (float(v) for v in pde_spec["value_range"])
    background = float(pde_spec["background_value"])

    vol = np.asarray(volume, dtype=np.float32)

    # 1. Resample to isotropic target spacing. zoom factor > 1 upsamples.
    zoom_factors = [s / t for s, t in zip(spacing_mm, spacing_target)]
    resampled = zoom(vol, zoom_factors, order=1)  # linear interpolation

    # 2. Normalize intensities into the contract value range.
    rmin, rmax = float(resampled.min()), float(resampled.max())
    if rmax > rmin:
        norm = (resampled - rmin) / (rmax - rmin)
    else:
        norm = np.zeros_like(resampled)
    norm = norm * (vmax_out - vmin_out) + vmin_out

    # 3. Otsu segmentation: zero out background, keep continuous tumor density.
    finite_vals = norm[np.isfinite(norm)]
    if finite_vals.size and finite_vals.max() > finite_vals.min():
        thresh = threshold_otsu(finite_vals)
        tumor_mask = norm > thresh
    else:
        tumor_mask = np.zeros_like(norm, dtype=bool)
    segmented = np.where(tumor_mask, norm, background).astype(np.float32)

    # 4. Crop/pad to max_shape, centered on the tumor (fallback: array center).
    if tumor_mask.any():
        center = center_of_mass(tumor_mask.astype(np.float32))
    else:
        center = tuple(s / 2 for s in segmented.shape)
    pde_volume = _crop_or_pad_centered(segmented, shape_limit, center, background)

    return pde_volume, list(spacing_target)


def save_pde_input(
    volume: np.ndarray,
    spacing_mm: list[float],
    raw_metadata: dict,
    *,
    slug: str | None = None,
) -> tuple[Path, Path]:
    ensure_spike_dirs()
    name = slug or SPIKE_PATIENT["slug"]
    npy_path = pde_input_npy(name)
    json_path = pde_input_metadata(name)

    pde_spec = pde_input_spec()
    np.save(npy_path, volume.astype(np.float32))
    metadata = {
        "contract_version": contract_version(),
        "slug": name,
        "source_raw_extract": str(raw_extract_npy(name).relative_to(SPIKE_ROOT.parent)),
        "shape": list(volume.shape),
        "dtype": pde_spec["dtype"],
        "axis_order": pde_spec["axis_order"],
        "spacing_mm": spacing_mm,
        "value_range": pde_spec["value_range"],
        "background_value": pde_spec["background_value"],
        "tumor_burden_rule": pde_spec["tumor_burden_rule"],
        "value_semantics": {
            str(pde_spec["background_value"]): "background/healthy",
            ">0": "initial tumor burden",
        },
        "upstream": {
            "tcga_id": raw_metadata.get("tcga_id"),
            "study_date": raw_metadata.get("study_date"),
            "raw_contract_version": raw_metadata.get("contract_version"),
        },
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return npy_path, json_path


def main() -> None:
    raw_volume, raw_metadata = load_raw_extract()
    spacing_mm = raw_metadata["spacing_mm"]
    pde_volume, pde_spacing = prepare_pde_input(raw_volume, spacing_mm)
    npy_path, json_path = save_pde_input(pde_volume, pde_spacing, raw_metadata)
    print(f"Wrote {npy_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
